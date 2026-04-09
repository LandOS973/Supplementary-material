"""
Hydra runner for PPO-EDA (SVGD-EDA) on DesignBench.
Stores best_metrics.csv + raw_scores.csv under:
results/config/<ConfigName>/designbench/<TaskName>/
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.designbench import get_Score_trajectories_designbench_cuda
from problems.designbench import (
    infer_task_space,
    load_designbench_task,
    oracle_sanity_check,
    resolve_designbench_problem,
)


def _load_kernel_config(kernel_name: str, repo_root: str) -> dict:
    kernel_dir = Path(repo_root) / "config" / "kernel"
    kernel_path = kernel_dir / f"{kernel_name}.yaml"
    if not kernel_path.exists():
        available = ", ".join(sorted(p.stem for p in kernel_dir.glob("*.yaml"))) if kernel_dir.exists() else "none"
        raise FileNotFoundError(
            f"Kernel config '{kernel_name}' introuvable dans {kernel_dir}. Kernels disponibles: {available}"
        )
    cfg = OmegaConf.load(str(kernel_path))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) or {}
    if "name" not in cfg_dict:
        cfg_dict["name"] = kernel_name
    return cfg_dict


def _slugify(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        s = f"{float(value):.6f}".rstrip("0").rstrip(".")
        if s in ("", "-0"):
            s = "0"
    else:
        s = str(value)
    return s.strip().replace(".", "p").replace("-", "m").replace("/", "_")


def _build_config_name(params: dict) -> str:
    parts = [
        f"k{_slugify(params['kernel'])}",
        f"adv{_slugify(params['advantage'])}",
        f"M{_slugify(params['M'])}",
        f"L{_slugify(params['lambda_'])}",
        f"eps{_slugify(params['epsilon_svgd'])}",
        f"g{_slugify(params['gamma'])}",
        f"ds{_slugify(params['decay_start_ratio'])}",
        f"dm{_slugify(params['decay_min_factor'])}",
    ]
    if params.get("bandwith_kernel") is not None:
        parts.append(f"bw{_slugify(params['bandwith_kernel'])}")
    return "__".join(parts)


def _save_history_csv(out_dir: str, history: dict) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runtime = history.get("runtime") or list(range(1, len(history.get("best_fitness", [])) + 1))
    rows = zip(
        runtime,
        history.get("best_fitness", []),
        history.get("avg_hamming", []),
        history.get("avg_l1", []),
        history.get("avg_entropy", []),
        history.get("score_mean", []),
        history.get("score_median", []),
        history.get("score_std", []),
        history.get("score_p2", []),
        history.get("score_p5", []),
        history.get("score_p10", []),
        history.get("score_p25", []),
        history.get("score_p50", []),
        history.get("score_p75", []),
        history.get("score_p90", []),
        history.get("score_p95", []),
        history.get("score_p98", []),
    )
    csv_path = os.path.join(out_dir, "best_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "step,best_fitness,avg_hamming,avg_l1,avg_entropy,"
            "mean,median,std,2%,5%,10%,25%,50%,75%,90%,95%,98%\n"
        )
        for (step, bf, ham, l1, ent, mean, median, std, p2, p5, p10, p25, p50, p75, p90, p95, p98) in rows:
            f.write(
                f"{step},{bf},{ham},{l1},{ent},"
                f"{mean},{median},{std},{p2},{p5},{p10},{p25},{p50},{p75},{p90},{p95},{p98}\n"
            )


def _save_raw_scores_csv(out_dir: str, scores_array):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    raw_path = os.path.join(out_dir, "raw_scores.csv")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write("score\n")
        for val in scores_array:
            f.write(f"{float(val)}\n")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"running on device: {device}")

    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    def agent_val(key):
        try:
            return OmegaConf.select(cfg, f"agent.{key}")
        except Exception:
            return None

    problem_cfg = resolve_designbench_problem(cfg)
    task_name = problem_cfg["task_name"]
    dim = problem_cfg["dim"]
    alphabet_size = problem_cfg["alphabet_size"]
    oracle_batch_size = problem_cfg["oracle_batch_size"]
    if problem_cfg["used_fallback"]:
        print("[WARN] problem config is not designbench; using defaults task=GFP-v0 dim=237 alphabet=20")

    nb_instances_test = int(cfg.nb_instances_test)
    if nb_instances_test <= 0:
        raise ValueError("nb_instances_test must be >= 1.")
    seed = int(cfg.seed)
    verbose = bool(cfg.get("verbose", True))
    budget = int(cfg.get("budget", 10000))
    visualization_enabled = bool(cfg.get("visualization", True))
    lambda_ = int(agent_val("lambda") or cfg.get("lambda") or cfg.get("lambda_") or 10)
    M = int(agent_val("M") or cfg.get("M") or 1)

    kernel_name = str(agent_val("kernel") or cfg.get("kernel") or "rbf").lower()
    kernel_cfg = _load_kernel_config(kernel_name, repo_root)
    bandwith_override = agent_val("bandwith_kernel") or cfg.get("bandwith_kernel")
    if bandwith_override is not None:
        kernel_cfg["bandwith_kernel"] = bandwith_override
    kernel_cfg["debug_svgd"] = True
    kernel_cfg["debug_every"] = 1

    epsilon_svgd = float(agent_val("epsilon_svgd") or cfg.get("epsilon_svgd") or kernel_cfg.get("epsilon_svgd") or 0.1)
    svgd_gamma = float(agent_val("gamma") or cfg.get("gamma") or kernel_cfg.get("gamma") or 0.01)
    advantage_cfg = agent_val("advantage") or cfg.get("advantage") or "globalrankweighted"
    if isinstance(advantage_cfg, DictConfig):
        advantage_cfg = OmegaConf.to_container(advantage_cfg, resolve=True)

    decay_enabled = bool(agent_val("decay") or cfg.get("decay") or False)
    decay_default_start_ratio = 0.0 if decay_enabled else 0.8
    decay_default_min_factor = 0.05 if decay_enabled else 0.1
    decay_start_ratio = float(agent_val("decay_start_ratio") or cfg.get("decay_start_ratio") or decay_default_start_ratio)
    decay_min_factor = float(
        agent_val("min_factor")
        or agent_val("decay_min_factor")
        or cfg.get("min_factor")
        or cfg.get("decay_min_factor")
        or decay_default_min_factor
    )

    no_interact = bool(agent_val("no_interact") or cfg.get("no_interact") or False)
    no_repulsion = bool(agent_val("no_repulsion") or cfg.get("no_repulsion") or False)

    print(
        f"Config: task={task_name} dim={dim} alphabet={alphabet_size} | "
        f"M={M} lambda={lambda_} eps={epsilon_svgd} gamma={svgd_gamma} | "
        f"kernel={kernel_name} advantage={advantage_cfg} decay={decay_enabled} "
        f"visualization={visualization_enabled}"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    task = load_designbench_task(task_name, use_cuda=torch.cuda.is_available())

    inferred_dim, inferred_alpha, inferred_mode, discrete_ok = infer_task_space(task, dim, alphabet_size)
    if inferred_dim != dim or inferred_alpha != alphabet_size:
        if verbose:
            print(
                f"[DesignBench] inferred space: dim={inferred_dim} alphabet={inferred_alpha} "
                f"(config was dim={dim} alphabet={alphabet_size})"
            )
        dim = inferred_dim
        alphabet_size = inferred_alpha
    if not discrete_ok:
        raise RuntimeError(
            "DesignBench task appears continuous/non-integer. Current SVGD-EDA runner expects discrete tokens/onehot."
        )

    oracle_sanity_check(task, dim=dim, alphabet_size=alphabet_size, verbose=verbose)
    dim_variables = [alphabet_size for _ in range(dim)]
    factory = FactoryStrategyEA()

    params = dict(
        kernel=kernel_name,
        advantage=advantage_cfg,
        M=M,
        lambda_=lambda_,
        epsilon_svgd=epsilon_svgd,
        gamma=svgd_gamma,
        decay_start_ratio=decay_start_ratio,
        decay_min_factor=decay_min_factor,
        bandwith_kernel=kernel_cfg.get("bandwith_kernel"),
    )
    config_name = _build_config_name(params)
    out_dir = os.path.join(repo_root, "results", "config", config_name, "designbench", task_name)
    print(f"Output dir: {out_dir}")

    strategy = factory.createStrategyEA(
        "PPO-EDA",
        dim,
        lambda_,
        device,
        dim_variables,
        M,
        learning_rate=epsilon_svgd,
        epsilon_svgd=epsilon_svgd,
        enable_visualization=visualization_enabled,
        svgd_gamma=svgd_gamma,
        decay_start_ratio=decay_start_ratio,
        decay_min_factor=decay_min_factor,
        decay_enabled=decay_enabled,
        advantage_cfg=advantage_cfg,
        kernel_config=kernel_cfg,
        no_interact=no_interact,
        no_repulsion=no_repulsion,
        is_nk3=False,
    ).to(device)

    scores_array, history = get_Score_trajectories_designbench_cuda(
        task,
        strategy,
        nb_instances_test,
        budget,
        lambda_,
        device,
        verbose=verbose,
        oracle_batch_size=oracle_batch_size,
        alphabet_size=alphabet_size,
        enable_visualization=visualization_enabled,
        name_file=None,
        return_history=True,
        warm_start=False,
    )

    avg_score = float(np.mean(scores_array)) if len(scores_array) else float("nan")
    print(f"average_test_score: {avg_score}")

    _save_history_csv(out_dir, history)
    _save_raw_scores_csv(out_dir, scores_array)


if __name__ == "__main__":
    main()
