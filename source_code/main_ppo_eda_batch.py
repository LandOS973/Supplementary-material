#!/usr/bin/env python3
"""Run PPO-EDA in batch: best config per instance, budgets 20k/30k, modes normal/no_interact/decay."""

from __future__ import annotations

import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from environment.blockwise import get_Score_trajectoriesBLOCK_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda


DEFAULT_BUDGETS = [20000, 30000]
DEFAULTS = dict(
    seed=0,
    nb_instances_test=10,
    nb_restarts=10,
    visualization=False,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def _parse_summary_config(summary_path: Path) -> dict | None:
    cfg: dict = {}
    try:
        with summary_path.open() as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if not value:
                    continue
                lowered = value.lower()
                if lowered in ("none", "null", "n/a"):
                    parsed = None
                elif lowered in ("true", "false"):
                    parsed = lowered == "true"
                else:
                    try:
                        if "." in value:
                            parsed = float(value)
                        else:
                            parsed = int(value)
                    except ValueError:
                        parsed = value
                cfg[key] = parsed
    except OSError:
        return None
    return cfg


def _find_best_kernel_summary(summary_dir: Path, type_problem: str) -> Tuple[str | None, dict | None]:
    best_kernel = None
    best_cfg = None
    best_score = None
    maximize = type_problem in ("NK", "BLOCK")
    summary_files = list(summary_dir.glob(f"{type_problem}_*_best_summary.txt"))
    for path in summary_files:
        cfg = _parse_summary_config(path)
        if not cfg:
            continue
        kernel = cfg.get("kernel")
        avg_score = cfg.get("avg_score")
        if kernel is None or avg_score is None:
            continue
        if best_score is None:
            best_score = avg_score
            best_kernel = str(kernel).lower()
            best_cfg = cfg
            continue
        if maximize and avg_score > best_score:
            best_score = avg_score
            best_kernel = str(kernel).lower()
            best_cfg = cfg
        elif not maximize and avg_score < best_score:
            best_score = avg_score
            best_kernel = str(kernel).lower()
            best_cfg = cfg
    return best_kernel, best_cfg


def _resolve_best_config(best_cfg: dict, kernel_name: str, repo_root: str, mode: str) -> dict:
    try:
        kernel_cfg = _load_kernel_config(kernel_name, repo_root)
    except Exception:
        kernel_cfg = {}
    epsilon_svgd = best_cfg.get("epsilon_svgd")
    if epsilon_svgd is None:
        epsilon_svgd = kernel_cfg.get("epsilon_svgd")
    if epsilon_svgd is None:
        epsilon_svgd = 0.01
    gamma = best_cfg.get("gamma")
    if gamma is None:
        gamma = kernel_cfg.get("gamma")
    if gamma is None:
        gamma = 0.001
    bandwith_kernel = best_cfg.get("bandwith_kernel")
    if bandwith_kernel is None:
        bandwith_kernel = kernel_cfg.get("bandwith_kernel") or (kernel_cfg.get("params") or {}).get("bandwith_kernel")

    advantage = best_cfg.get("advantage") or "peragentrankweighted"
    M = int(best_cfg.get("m") or best_cfg.get("M") or 1)
    lambda_ = int(best_cfg.get("lambda") or best_cfg.get("lambda_") or 1)
    no_interact = bool(best_cfg.get("no_interact") or False)
    if mode == "no_interact":
        no_interact = True
    elif mode == "decay":
        no_interact = False
    else:
        no_interact = False

    if mode == "decay":
        decay_start_ratio = best_cfg.get("decay_start_ratio")
        decay_min_factor = best_cfg.get("min_factor") or best_cfg.get("decay_min_factor")
        if decay_start_ratio is None:
            decay_start_ratio = 0.0
        if decay_min_factor is None:
            decay_min_factor = 0.05
        decay_enabled = True
    else:
        decay_start_ratio = 0.8
        decay_min_factor = 0.1
        decay_enabled = False

    return dict(
        kernel_name=str(kernel_name).lower(),
        advantage=advantage,
        M=M,
        lambda_=lambda_,
        epsilon_svgd=float(epsilon_svgd),
        gamma=float(gamma),
        bandwith_kernel=bandwith_kernel,
        no_interact=no_interact,
        decay_start_ratio=float(decay_start_ratio),
        decay_min_factor=float(decay_min_factor),
        decay_enabled=decay_enabled,
    )


def _write_history_csv(out_dir: Path, filename: str, history: dict, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = history.get("runtime") or list(range(1, len(history.get("best_fitness", [])) + 1))
    score_mean = history.get("score_mean", [])
    score_median = history.get("score_median", [])
    score_std = history.get("score_std", [])
    score_p2 = history.get("score_p2", [])
    score_p5 = history.get("score_p5", [])
    score_p10 = history.get("score_p10", [])
    score_p25 = history.get("score_p25", [])
    score_p50 = history.get("score_p50", [])
    score_p75 = history.get("score_p75", [])
    score_p90 = history.get("score_p90", [])
    score_p95 = history.get("score_p95", [])
    score_p98 = history.get("score_p98", [])
    rows = zip(
        runtime,
        history.get("best_fitness", []),
        history.get("avg_hamming", []),
        history.get("avg_l1", []),
        history.get("avg_entropy", []),
        score_mean,
        score_median,
        score_std,
        score_p2,
        score_p5,
        score_p10,
        score_p25,
        score_p50,
        score_p75,
        score_p90,
        score_p95,
        score_p98,
    )
    csv_path = out_dir / filename
    with csv_path.open("w") as f:
        for key, value in meta.items():
            f.write(f"# {key}: {value}\n")
        f.write(
            "step,best_fitness,avg_hamming,avg_l1,avg_entropy,"
            "mean,median,std,2%,5%,10%,25%,50%,75%,90%,95%,98%\n"
        )
        for (step, bf, ham, l1, ent, mean, median, std, p2, p5, p10, p25, p50, p75, p90, p95, p98) in rows:
            f.write(
                f"{step},{bf},{ham},{l1},{ent},"
                f"{mean},{median},{std},{p2},{p5},{p10},{p25},{p50},{p75},{p90},{p95},{p98}\n"
            )


def _should_skip_budget_file(budget_dir: Path, filename: str) -> bool:
    target = budget_dir / filename
    if not target.exists():
        return False
    try:
        return target.stat().st_size > 0
    except OSError:
        return True


def _discover_instances(repo_root: Path) -> List[dict]:
    exp_root = repo_root / "results" / "experiments"
    if not exp_root.exists():
        return []
    instances: List[dict] = []
    seen = set()
    for entry in exp_root.iterdir():
        if not entry.is_dir():
            continue
        match = re.match(r"^(?P<name>.+)_dim(?P<dim>\d+)_t(?P<t>\d+)$", entry.name)
        if not match:
            continue
        name = match.group("name")
        dim = int(match.group("dim"))
        type_instance = int(match.group("t"))
        key = (name, dim, type_instance)
        if key in seen:
            continue
        seen.add(key)
        instances.append(dict(name=name, dim=dim, type_instance=type_instance))
    return sorted(instances, key=lambda d: (d["name"], d["dim"], d["type_instance"]))


def _load_problem_context(problem_cfg: dict) -> dict:
    script_dir = Path(__file__).resolve().parent
    name = problem_cfg["name"]
    dim = int(problem_cfg["dim"])
    type_instance = int(problem_cfg["type_instance"])
    device = DEFAULTS["device"]

    if name == "QUBO":
        instance_path = script_dir / "instances" / "QUBO"
        tensor_Q_test = getTensorInstances_QUBO(
            str(instance_path) + os.sep,
            DEFAULTS["nb_instances_test"],
            DEFAULTS["nb_restarts"],
            dim,
            type_instance,
            device,
            "test",
        )
        return dict(
            type_problem="QUBO",
            dim=dim,
            type_instance=type_instance,
            tensor_Q_test=tensor_Q_test,
            dim_variables=None,
            D=None,
            vectorIndex_th=None,
            nk_base_path=None,
            block_size=None,
        )

    if name == "NK":
        D = 2
        vectorIndex = np.zeros((type_instance + 1))
        for vi in range(type_instance + 1):
            vectorIndex[vi] = D ** (type_instance - vi)
        vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(device)
        nk_base_path = script_dir / "instances" / "nk" / str(dim) / str(type_instance)
        return dict(
            type_problem="NK",
            dim=dim,
            type_instance=type_instance,
            tensor_Q_test=None,
            dim_variables=None,
            D=D,
            vectorIndex_th=vectorIndex_th,
            nk_base_path=nk_base_path,
            block_size=None,
        )

    if name == "BLOCK":
        block_size = type_instance
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if dim % block_size != 0:
            raise ValueError(f"dim={dim} must be divisible by block_size={block_size}")
        return dict(
            type_problem="BLOCK",
            dim=dim,
            type_instance=type_instance,
            tensor_Q_test=None,
            dim_variables=None,
            D=None,
            vectorIndex_th=None,
            nk_base_path=None,
            block_size=block_size,
        )

    raise ValueError(f"Unsupported problem {name}")


def _run_once(problem_ctx: dict, params: dict, budget: int) -> Tuple[np.ndarray, dict]:
    device = DEFAULTS["device"]
    kernel_config = {
        "name": params["kernel_name"],
        "epsilon_svgd": params["epsilon_svgd"],
        "gamma": params["gamma"],
    }
    if params["kernel_name"] in ("rbf", "pk") and params.get("bandwith_kernel") is not None:
        kernel_config["bandwith_kernel"] = params["bandwith_kernel"]

    factory = FactoryStrategyEA()
    strategy = factory.createStrategyEA(
        "PPO-EDA",
        problem_ctx["dim"],
        params["lambda_"],
        device,
        problem_ctx["dim_variables"],
        params["M"],
        learning_rate=params["epsilon_svgd"],
        epsilon_svgd=params["epsilon_svgd"],
        enable_visualization=DEFAULTS["visualization"],
        svgd_gamma=params["gamma"],
        decay_start_ratio=params["decay_start_ratio"],
        decay_min_factor=params["decay_min_factor"],
        decay_enabled=params["decay_enabled"],
        advantage_cfg=params["advantage"],
        kernel_config=kernel_config,
        no_interact=params["no_interact"],
    ).to(device)

    if problem_ctx["type_problem"] == "QUBO":
        list_scores, history = get_Score_trajectoriesQUBO_cuda(
            strategy,
            problem_ctx["dim"],
            DEFAULTS["nb_instances_test"],
            DEFAULTS["nb_restarts"],
            budget,
            params["lambda_"],
            problem_ctx["tensor_Q_test"],
            device,
            False,
            enable_visualization=False,
            return_history=True,
        )
    elif problem_ctx["type_problem"] == "BLOCK":
        list_scores, history = get_Score_trajectoriesBLOCK_cuda(
            strategy,
            problem_ctx["dim"],
            problem_ctx["block_size"],
            DEFAULTS["nb_instances_test"],
            DEFAULTS["nb_restarts"],
            budget,
            params["lambda_"],
            device,
            False,
            enable_visualization=False,
            return_history=True,
        )
    else:
        total_lambda = params["lambda_"] * params["M"]
        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK(
            str(problem_ctx["nk_base_path"]) + os.sep,
            DEFAULTS["nb_instances_test"],
            DEFAULTS["nb_restarts"],
            total_lambda,
            problem_ctx["dim"],
            problem_ctx["D"],
            problem_ctx["type_instance"],
            device,
        )
        list_scores, history = get_Score_trajectoriesNK_cuda(
            strategy,
            problem_ctx["dim"],
            problem_ctx["type_instance"],
            problem_ctx["D"],
            DEFAULTS["nb_instances_test"],
            DEFAULTS["nb_restarts"],
            budget,
            total_lambda,
            problem_ctx["vectorIndex_th"],
            tensor_matrix_locus,
            tensor_matrix_contrib,
            device,
            False,
            enable_visualization=False,
            return_history=True,
        )

    scores_array = (
        np.asarray(list_scores)
        if isinstance(list_scores, (list, tuple, np.ndarray))
        else (list_scores.detach().cpu().numpy() if torch.is_tensor(list_scores) else np.asarray(list_scores))
    )
    return scores_array, history


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    instances = _discover_instances(repo_root)
    if not instances:
        print(f"[WARN] Aucun dossier d'instances dans {repo_root / 'results' / 'experiments'}")
        return

    _set_seeds(DEFAULTS["seed"])

    modes = ["normal", "no_interact", "decay"]
    start_all = time.time()

    for inst in instances:
        problem_name = inst["name"]
        dim = inst["dim"]
        type_instance = inst["type_instance"]
        instance_name = f"{problem_name}_dim{dim}_t{type_instance}"
        print(f"[INFO] Instance {instance_name}")

        problem_ctx = _load_problem_context(inst)
        for mode in modes:
            summary_dir = repo_root / "results" / "experiments" / instance_name
            if mode == "no_interact":
                summary_dir = summary_dir / "no_interact"
            best_kernel, best_cfg = _find_best_kernel_summary(summary_dir, problem_ctx["type_problem"])
            if best_kernel is None or best_cfg is None:
                print(f"[WARN] Pas de resume pour {instance_name} ({mode}) dans {summary_dir}")
                continue

            if mode == "decay":
                # use best config from decay summaries
                decay_dir = repo_root / "results" / "experiments" / instance_name / "decay"
                best_kernel, best_cfg = _find_best_kernel_summary(decay_dir, problem_ctx["type_problem"])
                if best_kernel is None or best_cfg is None:
                    print(f"[WARN] Pas de resume decay pour {instance_name} (decay)")
                    continue

            params = _resolve_best_config(best_cfg, best_kernel, str(repo_root), mode)
            print(
                f"  -> mode={mode} kernel={params['kernel_name']} M={params['M']} lambda={params['lambda_']} "
                f"eps={params['epsilon_svgd']} gamma={params['gamma']}"
            )

            for budget in DEFAULT_BUDGETS:
                budget_dir = repo_root / "results" / "experiments" / instance_name / str(budget)
                filename = (
                    "decay.csv"
                    if mode == "decay"
                    else ("no_interact.csv" if mode == "no_interact" else "interact.csv")
                )
                if _should_skip_budget_file(budget_dir, filename):
                    print(f"    [SKIP] budget={budget} {filename} deja present")
                    continue
                _set_seeds(DEFAULTS["seed"])
                t0 = time.time()
                scores, history = _run_once(problem_ctx, params, budget)
                avg_score = float(np.mean(scores)) if len(scores) else float("nan")
                dt = time.time() - t0
                print(f"    budget={budget} avg_score={avg_score:.6f} runtime={dt:.2f}s")

                meta = dict(
                    epsilon_svgd=params["epsilon_svgd"],
                    lambda_=params["lambda_"],
                    gamma=params["gamma"],
                    decay_start_ratio=params["decay_start_ratio"],
                    decay_min_factor=params["decay_min_factor"],
                    decay_enabled=params["decay_enabled"],
                    kernel=params["kernel_name"],
                    advantage=params["advantage"],
                    M=params["M"],
                    mode=mode,
                    budget=budget,
                )
                _write_history_csv(budget_dir, filename, history, meta=meta)

    print(f"[DONE] batch finished in {time.time() - start_all:.2f}s")


if __name__ == "__main__":
    main()
