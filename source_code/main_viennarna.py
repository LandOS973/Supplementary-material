"""
Hydra runner for PPO-EDA (SVGD-EDA) on ViennaRNA inverse folding.

This runner uses ViennaRNA ensemble defect during optimization:
`score = -ensemble_defect(target)`.

Outputs:
results/config/<ConfigName>/viennarna/<TargetName>/
  - best_metrics.csv
  - raw_scores.csv
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.metrics import MetricsCalculator
from environment.visualization import render_agent_dashboard, render_svgd_field_plot
from problems.viennarna import (
    ETERNA100_TSV_URL,
    RNA_ALPHABET,
    load_target_from_eterna100,
    normalize_target_struct,
    tokens_to_rna_strings,
)

try:
    import RNA
except Exception:                                                    
    RNA = None


DEFAULT_TARGET_NAME = "Kudzu"
DEFAULT_TARGET_STRUCT = "(" * 25 + "." * 50 + ")" * 25

_WORKER_TARGET_STRUCT = None


def _init_viennarna_worker(target_struct: str) -> None:
    global _WORKER_TARGET_STRUCT
    _WORKER_TARGET_STRUCT = target_struct


def _score_sequence_worker(sequence: str) -> float:
    if _WORKER_TARGET_STRUCT is None:
        raise RuntimeError("Worker target structure was not initialized.")
    fc = RNA.fold_compound(sequence)
    fc.pf()
    defect = float(fc.ensemble_defect(_WORKER_TARGET_STRUCT))
    return -defect


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
    return s.strip().replace(".", "p").replace("-", "m").replace("/", "_").replace(" ", "_")


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
        f"target{_slugify(params['target_name'])}",
        f"n{_slugify(params['dim'])}",
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


def _tensor_to_rna_strings(tensor_solution: torch.Tensor) -> list[str]:
    sequences = tensor_solution[..., 0].detach().cpu().numpy().astype(np.int64)
    flat_sequences = sequences.reshape(-1, sequences.shape[-1])
    return tokens_to_rna_strings(flat_sequences)


def get_Score_trajectories_viennarna_cuda(
    target_struct,
    strategy,
    nb_instances,
    budget,
    size_popEA,
    device,
    verbose,
    num_workers=None,
    enable_visualization=False,
    enable_pairwise_visualization=True,
    name_file=None,
    return_history=False,
):
    """
    ViennaRNA online evaluation loop with fast MFE proxy scoring.

    The strategy is updated online:
    sample -> evaluate with fast proxy -> updateDistribution.
    """

    del size_popEA                                                        

    if nb_instances <= 0:
        raise ValueError("nb_instances must be >= 1.")

    target_struct = normalize_target_struct(target_struct)
    target_len = len(target_struct)

    strategy.reset_learned_parameters(nb_instances=nb_instances)

    best_score = torch.full((nb_instances,), -float("inf"), device=device)
    size_pop = int(strategy.lambda_)
    if size_pop <= 0:
        raise ValueError("strategy.lambda_ must be > 0.")

    nb_iterations = budget // size_pop
    stochastic_remainder = budget - (nb_iterations * size_pop)

    runtime_steps = []
    best_fitness_history = []
    avg_hamming_history = []
    avg_l1_history = []
    avg_entropy_history = []
    score_mean_history = []
    score_median_history = []
    score_std_history = []
    score_p2_history = []
    score_p5_history = []
    score_p10_history = []
    score_p25_history = []
    score_p50_history = []
    score_p75_history = []
    score_p90_history = []
    score_p95_history = []
    score_p98_history = []
    avg_js_history = []
    avg_l2_history = []
    agent_fitness_history = []
    hamming_pairwise_history = []
    js_pairwise_history = []
    l2_pairwise_history = []
    l1_pairwise_history = []
    entropy_agent_history = []
    avg_kernel_value_history = []
    avg_kernel_grad_history = []

    if name_file is not None:
        with open(name_file, "w", encoding="utf-8") as f_results:
            f_results.write(
                "runtime, mean, median, std, 2%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 98%\n"
            )

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    num_workers = max(1, int(num_workers))
    _init_viennarna_worker(target_struct)

    pool = None
    if num_workers > 1:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=num_workers,
            initializer=_init_viennarna_worker,
            initargs=(target_struct,),
        )

    def _evaluate_population(tensor_solution: torch.Tensor) -> torch.Tensor:
        pop_size = tensor_solution.size(1)
        sequences = _tensor_to_rna_strings(tensor_solution)
        if any(len(seq) != target_len for seq in sequences):
            raise ValueError(
                f"Sequence length mismatch with target length={target_len}. "
                f"Got lengths sample={sorted(set(len(seq) for seq in sequences))[:5]}"
            )

        if pool is None:
            scores = [_score_sequence_worker(seq) for seq in sequences]
        else:
            chunksize = max(1, len(sequences) // (num_workers * 8))
            scores = pool.map(_score_sequence_worker, sequences, chunksize=chunksize)

        scores = np.asarray(scores, dtype=np.float32).reshape(nb_instances, pop_size)
        return torch.from_numpy(scores).to(device=device, dtype=torch.float32)

    def _record_history(step, avg_hamming=0.0, avg_l1=0.0, avg_entropy=0.0):
        best_np = best_score.detach().cpu().numpy()
        mean = float(np.mean(best_np))
        median = float(np.percentile(best_np, 50))
        std = float(np.std(best_np))
        p2 = float(np.percentile(best_np, 2))
        p5 = float(np.percentile(best_np, 5))
        p10 = float(np.percentile(best_np, 10))
        p25 = float(np.percentile(best_np, 25))
        p75 = float(np.percentile(best_np, 75))
        p90 = float(np.percentile(best_np, 90))
        p95 = float(np.percentile(best_np, 95))
        p98 = float(np.percentile(best_np, 98))

        runtime_steps.append(int(step))
        best_fitness_history.append(mean)
        avg_hamming_history.append(float(avg_hamming))
        avg_l1_history.append(float(avg_l1))
        avg_entropy_history.append(float(avg_entropy))
        score_mean_history.append(mean)
        score_median_history.append(median)
        score_std_history.append(std)
        score_p2_history.append(p2)
        score_p5_history.append(p5)
        score_p10_history.append(p10)
        score_p25_history.append(p25)
        score_p50_history.append(median)
        score_p75_history.append(p75)
        score_p90_history.append(p90)
        score_p95_history.append(p95)
        score_p98_history.append(p98)

        if name_file is not None and step % 100 == 0:
            with open(name_file, "a", encoding="utf-8") as f_results:
                f_results.write(
                    f"{step},{mean},{median},{std},"
                    f"{p2},{p5},{p10},{p25},{median},"
                    f"{p75},{p90},{p95},{p98}\n"
                )

    use_tqdm = bool(verbose)
    pbar = tqdm(range(nb_iterations)) if use_tqdm else range(nb_iterations)
    collect_dashboard = bool(enable_visualization) and hasattr(strategy, "agents")
    collect_pairwise_metrics = collect_dashboard and bool(enable_pairwise_visualization)
    metrics = MetricsCalculator() if collect_dashboard else None

    try:
        for epoch in pbar:
            tensor_solution = strategy.sample_solutions()
            tensor_score = _evaluate_population(tensor_solution)
            batch_score_std = float(torch.std(tensor_score).item())

            current_score = torch.max(tensor_score, dim=1).values
            best_score = torch.where(current_score > best_score, current_score, best_score)

            if hasattr(strategy, "decay_svgd_gamma"):
                strategy.decay_svgd_gamma(epoch, nb_iterations)
            strategy.updateDistribution(tensor_solution, tensor_score)

            step = (epoch + 1) * size_pop
            avg_hamming = 0.0
            avg_l1 = 0.0
            avg_entropy = 0.0
            if collect_dashboard and metrics is not None:
                avg_entropy, per_agent_entropy = metrics.compute_entropy(strategy.agents)
                entropy_agent_history.append(per_agent_entropy if per_agent_entropy is not None else None)
                if collect_pairwise_metrics:
                    avg_hamming, hamming_pairwise = metrics.compute_average_hamming(strategy.agents)
                    avg_js, js_pairwise = metrics.compute_average_js(strategy.agents)
                    avg_l1, l1_pairwise = metrics.compute_l1_distance(strategy.agents)
                    _avg_l2, _pair_l2 = 0.0, None
                    avg_js_history.append(float(avg_js))
                    avg_l2_history.append(float(_avg_l2))
                    hamming_pairwise_history.append(hamming_pairwise.tolist() if hamming_pairwise is not None else None)
                    js_pairwise_history.append(js_pairwise.tolist() if js_pairwise is not None else None)
                    l2_pairwise_history.append(_pair_l2.tolist() if _pair_l2 is not None else None)
                    l1_pairwise_history.append(l1_pairwise.tolist() if l1_pairwise is not None else None)

                fitness_snapshot_fn = getattr(strategy, "get_agent_fitness_snapshot", None)
                if callable(fitness_snapshot_fn):
                    agent_fitness_history.append(fitness_snapshot_fn(tensor_score))
                else:
                    agent_lambdas = getattr(strategy, "agent_lambdas", None)
                    per_agent_fitness = []
                    if isinstance(agent_lambdas, (list, tuple)) and len(agent_lambdas) > 0:
                        start_idx = 0
                        for agent_lambda in agent_lambdas:
                            end_idx = min(start_idx + int(agent_lambda), tensor_score.size(1))
                            if end_idx <= start_idx:
                                break
                            agent_scores = tensor_score[:, start_idx:end_idx]
                            agent_best_values, _ = torch.max(agent_scores, dim=1)
                            per_agent_fitness.append(float(agent_best_values.mean().item()))
                            start_idx = end_idx
                    agent_fitness_history.append(per_agent_fitness)

                kernel_stats_fn = getattr(strategy, "get_latest_kernel_metrics", None)
                kernel_stats = kernel_stats_fn() if callable(kernel_stats_fn) else None
                if kernel_stats:
                    avg_kernel_value_history.append(float(kernel_stats.get("avg_kernel_value", 0.0)))
                    avg_kernel_grad_history.append(float(kernel_stats.get("avg_kernel_grad", 0.0)))
                else:
                    avg_kernel_value_history.append(0.0)
                    avg_kernel_grad_history.append(0.0)

            _record_history(step, avg_hamming=avg_hamming, avg_l1=avg_l1, avg_entropy=avg_entropy)

            if use_tqdm:
                pbar.set_postfix(
                    bestScore=float(torch.mean(best_score).item()),
                    current_score=float(torch.mean(current_score).item()),
                    score_std=batch_score_std,
                )

        if stochastic_remainder > 0:
            tensor_solution = strategy.sample_solutions()[:, :stochastic_remainder, :, :]
            tensor_score = _evaluate_population(tensor_solution)
            current_score = torch.max(tensor_score, dim=1).values
            best_score = torch.where(current_score > best_score, current_score, best_score)
            if collect_dashboard:
                avg_js_history.append(avg_js_history[-1] if avg_js_history else 0.0)
                avg_l2_history.append(avg_l2_history[-1] if avg_l2_history else 0.0)
                hamming_pairwise_history.append(hamming_pairwise_history[-1] if hamming_pairwise_history else None)
                js_pairwise_history.append(js_pairwise_history[-1] if js_pairwise_history else None)
                l2_pairwise_history.append(l2_pairwise_history[-1] if l2_pairwise_history else None)
                l1_pairwise_history.append(l1_pairwise_history[-1] if l1_pairwise_history else None)
                entropy_agent_history.append(entropy_agent_history[-1] if entropy_agent_history else None)
                agent_fitness_history.append(agent_fitness_history[-1] if agent_fitness_history else [])
                avg_kernel_value_history.append(avg_kernel_value_history[-1] if avg_kernel_value_history else 0.0)
                avg_kernel_grad_history.append(avg_kernel_grad_history[-1] if avg_kernel_grad_history else 0.0)
            _record_history(budget)

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if collect_dashboard and hasattr(strategy, "agents"):
        theta_history = None
        theta_history_fn = getattr(strategy, "get_theta_history", None)
        if callable(theta_history_fn):
            theta_history = theta_history_fn()
        num_agents = len(strategy.agents)
        render_agent_dashboard(
            runtime_steps,
            avg_hamming_history,
            avg_js_history,
            agent_fitness_history,
            num_agents,
            theta_history,
            None,
            hamming_pairwise_history,
            js_pairwise_history,
            avg_l2_history,
            l2_pairwise_history,
            avg_l1_history,
            l1_pairwise_history,
            avg_entropy_history,
            entropy_agent_history,
            avg_kernel_value_history,
            avg_kernel_grad_history,
        )

        svgd_snapshot_fn = getattr(strategy, "get_svgd_field_snapshot", None)
        if callable(svgd_snapshot_fn):
            snapshot = svgd_snapshot_fn()
            if snapshot:
                render_svgd_field_plot(snapshot)

    best_np = best_score.detach().cpu().numpy()
    if return_history:
        history = dict(
            runtime=runtime_steps,
            best_fitness=best_fitness_history,
            avg_hamming=avg_hamming_history,
            avg_l1=avg_l1_history,
            avg_entropy=avg_entropy_history,
            score_mean=score_mean_history,
            score_median=score_median_history,
            score_std=score_std_history,
            score_p2=score_p2_history,
            score_p5=score_p5_history,
            score_p10=score_p10_history,
            score_p25=score_p25_history,
            score_p50=score_p50_history,
            score_p75=score_p75_history,
            score_p90=score_p90_history,
            score_p95=score_p95_history,
            score_p98=score_p98_history,
        )
        return best_np, history
    return best_np


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if RNA is None:
        raise RuntimeError(
            "Import `RNA` failed. Install ViennaRNA Python bindings first "
            "(e.g. `pip install ViennaRNA` or your `officievienna` package). "
            "If build fails, install SWIG and ViennaRNA development headers."
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"running on device: {device}")

    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    def agent_val(key):
        try:
            return OmegaConf.select(cfg, f"agent.{key}")
        except Exception:
            return None

    target_name = str(OmegaConf.select(cfg, "problem.target_name") or DEFAULT_TARGET_NAME)
    target_source = str(OmegaConf.select(cfg, "problem.target_source") or ETERNA100_TSV_URL)
    target_struct_cfg = OmegaConf.select(cfg, "problem.target_struct")
    if target_struct_cfg:
        target_struct = normalize_target_struct(str(target_struct_cfg))
        target_resolved_name = "cfg_target_struct"
        print(f"[ViennaRNA] using target from cfg.problem.target_struct (len={len(target_struct)})")
    else:
        target_struct, target_resolved_name = load_target_from_eterna100(
            target_name=target_name,
            source=target_source,
            fallback_target=DEFAULT_TARGET_STRUCT,
            verbose=bool(cfg.get("verbose", True)),
        )

    dim = len(target_struct)
    alphabet_size = 4
    num_workers_cfg = OmegaConf.select(cfg, "problem.num_workers")
    if num_workers_cfg is None:
        num_workers_cfg = cfg.get("num_workers")
    num_workers = max(1, int(num_workers_cfg)) if num_workers_cfg is not None else max(1, mp.cpu_count() - 1)

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
    kernel_cfg["debug_svgd"] = bool(OmegaConf.select(cfg, "debug_svgd") or cfg.get("debug_svgd", False))
    kernel_cfg["debug_every"] = int(OmegaConf.select(cfg, "debug_every") or cfg.get("debug_every") or 25)

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
        f"Config: target={target_resolved_name} len={dim} alphabet={alphabet_size} workers={num_workers} | "
        f"M={M} lambda={lambda_} eps={epsilon_svgd} gamma={svgd_gamma} | "
        f"kernel={kernel_name} advantage={advantage_cfg} decay={decay_enabled} "
        f"visualization={visualization_enabled}"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
        target_name=target_resolved_name,
        dim=dim,
    )
    config_name = _build_config_name(params)
    out_dir = os.path.join(repo_root, "results", "config", config_name, "viennarna", _slugify(target_resolved_name))
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

    scores_array, history = get_Score_trajectories_viennarna_cuda(
        target_struct=target_struct,
        strategy=strategy,
        nb_instances=nb_instances_test,
        budget=budget,
        size_popEA=lambda_,
        device=device,
        verbose=verbose,
        num_workers=num_workers,
        enable_visualization=visualization_enabled,
        name_file=None,
        return_history=True,
    )

    avg_score = float(np.mean(scores_array)) if len(scores_array) else float("nan")
    print(f"average_test_score: {avg_score}")
    print("Objective: score = -ensemble_defect(target), higher is better (best possible: 0.0).")

    _save_history_csv(out_dir, history)
    _save_raw_scores_csv(out_dir, scores_array)


if __name__ == "__main__":
    main()
