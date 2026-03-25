#!/usr/bin/env python3
"""
Run missing instances for a specific PPO-EDA config.
Usage: python main_run_config.py --config krbf__advnormalizedfitness__M2__L10__eps0p015__g0p005__ds0p01__dm0p01
"""

from __future__ import annotations

import argparse
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda


DEFAULTS = dict(
    seed=0,
    nb_instances_test=10,
    nb_restarts=10,
    budget=50000,
    visualization=False,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)

QUBO_PATTERN = re.compile(r"^puboi_evo_n_(?P<dim>\d+)_t_(?P<t>\d+)_i_(?P<i>\d+)\.json$")
NK_PATTERN = re.compile(r"^nk_(?P<dim>\d+)_(?P<t>\d+)_?(?P<i>\d+)\.txt$")
INSTANCE_DIR_RE = re.compile(r"^(?P<problem>QUBO|NK)_dim(?P<dim>\d+)_t(?P<t>\d+)$")


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _round_float(value, ndigits: int = 8):
    try:
        return round(float(value), ndigits)
    except Exception:
        return value


def _format_float(value, ndigits: int = 4) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value)


def _parse_summary_config(summary_path: Path):
    cfg = {}
    try:
        with open(summary_path, "r") as f:
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
                        if any(token in value for token in (".", "e", "E")):
                            parsed = float(value)
                        else:
                            parsed = int(value)
                    except ValueError:
                        parsed = value
                cfg[key] = parsed
    except OSError:
        return None
    return cfg


def _discover_qubo_instances(instances_root: Path, nb_instances: int):
    seen = {}
    for fname in os.listdir(instances_root):
        m = QUBO_PATTERN.match(fname)
        if not m:
            continue
        dim = int(m.group("dim"))
        t = int(m.group("t"))
        idx = int(m.group("i"))
        seen.setdefault((dim, t), set()).add(idx)

    instances = []
    for (dim, t), indices in sorted(seen.items()):
        idx_set = set(indices)
        has_zero_based = all(i in idx_set for i in range(nb_instances))
        has_one_based = all(i in idx_set for i in range(1, nb_instances + 1))
        if not (has_zero_based or has_one_based):
            continue
        instances.append(dict(name="QUBO", dim=dim, type_instance=t))
    return instances


def _discover_nk_instances(instances_root: Path, nb_instances: int):
    instances = []
    if not instances_root.is_dir():
        return instances
    for dim_dir in sorted(instances_root.iterdir()):
        if not dim_dir.is_dir() or not dim_dir.name.isdigit():
            continue
        dim = int(dim_dir.name)
        for t_dir in sorted(dim_dir.iterdir()):
            if not t_dir.is_dir() or not t_dir.name.isdigit():
                continue
            t = int(t_dir.name)
            indices = []
            for fname in os.listdir(t_dir):
                m = NK_PATTERN.match(fname)
                if not m:
                    continue
                indices.append(int(m.group("i")))
            if not indices:
                continue
            max_contig = 0
            for i in sorted(set(indices)):
                if i == max_contig:
                    max_contig += 1
                else:
                    break
            if max_contig < nb_instances:
                continue
            instances.append(dict(name="NK", dim=dim, type_instance=t))
    return instances


def _load_instances(problem_cfg, device):
    script_dir = os.path.abspath(os.path.dirname(__file__))
    name = problem_cfg["name"]
    dim = int(problem_cfg["dim"])
    type_instance = int(problem_cfg["type_instance"])

    if name == "QUBO":
        instance_path = os.path.join(script_dir, "instances", "QUBO") + os.sep
        tensor_Q_test = getTensorInstances_QUBO(
            instance_path,
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
            tensor_matrix_locus=None,
            tensor_matrix_contrib=None,
        )

    if name == "NK":
        D = 2
        vectorIndex = np.zeros((type_instance + 1))
        for vi in range(type_instance + 1):
            vectorIndex[vi] = D ** (type_instance - vi)
        vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(device)
        base_path = os.path.join(script_dir, "instances", "nk", str(dim), str(type_instance)) + os.sep
        return dict(
            type_problem="NK",
            dim=dim,
            type_instance=type_instance,
            tensor_Q_test=None,
            dim_variables=None,
            D=D,
            vectorIndex_th=vectorIndex_th,
            tensor_matrix_locus=None,
            tensor_matrix_contrib=None,
            nk_base_path=base_path,
        )

    raise ValueError(f"Unsupported problem {name}")


def _run_once(
    problem_ctx,
    kernel_name,
    advantage,
    M,
    lambda_,
    epsilon_svgd,
    gamma,
    decay_start_ratio,
    decay_min_factor,
    bandwith_kernel,
    device=None,
    nb_restarts=None,
):
    device = device or DEFAULTS["device"]
    nb_restarts = DEFAULTS["nb_restarts"] if nb_restarts is None else int(nb_restarts)

    kernel_config = {"name": kernel_name, "epsilon_svgd": epsilon_svgd, "gamma": gamma}
    if kernel_name in ("rbf", "pk") and bandwith_kernel is not None:
        kernel_config["bandwith_kernel"] = bandwith_kernel

    factory = FactoryStrategyEA()
    strategy = factory.createStrategyEA(
        "PPO-EDA",
        problem_ctx["dim"],
        lambda_,
        device,
        problem_ctx["dim_variables"],
        M,
        learning_rate=epsilon_svgd,
        epsilon_svgd=epsilon_svgd,
        enable_visualization=DEFAULTS["visualization"],
        svgd_gamma=gamma,
        decay_start_ratio=decay_start_ratio,
        decay_min_factor=decay_min_factor,
        decay_enabled=True,
        advantage_cfg=advantage,
        kernel_config=kernel_config,
        no_interact=False,
    ).to(device)

    if problem_ctx["type_problem"] == "QUBO":
        list_scores, history = get_Score_trajectoriesQUBO_cuda(
            strategy,
            problem_ctx["dim"],
            DEFAULTS["nb_instances_test"],
            nb_restarts,
            DEFAULTS["budget"],
            lambda_,
            problem_ctx["tensor_Q_test"],
            device,
            False,
            enable_visualization=False,
            return_history=True,
        )
    else:
        total_lambda = strategy.lambda_
        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK(
            problem_ctx["nk_base_path"],
            DEFAULTS["nb_instances_test"],
            nb_restarts,
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
            nb_restarts,
            DEFAULTS["budget"],
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
    avg_score = float(np.mean(scores_array))

    run_meta = dict(
        problem=problem_ctx["type_problem"],
        dim=problem_ctx["dim"],
        type_instance=problem_ctx["type_instance"],
        kernel=kernel_name,
        advantage=advantage,
        M=M,
        lambda_=lambda_,
        epsilon_svgd=epsilon_svgd,
        gamma=gamma,
        decay_start_ratio=decay_start_ratio,
        decay_min_factor=decay_min_factor,
        bandwith_kernel=bandwith_kernel,
        no_interact=False,
        avg_score=avg_score,
    )
    return avg_score, history, run_meta


def _save_history_csv(out_dir, problem_name, kernel_name, entry, config_name=None):
    history = entry["history"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
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
    csv_path = os.path.join(out_dir, "best_metrics.csv")
    with open(csv_path, "w") as f:
        f.write(
            "step,best_fitness,avg_hamming,avg_l1,avg_entropy,"
            "mean,median,std,2%,5%,10%,25%,50%,75%,90%,95%,98%\n"
        )
        for (step, bf, ham, l1, ent, mean, median, std, p2, p5, p10, p25, p50, p75, p90, p95, p98) in rows:
            f.write(
                f"{step},{bf},{ham},{l1},{ent},"
                f"{mean},{median},{std},{p2},{p5},{p10},{p25},{p50},{p75},{p90},{p95},{p98}\n"
            )

    summary_path = os.path.join(out_dir, "best_summary.txt")
    meta = entry["meta"]
    with open(summary_path, "w") as f:
        if config_name:
            f.write(f"ConfigName: {config_name}\n")
        f.write(f"Problem: {problem_name}\n")
        f.write(f"Kernel: {kernel_name}\n")
        f.write(f"Advantage: {meta['advantage']}\n")
        f.write(f"M: {meta['M']}\n")
        f.write(f"lambda: {meta['lambda_']}\n")
        f.write(f"epsilon_svgd: {meta['epsilon_svgd']}\n")
        f.write(f"gamma: {meta['gamma']}\n")
        f.write(f"decay_start_ratio: {meta['decay_start_ratio']}\n")
        f.write(f"decay_min_factor: {meta['decay_min_factor']}\n")
        f.write(f"bandwith_kernel: {meta['bandwith_kernel']}\n")
        f.write(f"no_interact: {meta['no_interact']}\n")
        f.write(f"avg_score: {meta['avg_score']}\n")


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "cuda out of memory" in msg or "outofmemoryerror" in msg


def main():
    parser = argparse.ArgumentParser(description="Run missing instances for a specific PPO-EDA config.")
    parser.add_argument("--config", type=str, required=True, help="Config name (e.g., krbf__advnormalizedfitness__M2__L10__eps0p015__g0p005__ds0p01__dm0p01)")
    parser.add_argument("--outdir", type=str, default=None, help="Root output dir (default: results/config).")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_root = args.outdir or os.path.join(repo_root, "results", "config")
    config_name = args.config
    config_dir = os.path.join(out_root, config_name)

    # Check if config exists
    if not os.path.isdir(config_dir):
        raise SystemExit(f"Config directory not found: {config_dir}")

    # Extract params from existing summary (if any)
    params = None
    for child in sorted(Path(config_dir).iterdir()):
        if not child.is_dir():
            continue
        summary_path = child / "best_summary.txt"
        if summary_path.is_file():
            cfg = _parse_summary_config(summary_path)
            if cfg:
                # Extract parameters from summary
                params = dict(
                    kernel=str(cfg.get("kernel", "rbf")).lower(),
                    advantage=str(cfg.get("advantage", "normalizedfitness")),
                    M=int(cfg.get("m", 1)),
                    lambda_=int(cfg.get("lambda", 10)),
                    epsilon_svgd=float(cfg.get("epsilon_svgd", 0.01)),
                    gamma=float(cfg.get("gamma", 0.005)),
                    decay_start_ratio=float(cfg.get("decay_start_ratio", 0.01)),
                    decay_min_factor=float(cfg.get("decay_min_factor", 0.01)),
                    bandwith_kernel=cfg.get("bandwith_kernel"),
                )
                break

    if params is None:
        raise SystemExit(f"No completed instances found in {config_dir} to extract parameters from")

    print(f"[CONFIG] {config_name}")
    print(f"  Parameters extracted:")
    print(f"    Kernel: {params['kernel']}, Advantage: {params['advantage']}")
    print(f"    M: {params['M']}, Lambda: {params['lambda_']}, Epsilon: {params['epsilon_svgd']}, Gamma: {params['gamma']}")
    print()

    instances_root = Path(repo_root) / "source_code" / "instances"
    qubo_instances = _discover_qubo_instances(instances_root / "QUBO", DEFAULTS["nb_instances_test"])
    nk_instances = _discover_nk_instances(instances_root / "nk", DEFAULTS["nb_instances_test"])
    instances = qubo_instances + nk_instances
    if not instances:
        raise SystemExit("No QUBO/NK instances found.")

    _set_seeds(DEFAULTS["seed"])

    # Find pending instances
    pending_instances = []
    skipped_instances = []
    for inst in instances:
        inst_name = f"{inst['name']}_dim{inst['dim']}_t{inst['type_instance']}"
        inst_dir = os.path.join(config_dir, inst_name)
        summary_path = os.path.join(inst_dir, "best_summary.txt")
        legacy_summary = os.path.join(inst_dir, f"{inst['name']}_{params['kernel']}_best_summary.txt")
        
        if os.path.isfile(summary_path) or os.path.isfile(legacy_summary):
            cfg_path = Path(summary_path) if os.path.isfile(summary_path) else Path(legacy_summary)
            cfg = _parse_summary_config(cfg_path)
            avg_score = cfg.get("avg_score") if cfg else None
            if avg_score is not None:
                skipped_instances.append(inst_name)
                continue
        pending_instances.append(inst)

    for inst_name in skipped_instances:
        print(f"  -> skip {inst_name} (already done)")

    if not pending_instances:
        print("  -> all instances complete, nothing to do.")
        return

    print(f"  -> {len(pending_instances)} instances to run\n")

    # Separate QUBO and NK instances
    qubo_pending = [inst for inst in pending_instances if inst['name'] == 'QUBO']
    nk_pending = [inst for inst in pending_instances if inst['name'] == 'NK']
    
    qubo_top1_count = 0
    start_time = time.time()
    
    # Run QUBO instances first
    for inst in qubo_pending:
        inst_name = f"{inst['name']}_dim{inst['dim']}_t{inst['type_instance']}"
        inst_dir = os.path.join(config_dir, inst_name)
        problem_ctx = _load_instances(inst, DEFAULTS["device"])
        print(f"  -> run {inst_name}")
        t0 = time.time()
        nb_restarts = DEFAULTS["nb_restarts"]
        success = False
        while nb_restarts > 0 and not success:
            try:
                avg_score, history, meta = _run_once(
                    problem_ctx,
                    params["kernel"],
                    params["advantage"],
                    params["M"],
                    params["lambda_"],
                    params["epsilon_svgd"],
                    params["gamma"],
                    params["decay_start_ratio"],
                    params["decay_min_factor"],
                    params.get("bandwith_kernel"),
                    device=DEFAULTS["device"],
                    nb_restarts=nb_restarts,
                )
                success = True
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if not _is_cuda_oom(exc):
                    raise
                nb_restarts -= 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if nb_restarts > 0:
                    print(f"     [OOM] retry with nb_restarts={nb_restarts}.")
                else:
                    print("     [OOM] nb_restarts=0, skip instance.")

        if not success:
            continue
        dt = time.time() - t0
        print(f"     avg_score={avg_score:.6f} | runtime={dt:.2f}s")
        _save_history_csv(
            inst_dir,
            inst["name"],
            params["kernel"],
            {"history": history, "meta": meta},
            config_name=config_name,
        )
        qubo_top1_count += 1

    # Check if we should skip NK
    if qubo_top1_count < 3 and nk_pending:
        print(f"\n  [SKIP NK] Only {qubo_top1_count} QUBO run (threshold: 3)")
        for inst in nk_pending:
            inst_name = f"{inst['name']}_dim{inst['dim']}_t{inst['type_instance']}"
            print(f"  -> skip {inst_name} (insufficient QUBO runs)")
    else:
        # Run NK instances
        for inst in nk_pending:
            inst_name = f"{inst['name']}_dim{inst['dim']}_t{inst['type_instance']}"
            inst_dir = os.path.join(config_dir, inst_name)
            problem_ctx = _load_instances(inst, DEFAULTS["device"])
            print(f"  -> run {inst_name}")
            t0 = time.time()
            nb_restarts = DEFAULTS["nb_restarts"]
            success = False
            while nb_restarts > 0 and not success:
                try:
                    avg_score, history, meta = _run_once(
                        problem_ctx,
                        params["kernel"],
                        params["advantage"],
                        params["M"],
                        params["lambda_"],
                        params["epsilon_svgd"],
                        params["gamma"],
                        params["decay_start_ratio"],
                        params["decay_min_factor"],
                        params.get("bandwith_kernel"),
                        device=DEFAULTS["device"],
                        nb_restarts=nb_restarts,
                    )
                    success = True
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    if not _is_cuda_oom(exc):
                        raise
                    nb_restarts -= 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if nb_restarts > 0:
                        print(f"     [OOM] retry with nb_restarts={nb_restarts}.")
                    else:
                        print("     [OOM] nb_restarts=0, skip instance.")

            if not success:
                continue
            dt = time.time() - t0
            print(f"     avg_score={avg_score:.6f} | runtime={dt:.2f}s")
            _save_history_csv(
                inst_dir,
                inst["name"],
                params["kernel"],
                {"history": history, "meta": meta},
                config_name=config_name,
            )

    elapsed = time.time() - start_time
    print(f"\n[DONE] missing instances complete")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
