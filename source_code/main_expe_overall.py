#!/usr/bin/env python3
"""
Run PPO-EDA (decay mode) over all QUBO/NK instances for multiple config grids.
Stores per-instance history/summary under results/config/<ConfigName>/<InstanceName>/.
Aggregates ranks across instances into an Excel summary.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
    budget=10000,
    visualization=False,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)

DEFAULT_GRIDS = [
    dict(
        kernels=["fr"],
        advantages=["globalrankweighted"],
        M_values=[4,8],
        lambda_values=[5,13,24],
        epsilon_svgd=[0.01,0.03,0.06],
        gamma=[0.001,0.005,0.007],
        decay_start_ratio=[0.01],
        decay_min_factor=[0.01],
        bandwith_kernel=[None],
    )
]


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
    s = s.strip().replace(".", "p").replace("-", "m").replace("/", "_")
    return s


def _build_config_name(prefix: str | None, params: dict) -> str:
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
    if prefix:
        return f"{prefix}__" + "__".join(parts)
    return "__".join(parts)


def _expand_grid(grid: dict):
    if "configs" in grid and grid["configs"]:
        for cfg in grid["configs"]:
            params = {
                "kernel": cfg["kernel"],
                "advantage": cfg["advantage"],
                "M": int(cfg["M"]),
                "lambda_": int(cfg["lambda"]),
                "epsilon_svgd": float(cfg["epsilon_svgd"]),
                "gamma": float(cfg["gamma"]),
                "decay_start_ratio": float(cfg["decay_start_ratio"]),
                "decay_min_factor": float(cfg["decay_min_factor"]),
                "bandwith_kernel": cfg.get("bandwith_kernel"),
            }
            cfg_name = _build_config_name(None, params)
            yield cfg_name, params
        return

    kernels = grid.get("kernels", ["rbf"])
    advantages = grid.get("advantages", ["peragentrankweighted"])
    M_values = grid.get("M_values", [1])
    lambda_values = grid.get("lambda_values", [1])
    epsilon_svgd = grid.get("epsilon_svgd", [0.01])
    gamma = grid.get("gamma", [0.001])
    decay_start_ratio = grid.get("decay_start_ratio", [0.8])
    decay_min_factor = grid.get("decay_min_factor", [0.1])
    bandwith_kernel = grid.get("bandwith_kernel", [None])

    for (kernel, advantage, M, lambda_, eps, gam, ds, dm, bw) in itertools.product(
        kernels,
        advantages,
        M_values,
        lambda_values,
        epsilon_svgd,
        gamma,
        decay_start_ratio,
        decay_min_factor,
        bandwith_kernel,
    ):
        params = dict(
            kernel=str(kernel).lower(),
            advantage=str(advantage),
            M=int(M),
            lambda_=int(lambda_),
            epsilon_svgd=float(eps),
            gamma=float(gam),
            decay_start_ratio=float(ds),
            decay_min_factor=float(dm),
            bandwith_kernel=bw,
        )
        cfg_name = _build_config_name(None, params)
        yield cfg_name, params


def _load_grids():
    return DEFAULT_GRIDS


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
    median_score = float(np.percentile(scores_array, 50))
    std_score = float(np.std(scores_array))
    p2 = float(np.percentile(scores_array, 2))
    p5 = float(np.percentile(scores_array, 5))
    p10 = float(np.percentile(scores_array, 10))
    p25 = float(np.percentile(scores_array, 25))
    p50 = float(np.percentile(scores_array, 50))
    p75 = float(np.percentile(scores_array, 75))
    p90 = float(np.percentile(scores_array, 90))
    p95 = float(np.percentile(scores_array, 95))
    p98 = float(np.percentile(scores_array, 98))

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
        median_score=median_score,
        std_score=std_score,
        p2=p2,
        p5=p5,
        p10=p10,
        p25=p25,
        p50=p50,
        p75=p75,
        p90=p90,
        p95=p95,
        p98=p98,
    )
    return avg_score, history, run_meta


def _save_history_csv(out_dir, problem_name, kernel_name, entry, ranking=None, config_name=None):
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
        f.write(f"median_score: {meta['median_score']}\n")
        f.write(f"std_score: {meta['std_score']}\n")
        f.write(
            "percentiles: "
            f"2%={meta['p2']}, 5%={meta['p5']}, 10%={meta['p10']}, 25%={meta['p25']}, "
            f"50%={meta['p50']}, 75%={meta['p75']}, 90%={meta['p90']}, 95%={meta['p95']}, 98%={meta['p98']}\n"
        )
        if ranking:
            best_algo, best_score, my_rank, n_rank, my_pct = ranking[:5]
            if best_algo is not None and n_rank:
                pct_str = f"{my_pct:.1f}%" if my_pct is not None else "n/a"
                f.write(f"ranking_best_algo: {best_algo}\n")
                f.write(f"ranking_best_score: {best_score}\n")
                f.write(f"ranking_my_rank: {my_rank}/{n_rank} ({pct_str})\n")
                if my_pct is not None:
                    f.write(f"ranking_my_percent: {my_pct}\n")
        else:
            f.write("ranking: unavailable\n")


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


def _rank_vs_global_ranking_excluding_ppo(
    repo_root: str, problem: str, dim: int, type_instance: int, my_score: float, exclude_algo: str = "PPO-EDA"
):
    problem = (problem or "").upper()
    if problem in ("QUBO", "UBQP"):
        filename = f"UBQP_N_{dim}_K_{type_instance}_ranks.csv"
    elif problem == "NK":
        filename = f"NK_N_{dim}_K_{type_instance}_ranks.csv"
    else:
        return None, None, None, 0, None

    path = os.path.join(repo_root, "additional_results", "global_ranking", filename)
    if not os.path.isfile(path):
        return None, None, None, 0, None

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = [row for row in reader if row]
        if not rows:
            return None, None, None, 0, None

        header = [h.strip().lower() for h in rows[0]]
        rows = rows[1:]

        algo_candidates = ["name_algo", "algo", "algorithm", "name"]
        score_candidates = ["score", "best_score", "value", "objective", "obj"]

        def find_idx(cands):
            for c in cands:
                if c in header:
                    return header.index(c)
            return None

        idx_algo = find_idx(algo_candidates)
        idx_score = find_idx(score_candidates)
        if idx_score is None:
            return None, None, None, 0, None

        entries = []
        for r in rows:
            if idx_score >= len(r):
                continue
            try:
                s = float(r[idx_score])
            except Exception:
                continue
            name = r[idx_algo] if (idx_algo is not None and idx_algo < len(r)) else "unknown"
            if str(name).strip().lower() == exclude_algo.lower():
                continue
            entries.append((name, s))

        if not entries:
            return None, None, None, 0, None

        scores_only = [s for _, s in entries]
        n = len(scores_only)

        frac_pos = sum(1 for s in scores_only if s > 0) / max(1, n)
        flip_sign = (frac_pos > 0.8 and my_score < 0)

        def to_cmp(v):
            return (-v) if flip_sign else v

        best_algo, best_score = max(entries, key=lambda t: t[1])
        best_cmp = best_score
        my_cmp = to_cmp(my_score)
        count_gt = sum(1 for s in scores_only if s > my_cmp)
        my_rank = 1 + count_gt
        my_rank = min(max(1, my_rank), n)
        my_percentile = 100.0 * (n - my_rank + 1) / n if n > 0 else None
        return best_algo, best_score, my_rank, n, my_percentile, my_cmp, best_cmp

    except Exception:
        return None, None, None, 0, None, None, None


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "cuda out of memory" in msg or "outofmemoryerror" in msg


def _collect_config_stats(config_dir: str, config_name: str, params: dict, repo_root: str):
    rows = []
    config_path = Path(config_dir)
    if not config_path.is_dir():
        return dict(
            config_name=config_name,
            kernel=params["kernel"],
            advantage=params["advantage"],
            M=params["M"],
            lambda_=params["lambda_"],
            epsilon_svgd=_round_float(params["epsilon_svgd"]),
            gamma=_round_float(params["gamma"]),
            decay_start_ratio=_round_float(params["decay_start_ratio"]),
            decay_min_factor=_round_float(params["decay_min_factor"]),
            mean_rank=None,
            median_rank=None,
            std_percent=None,
            top1_count=0,
            top3_count=0,
            top5_count=0,
            top10_count=0,
            top_1_nk=0,
            top_1_qubo=0,
            win_rate_mean=None,
            mean_hamming_norm=None,
            mean_l1_norm=None,
            n_instances=0,
            n_ranked=0,
        )

    for child in sorted(config_path.iterdir()):
        if not child.is_dir():
            continue
        match = INSTANCE_DIR_RE.match(child.name)
        if not match:
            continue
        problem = match.group("problem")
        dim = int(match.group("dim"))
        t = int(match.group("t"))
        summary_path = child / "best_summary.txt"
        if not summary_path.is_file():
            legacy = child / f"{problem}_{params['kernel']}_best_summary.txt"
            if legacy.is_file():
                summary_path = legacy
            else:
                continue
        metrics_path = child / "best_metrics.csv"
        if not metrics_path.is_file():
            legacy_metrics = child / f"{problem}_{params['kernel']}_best_metrics.csv"
            if legacy_metrics.is_file():
                metrics_path = legacy_metrics
            else:
                metrics_path = None
        cfg = _parse_summary_config(summary_path)
        avg_score = cfg.get("avg_score") if cfg else None
        if avg_score is None:
            continue
        try:
            avg_score = float(avg_score)
        except Exception:
            continue
        best_algo, best_score, my_rank, n_rank, my_pct, my_cmp, best_cmp = _rank_vs_global_ranking_excluding_ppo(
            repo_root, problem, dim, t, avg_score
        )
        win_rate = None
        if my_rank is not None and n_rank:
            win_rate = (n_rank - my_rank) / n_rank
        hamming_norm = None
        l1_norm = None
        if metrics_path is not None and metrics_path.is_file():
            try:
                with open(metrics_path, "r") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                if len(lines) >= 2:
                    header = lines[0].split(",")
                    last = lines[-1].split(",")
                    idx_ham = header.index("avg_hamming") if "avg_hamming" in header else None
                    idx_l1 = header.index("avg_l1") if "avg_l1" in header else None
                    if idx_ham is not None and idx_ham < len(last):
                        hamming_val = float(last[idx_ham])
                        hamming_norm = hamming_val / dim if hamming_val > 1 else hamming_val
                    if idx_l1 is not None and idx_l1 < len(last):
                        l1_val = float(last[idx_l1])
                        l1_norm = l1_val / dim if l1_val > 1 else l1_val
            except Exception:
                hamming_norm = None
                l1_norm = None
        rows.append(
            dict(
                problem=problem,
                dim=dim,
                type_instance=t,
                avg_score=avg_score,
                rank=my_rank,
                percent=my_pct,
                top1_count=1 if my_rank == 1 else 0,
                top3_count=1 if my_rank is not None and my_rank <= 3 else 0,
                top5_count=1 if my_rank is not None and my_rank <= 5 else 0,
                top10_count=1 if my_rank is not None and my_rank <= 10 else 0,
                ranking_best_algo=best_algo,
                ranking_best_score=best_score,
                n_rank=n_rank,
                win_rate=win_rate,
                hamming_norm=hamming_norm,
                l1_norm=l1_norm,
            )
        )

    n_instances = len(rows)
    ranks = [r["rank"] for r in rows if r["rank"] is not None]
    percents = [r["percent"] for r in rows if r["percent"] is not None]
    win_rates = [r["win_rate"] for r in rows if r.get("win_rate") is not None]
    hamming_vals = [r["hamming_norm"] for r in rows if r.get("hamming_norm") is not None]
    l1_vals = [r["l1_norm"] for r in rows if r.get("l1_norm") is not None]
    n_ranked = len(ranks)

    mean_rank = float(np.mean(ranks)) if ranks else None
    median_rank = float(np.median(ranks)) if ranks else None
    std_percent = float(np.std(percents)) if len(percents) > 1 else 0.0 if percents else None
    top1_count = sum(1 for r in rows if r.get("top1_count"))
    top3_count = sum(1 for r in rows if r.get("top3_count"))
    top5_count = sum(1 for r in rows if r.get("top5_count"))
    top10_count = sum(1 for r in rows if r.get("top10_count"))
    top_1_nk = sum(1 for r in rows if r.get("top1_count") and r.get("problem") == "NK")
    top_1_qubo = sum(1 for r in rows if r.get("top1_count") and r.get("problem") == "QUBO")
    win_rate_mean = float(np.mean(win_rates)) if win_rates else None
    mean_hamming_norm = float(np.mean(hamming_vals)) if hamming_vals else None
    mean_l1_norm = float(np.mean(l1_vals)) if l1_vals else None

    return dict(
        config_name=config_name,
        kernel=params["kernel"],
        advantage=params["advantage"],
        M=params["M"],
        lambda_=params["lambda_"],
        epsilon_svgd=_round_float(params["epsilon_svgd"]),
        gamma=_round_float(params["gamma"]),
        decay_start_ratio=_round_float(params["decay_start_ratio"]),
        decay_min_factor=_round_float(params["decay_min_factor"]),
        mean_rank=mean_rank,
        median_rank=median_rank,
        std_percent=std_percent,
        top1_count=top1_count,
        top3_count=top3_count,
        top5_count=top5_count,
        top10_count=top10_count,
        top_1_nk=top_1_nk,
        top_1_qubo=top_1_qubo,
        win_rate_mean=win_rate_mean,
        mean_hamming_norm=mean_hamming_norm,
        mean_l1_norm=mean_l1_norm,
        n_instances=n_instances,
        n_ranked=n_ranked,
    )


def main():
    parser = argparse.ArgumentParser(description="Overall PPO-EDA decay grid (QUBO + NK).")
    parser.add_argument("--outdir", type=str, default=None, help="Root output dir (default: results/config).")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_root = args.outdir or os.path.join(repo_root, "results", "config")
    Path(out_root).mkdir(parents=True, exist_ok=True)

    grids = _load_grids()

    instances_root = Path(repo_root) / "source_code" / "instances"
    qubo_instances = _discover_qubo_instances(instances_root / "QUBO", DEFAULTS["nb_instances_test"])
    nk_instances = _discover_nk_instances(instances_root / "nk", DEFAULTS["nb_instances_test"])
    instances = qubo_instances + nk_instances
    if not instances:
        raise SystemExit("Aucune instance QUBO/NK compatible avec nb_instances_test.")

    _set_seeds(DEFAULTS["seed"])

    start_all = time.time()
    for grid in grids:
        for config_name, params in _expand_grid(grid):
            config_dir = os.path.join(out_root, config_name)
            print(f"[CONFIG] {config_name}")

            pending_instances = []
            skipped_instances = []
            for inst in instances:
                inst_name = f"{inst['name']}_dim{inst['dim']}_t{inst['type_instance']}"
                inst_dir = os.path.join(config_dir, inst_name)
                summary_path = os.path.join(inst_dir, "best_summary.txt")
                legacy_summary = os.path.join(
                    inst_dir, f"{inst['name']}_{params['kernel']}_best_summary.txt"
                )
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
                print("  -> already complete, skipping.")
                continue

            # Separate QUBO and NK instances
            qubo_pending = [inst for inst in pending_instances if inst['name'] == 'QUBO']
            nk_pending = [inst for inst in pending_instances if inst['name'] == 'NK']
            
            # Track TOP 1 count in QUBO
            qubo_top1_count = 0
            
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
                ranking = _rank_vs_global_ranking_excluding_ppo(
                    repo_root, inst["name"], inst["dim"], inst["type_instance"], avg_score
                )
                _save_history_csv(
                    inst_dir,
                    inst["name"],
                    params["kernel"],
                    {"history": history, "meta": meta},
                    ranking=ranking,
                    config_name=config_name,
                )
                if ranking and ranking[2] == 1:
                    print("     -> TOP 1")
                    qubo_top1_count += 1

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
                    ranking = _rank_vs_global_ranking_excluding_ppo(
                        repo_root, inst["name"], inst["dim"], inst["type_instance"], avg_score
                    )
                    _save_history_csv(
                        inst_dir,
                        inst["name"],
                        params["kernel"],
                        {"history": history, "meta": meta},
                        ranking=ranking,
                        config_name=config_name,
                    )
                    if ranking and ranking[2] == 1:
                        print("     -> TOP 1")

            # After each config, print stats
            stats = _collect_config_stats(config_dir, config_name, params, repo_root)
            print(f"\n  *** SUMMARY FOR CONFIG: {config_name} ***")
            print(f"  NOMBRE DE TOP 1 : {stats['top1_count']} (NK: {stats['top_1_nk']}, QUBO: {stats['top_1_qubo']})")
            print(f"  NOMBRE DE TOP 3 : {stats['top3_count']}")
            print(f"  NOMBRE DE TOP 5 : {stats['top5_count']}")
            print(f"  NOMBRE DE TOP 10 : {stats['top10_count']}")
            print(f"  Instances: {stats['n_ranked']}/{stats['n_instances']}")
            print()

    print(f"[DONE] experiments complete")
    print(f"Elapsed: {time.time() - start_all:.2f}s")


if __name__ == "__main__":
    main()
