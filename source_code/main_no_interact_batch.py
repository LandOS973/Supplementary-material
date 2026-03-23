#!/usr/bin/env python3
"""Run PPO-EDA in no_interact mode for a given config across all instances (batch budget)."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.nk import get_Score_trajectoriesNK_cuda, getTensorInstances_NK
from environment.qubo import get_Score_trajectoriesQUBO_cuda
from main_expe_overall import (
    DEFAULTS,
    _discover_nk3_instances,
    _discover_nk_instances,
    _discover_qubo_instances,
    _is_cuda_oom,
    _load_instances,
    _rank_vs_global_ranking_excluding_ppo,
    _save_history_csv,
    _set_seeds,
)


RUN_BUDGET = 50000
NO_INTERACT_KERNEL = "no_interact"


def _parse_config_name(config_name: str) -> dict:
    parts = config_name.split("__")
    out = {
        "kernel": None,
        "advantage": None,
        "M": None,
        "lambda_": None,
        "epsilon_svgd": None,
        "gamma": None,
        "decay_start_ratio": None,
        "decay_min_factor": None,
    }

    def parse_float(token: str):
        token = token.replace("p", ".").replace("m", "-")
        try:
            return float(token)
        except Exception:
            return None

    for p in parts:
        if p.startswith("k"):
            out["kernel"] = p[1:]
        elif p.startswith("adv"):
            out["advantage"] = p[3:]
        elif p.startswith("M"):
            try:
                out["M"] = int(p[1:])
            except Exception:
                pass
        elif p.startswith("L"):
            try:
                out["lambda_"] = int(p[1:])
            except Exception:
                pass
        elif p.startswith("eps"):
            out["epsilon_svgd"] = parse_float(p[3:])
        elif p.startswith("g"):
            out["gamma"] = parse_float(p[1:])
        elif p.startswith("ds"):
            out["decay_start_ratio"] = parse_float(p[2:])
        elif p.startswith("dm"):
            out["decay_min_factor"] = parse_float(p[2:])
    return out


def _run_once_no_interact(problem_ctx, params: dict, budget: int, device=None, nb_restarts=None):
    device = device or DEFAULTS["device"]
    nb_restarts = DEFAULTS["nb_restarts"] if nb_restarts is None else int(nb_restarts)

    kernel_config = {"name": NO_INTERACT_KERNEL, "epsilon_svgd": params["epsilon_svgd"], "gamma": params["gamma"]}

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
        decay_enabled=True,
        advantage_cfg=params["advantage"],
        kernel_config=kernel_config,
        no_interact=True,
    ).to(device)

    if problem_ctx["type_problem"] == "QUBO":
        list_scores, history = get_Score_trajectoriesQUBO_cuda(
            strategy,
            problem_ctx["dim"],
            DEFAULTS["nb_instances_test"],
            nb_restarts,
            budget,
            params["lambda_"],
            problem_ctx["tensor_Q_test"],
            device,
            False,
            enable_visualization=False,
            return_history=True,
        )
    else:
        total_lambda = strategy.lambda_
        tensor_matrix_locus, tensor_matrix_contrib, _ = getTensorInstances_NK(
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
    scores_array = np.ravel(scores_array)
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
        kernel=NO_INTERACT_KERNEL,
        advantage=params["advantage"],
        M=params["M"],
        lambda_=params["lambda_"],
        epsilon_svgd=params["epsilon_svgd"],
        gamma=params["gamma"],
        decay_start_ratio=params["decay_start_ratio"],
        decay_min_factor=params["decay_min_factor"],
        bandwith_kernel=None,
        no_interact=True,
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


def _read_last_metric_score(metrics_path: Path):
    try:
        lines = [line.strip() for line in metrics_path.read_text().splitlines() if line.strip()]
        if len(lines) < 2:
            return None, None
        header = [h.strip() for h in lines[0].split(",")]
        last = [v.strip() for v in lines[-1].split(",")]

        def pick(col: str):
            if col in header:
                idx = header.index(col)
                if idx < len(last) and last[idx] != "":
                    return last[idx]
            return None

        for col in ("mean", "median", "best_fitness"):
            val = pick(col)
            if val is not None:
                return float(val), col
        return None, None
    except Exception:
        return None, None


def _normalize_score_sign_local(problem_name: str, values):
    if not values:
        return values
    if problem_name.upper() in ("QUBO", "UBQP"):
        return [-val for val in values]
    return values


def _format_score(problem_name: str, value: float) -> str:
    if problem_name.upper() in ("QUBO", "UBQP"):
        return f"{value:.2f}"
    return f"{value:.4f}"


def _write_latex_table(config_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return

    order_problem = {"QUBO": 0, "UBQP": 0, "NK": 1, "NK3": 2}
    rows_sorted = sorted(
        rows, key=lambda r: (order_problem.get(r["problem"], 99), r["dim"], r["type_instance"])
    )

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Detailed comparison between Interacting Agents (SVGD-EDA) and Independent Agents. Bold values indicate the best average score between the two methods.}"
    )
    lines.append(r"\label{tab:interact_vs_no_interact}")
    lines.append(r"\begin{tabular}{lrrccc}")
    lines.append(r"\toprule")
    lines.append(r"Problem & Dim (n) & Type/K & Score (Interact) & Score (No Interact) & Gap (\%) \\")
    lines.append(r"\midrule")

    prev_problem = None
    prev_dim = None
    for row in rows_sorted:
        problem = row["problem"]
        dim = row["dim"]
        t = row["type_instance"]
        score_int = row["score_interact"]
        score_no = row["score_no_interact"]
        gap_pct = row["gap_pct"]

        if prev_problem is not None:
            if problem != prev_problem or dim != prev_dim:
                lines.append(r"\midrule")

        best_int = row["best"] in ("interact", "both")
        best_no = row["best"] in ("no_interact", "both")
        s_int = _format_score(problem, score_int)
        s_no = _format_score(problem, score_no)
        if best_int:
            s_int = rf"\best{{{s_int}}}"
        if best_no:
            s_no = rf"\best{{{s_no}}}"

        gap_str = f"{gap_pct:+.2f}\\%"
        lines.append(f"{problem} & {dim} & {t} & {s_int} & {s_no} & {gap_str} \\\\")

        prev_problem = problem
        prev_dim = dim

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_path = config_dir / "interact_vs_no_interact.tex"
    latex_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no_interact batch with a given config.")
    parser.add_argument("--budget", type=int, default=RUN_BUDGET, help="Budget (default: 10000).")
    parser.add_argument("--outdir", type=str, default=None, help="Root output dir (default: results/config).")
    parser.add_argument("--config", type=str, default=None, help="Config name (otherwise prompted).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if no_interact outputs already exist.",
    )
    args = parser.parse_args()

    config_name = args.config or input(
        "Config name to test (ex: kjsd__advperagentrankweighted__M4__L24__eps0p01__g0p0005__ds0p05__dm0p05): "
    ).strip()
    if not config_name:
        raise SystemExit("Config name is required.")

    params = _parse_config_name(config_name)
    missing = [k for k, v in params.items() if v is None and k != "kernel"]
    if missing:
        raise SystemExit(f"Invalid config_name, missing: {', '.join(missing)}")

    repo_root = Path(__file__).resolve().parent.parent
    out_root = Path(args.outdir) if args.outdir else (repo_root / "results" / "config")
    config_dir = out_root / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    instances_root = repo_root / "source_code" / "instances"
    qubo_instances = _discover_qubo_instances(instances_root / "QUBO", DEFAULTS["nb_instances_test"])
    nk_instances = _discover_nk_instances(instances_root / "nk", DEFAULTS["nb_instances_test"])
    nk3_instances = _discover_nk3_instances(instances_root / "nk3", DEFAULTS["nb_instances_test"])
    instances = qubo_instances + nk_instances + nk3_instances
    if not instances:
        raise SystemExit("Aucune instance QUBO/NK/NK3 compatible avec nb_instances_test.")

    _set_seeds(DEFAULTS["seed"])
    start_all = time.time()

    for inst in instances:
        inst_name = f"{inst['name']}_dim{inst['dim']}_t{inst['type_instance']}"
        inst_dir = config_dir / inst_name / "no_interact"
        metrics_path = inst_dir / "best_metrics.csv"
        if metrics_path.is_file() and not args.overwrite:
            print(f"  -> skip {inst_name} (already done)")
            continue

        print(f"  -> run {inst_name}")
        problem_ctx = _load_instances(inst, DEFAULTS["device"])
        nb_restarts = DEFAULTS["nb_restarts"]

        success = False
        t0 = time.time()
        while nb_restarts > 0 and not success:
            try:
                avg_score, history, meta = _run_once_no_interact(
                    problem_ctx,
                    params,
                    args.budget,
                    device=DEFAULTS["device"],
                    nb_restarts=nb_restarts,
                )
                success = True
            except (RuntimeError, Exception) as exc:
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
            str(repo_root), inst["name"], inst["dim"], inst["type_instance"], avg_score
        )
        _save_history_csv(
            str(inst_dir),
            inst["name"],
            NO_INTERACT_KERNEL,
            {"history": history, "meta": meta},
            ranking=ranking,
            config_name=config_name,
        )

    # Build interact vs no_interact summary
    gap_lines = []
    wins_no = 0
    wins_int = 0
    gap_values = []
    ranks_interact = []
    ranks_no = []
    rows_for_latex = []

    for inst in instances:
        inst_name = f"{inst['name']}_dim{inst['dim']}_t{inst['type_instance']}"
        normal_metrics = config_dir / inst_name / "best_metrics.csv"
        no_metrics = config_dir / inst_name / "no_interact" / "best_metrics.csv"
        if not normal_metrics.exists() or not no_metrics.exists():
            continue

        normal_score, _ = _read_last_metric_score(normal_metrics)
        no_score, _ = _read_last_metric_score(no_metrics)
        if normal_score is None or no_score is None:
            continue

        norm_norm = _normalize_score_sign_local(inst["name"], [normal_score])[0]
        norm_no = _normalize_score_sign_local(inst["name"], [no_score])[0]
        if norm_no == 0:
            continue
        gap_pct = (norm_norm - norm_no) / abs(norm_no) * 100.0
        if gap_pct > 0:
            wins_int += 1
            verdict = "interact better"
        elif gap_pct < 0:
            wins_no += 1
            verdict = "no_interact better"
        else:
            verdict = "tie"
        gap_values.append(gap_pct)
        gap_lines.append(
            f"{inst_name}: interact={norm_norm:.6f}, no_interact={norm_no:.6f}, gap={gap_pct:.2f}% ({verdict})"
        )

        _best_algo, _best_score, rank_int, _n_rank, _pct, _my_cmp, _best_cmp = (
            _rank_vs_global_ranking_excluding_ppo(
                str(repo_root), inst["name"], inst["dim"], inst["type_instance"], normal_score
            )
        )
        if rank_int is not None:
            ranks_interact.append(rank_int)
        _best_algo, _best_score, rank_no, _n_rank, _pct, _my_cmp, _best_cmp = (
            _rank_vs_global_ranking_excluding_ppo(
                str(repo_root), inst["name"], inst["dim"], inst["type_instance"], no_score
            )
        )
        if rank_no is not None:
            ranks_no.append(rank_no)

        # For LaTeX table
        best = "both"
        if norm_norm > norm_no:
            best = "interact"
        elif norm_no > norm_norm:
            best = "no_interact"
        rows_for_latex.append(
            dict(
                problem=inst["name"],
                dim=inst["dim"],
                type_instance=inst["type_instance"],
                score_interact=norm_norm,
                score_no_interact=norm_no,
                gap_pct=gap_pct,
                best=best,
            )
        )

    summary_path = config_dir / "interact_vs_no_interact.txt"
    with summary_path.open("w") as f:
        f.write("Interact vs No_Interact summary\n")
        f.write(f"wins_interact: {wins_int}\n")
        f.write(f"wins_no_interact: {wins_no}\n")
        mean_gap = sum(gap_values) / len(gap_values) if gap_values else 0.0
        f.write(f"mean_gap: {mean_gap:.2f}%\n")
        mean_rank_interact = float(np.mean(ranks_interact)) if ranks_interact else None
        median_rank_interact = float(np.median(ranks_interact)) if ranks_interact else None
        mean_rank_no = float(np.mean(ranks_no)) if ranks_no else None
        median_rank_no = float(np.median(ranks_no)) if ranks_no else None
        f.write(f"mean_rank_interact: {mean_rank_interact}\n")
        f.write(f"median_rank_interact: {median_rank_interact}\n")
        f.write(f"mean_rank_no_interact: {mean_rank_no}\n")
        f.write(f"median_rank_no_interact: {median_rank_no}\n")
        f.write("per_instance_gap:\n")
        for line in gap_lines:
            f.write(f"{line}\n")
    print(f"[DONE] Wrote summary to {summary_path}")
    _write_latex_table(config_dir, rows_for_latex)
    print(f"[DONE] Wrote LaTeX table to {config_dir / 'interact_vs_no_interact.tex'}")
    print(f"[DONE] no_interact batch finished in {time.time() - start_all:.2f}s")


if __name__ == "__main__":
    main()
