#!/usr/bin/env python3
"""Run PPO-EDA in batch: best normal config per instance, rerun in no_interact, rewrite /no_interact summaries."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np

from main_ppo_eda_batch import (
    DEFAULTS,
    _discover_instances,
    _find_best_kernel_summary,
    _load_problem_context,
    _resolve_best_config,
    _run_once,
    _set_seeds,
)


RUN_BUDGET = 10000


def _compute_stats(scores: np.ndarray) -> Dict[str, float]:
    scores_array = np.asarray(scores)
    if scores_array.size == 0:
        nan = float("nan")
        return dict(
            avg_score=nan,
            median_score=nan,
            std_score=nan,
            p2=nan,
            p5=nan,
            p10=nan,
            p25=nan,
            p50=nan,
            p75=nan,
            p90=nan,
            p95=nan,
            p98=nan,
        )
    return dict(
        avg_score=float(np.mean(scores_array)),
        median_score=float(np.percentile(scores_array, 50)),
        std_score=float(np.std(scores_array)),
        p2=float(np.percentile(scores_array, 2)),
        p5=float(np.percentile(scores_array, 5)),
        p10=float(np.percentile(scores_array, 10)),
        p25=float(np.percentile(scores_array, 25)),
        p50=float(np.percentile(scores_array, 50)),
        p75=float(np.percentile(scores_array, 75)),
        p90=float(np.percentile(scores_array, 90)),
        p95=float(np.percentile(scores_array, 95)),
        p98=float(np.percentile(scores_array, 98)),
    )


def _purge_no_interact_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    for path in out_dir.glob("*_best_summary.txt"):
        try:
            path.unlink()
        except OSError:
            pass
    for path in out_dir.glob("*_best_metrics.csv"):
        try:
            path.unlink()
        except OSError:
            pass


def _write_no_interact_summary(out_dir: Path, problem_name: str, kernel_name: str, history: dict, meta: dict) -> None:
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
    metrics_path = out_dir / f"{problem_name}_{kernel_name}_best_metrics.csv"
    with metrics_path.open("w") as f:
        f.write(
            "step,best_fitness,avg_hamming,avg_l1,avg_entropy,"
            "mean,median,std,2%,5%,10%,25%,50%,75%,90%,95%,98%\n"
        )
        for (step, bf, ham, l1, ent, mean, median, std, p2, p5, p10, p25, p50, p75, p90, p95, p98) in rows:
            f.write(
                f"{step},{bf},{ham},{l1},{ent},"
                f"{mean},{median},{std},{p2},{p5},{p10},{p25},{p50},{p75},{p90},{p95},{p98}\n"
            )

    summary_path = out_dir / f"{problem_name}_{kernel_name}_best_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"Problem: {problem_name}\n")
        f.write(f"Kernel: {kernel_name}\n")
        f.write(f"Advantage: {meta['advantage']}\n")
        f.write(f"M: {meta['M']}\n")
        f.write(f"lambda: {meta['lambda_']}\n")
        f.write(f"epsilon_svgd: {meta['epsilon_svgd']}\n")
        f.write(f"gamma: {meta['gamma']}\n")
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
        f.write("ranking: unavailable\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    instances = _discover_instances(repo_root)
    if not instances:
        print(f"[WARN] Aucun dossier d'instances dans {repo_root / 'results' / 'experiments'}")
        return

    _set_seeds(DEFAULTS["seed"])
    start_all = time.time()

    for inst in instances:
        problem_name = inst["name"]
        dim = inst["dim"]
        type_instance = inst["type_instance"]
        instance_name = f"{problem_name}_dim{dim}_t{type_instance}"
        print(f"[INFO] Instance {instance_name}")

        summary_dir = repo_root / "results" / "experiments" / instance_name
        no_interact_dir = summary_dir / "no_interact"
        if no_interact_dir.exists():
            print(f"  [SKIP] {instance_name} deja present dans no_interact")
            continue

        problem_ctx = _load_problem_context(inst)
        best_kernel, best_cfg = _find_best_kernel_summary(summary_dir, problem_ctx["type_problem"])
        if best_kernel is None or best_cfg is None:
            print(f"[WARN] Pas de resume normal pour {instance_name} dans {summary_dir}")
            continue

        params = _resolve_best_config(best_cfg, best_kernel, str(repo_root), mode="no_interact")
        print(
            f"  -> kernel={params['kernel_name']} M={params['M']} lambda={params['lambda_']} "
            f"eps={params['epsilon_svgd']} gamma={params['gamma']} budget={RUN_BUDGET}"
        )

        _set_seeds(DEFAULTS["seed"])
        t0 = time.time()
        scores, history = _run_once(problem_ctx, params, RUN_BUDGET)
        stats = _compute_stats(scores)
        dt = time.time() - t0
        print(f"    avg_score={stats['avg_score']:.6f} runtime={dt:.2f}s")

        _purge_no_interact_dir(no_interact_dir)
        meta = dict(
            advantage=params["advantage"],
            M=params["M"],
            lambda_=params["lambda_"],
            epsilon_svgd=params["epsilon_svgd"],
            gamma=params["gamma"],
            bandwith_kernel=params.get("bandwith_kernel"),
            no_interact=True,
            **stats,
        )
        _write_no_interact_summary(no_interact_dir, problem_ctx["type_problem"], params["kernel_name"], history, meta)

    print(f"[DONE] no_interact batch finished in {time.time() - start_all:.2f}s")


if __name__ == "__main__":
    main()
