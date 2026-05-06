"""
Compare PPO-EDA async (l_active/r_influence < M) vs normal (l_active = r_influence = M).
For each config in DEFAULT_GRIDS that defines l_active/r_influence, runs both variants
on all instances and writes a comparison report to <async_config_dir>/async_vs_normal.txt.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

SOURCE_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SOURCE_CODE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_CODE_DIR)

import numpy as np
import torch

from expe.main_expe_overall import (
    DEFAULTS,
    _apply_m_override,
    _build_config_name,
    _discover_nk3_instances,
    _discover_nk_instances,
    _discover_qubo_instances,
    _expand_grid,
    _get_grid_m_values,
    _instance_already_done,
    _is_cuda_oom,
    _load_instances,
    _parse_int_list,
    _rank_vs_global_ranking_excluding_ppo,
    _run_once,
    _save_history_csv,
    _save_raw_scores_csv,
    _set_seeds,
)

DEFAULT_GRIDS = [
    dict(
        kernels=["rbf"],
        advantages=["globalrankweighted"],
        M_values=[20, 30, 40],
        lambda_values=[10],
        epsilon_svgd=[0.08],
        gamma=[0.01],
        decay_start_ratio=[0.03],
        decay_min_factor=[0.01],
        l_active=[5],
        r_influence=[20],
    )
]


def _load_avg_score(inst_dir: Path, kernel: str, problem: str) -> float | None:
    for candidate in (
        inst_dir / "best_metrics.csv",
        inst_dir / f"{problem}_{kernel}_best_metrics.csv",
    ):
        if not candidate.is_file():
            continue
        try:
            with open(candidate, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            last = rows[-1]
            for col in ("mean", "median", "best_fitness"):
                if last.get(col) not in (None, ""):
                    return float(last[col])
        except Exception:
            continue
    return None


def _run_variant(
    inst: dict,
    params: dict,
    l_active: int | None,
    r_influence: int | None,
    config_name: str,
    out_root: str,
    repo_root: str,
) -> tuple[float, int | None, int] | None:
    """Run one variant on one instance. Returns (avg_score, rank, n_rank) or None."""
    inst_name = f"{inst['name']}_dim{inst['dim']}_t{inst['type_instance']}"
    inst_dir = Path(out_root) / config_name / inst_name

    if _instance_already_done(str(inst_dir), inst["name"], params["kernel"]):
        avg_score = _load_avg_score(inst_dir, params["kernel"], inst["name"])
        if avg_score is None:
            return None
        ranking = _rank_vs_global_ranking_excluding_ppo(
            repo_root, inst["name"], inst["dim"], inst["type_instance"], avg_score
        )
        return avg_score, ranking[2], ranking[3]

    problem_ctx = _load_instances(inst, DEFAULTS["device"])
    nb_restarts = DEFAULTS["nb_restarts"]
    success = False
    avg_score = history = meta = scores_array = None
    while nb_restarts > 0 and not success:
        try:
            avg_score, history, meta, scores_array = _run_once(
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
                l_active=l_active,
                r_influence=r_influence,
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
                print(f"       [OOM] retry nb_restarts={nb_restarts}")
            else:
                print("       [OOM] skip instance")

    if not success:
        return None

    ranking = _rank_vs_global_ranking_excluding_ppo(
        repo_root, inst["name"], inst["dim"], inst["type_instance"], avg_score
    )
    _save_history_csv(
        str(inst_dir),
        inst["name"],
        params["kernel"],
        {"history": history, "meta": meta},
        ranking=ranking,
        config_name=config_name,
    )
    _save_raw_scores_csv(str(inst_dir), scores_array)
    return avg_score, ranking[2], ranking[3]


def _write_comparison(
    out_path: Path,
    async_name: str,
    normal_name: str,
    results: list[dict],
) -> None:
    col_inst = 30
    col_score = 13
    col_gap = 11
    col_rank = 11

    header_row = (
        f"{'Instance':<{col_inst}}"
        f"{'Async Score':>{col_score}}"
        f"{'Normal Score':>{col_score}}"
        f"{'Score Gap':>{col_gap}}"
        f"{'Async Rank':>{col_rank}}"
        f"{'Norm Rank':>{col_rank}}"
    )
    sep = "-" * len(header_row)

    lines = [
        "=" * len(header_row),
        "ASYNC vs NORMAL — Comparison Report",
        "=" * len(header_row),
        f"Async  config : {async_name}",
        f"Normal config : {normal_name}",
        "",
        header_row,
        sep,
    ]

    score_gaps = []
    async_score_wins = normal_score_wins = 0

    for r in results:
        label = f"{r['problem']}_dim{r['dim']}_t{r['t']}"
        a_s = r.get("async_score")
        n_s = r.get("normal_score")
        a_r = r.get("async_rank")
        n_r = r.get("normal_rank")

        a_s_str = f"{abs(a_s):.6f}" if a_s is not None else "N/A"
        n_s_str = f"{abs(n_s):.6f}" if n_s is not None else "N/A"
        a_r_str = str(a_r) if a_r is not None else "N/A"
        n_r_str = str(n_r) if n_r is not None else "N/A"
        s_gap_str = "N/A"

        if a_s is not None and n_s is not None:
            sg = abs(a_s) - abs(n_s)
            score_gaps.append(sg)
            s_gap_str = f"{sg:+.6f}"
            if sg > 0:
                async_score_wins += 1
            elif sg < 0:
                normal_score_wins += 1

        lines.append(
            f"{label:<{col_inst}}"
            f"{a_s_str:>{col_score}}"
            f"{n_s_str:>{col_score}}"
            f"{s_gap_str:>{col_gap}}"
            f"{a_r_str:>{col_rank}}"
            f"{n_r_str:>{col_rank}}"
        )

    n = len(results)
    both = len(score_gaps)
    lines += [
        "",
        "SUMMARY",
        "-" * 40,
    ]
    if score_gaps:
        lines += [
            f"Mean score gap |async| - |normal| : {np.mean(score_gaps):+.6f}",
            f"Async wins (score)              : {async_score_wins}/{both}",
            f"Normal wins (score)             : {normal_score_wins}/{both}",
            f"Ties (score)                    : {both - async_score_wins - normal_score_wins}/{both}",
        ]
    lines.append(f"Instances compared              : {both}/{n}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [REPORT] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Async vs Normal PPO-EDA comparison.")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("-m", "--m-values", type=_parse_int_list, default=None)
    args = parser.parse_args()

    device = DEFAULTS["device"]
    if torch.cuda.is_available():
        print(f"[DEVICE] GPU — {torch.cuda.get_device_name(device)}")
    else:
        print("[DEVICE] CPU")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_root = args.outdir or os.path.join(repo_root, "results", "config")
    Path(out_root).mkdir(parents=True, exist_ok=True)

    grids = _apply_m_override(DEFAULT_GRIDS, args.m_values)
    for idx, grid in enumerate(grids, start=1):
        print(f"[GRID {idx}] M_values={_get_grid_m_values(grid)}")

    instances_root = Path(repo_root) / "source_code" / "instances"
    instances = (
        _discover_qubo_instances(instances_root / "QUBO", DEFAULTS["nb_instances_test"])
        + _discover_nk_instances(instances_root / "nk", DEFAULTS["nb_instances_test"])
        + _discover_nk3_instances(instances_root / "nk3", DEFAULTS["nb_instances_test"])
    )
    if not instances:
        raise SystemExit("No compatible instances found.")

    _set_seeds(DEFAULTS["seed"])
    start_all = time.time()

    for grid in grids:
        for config_name, params in _expand_grid(grid):
            l_active = params.get("l_active")
            r_influence = params.get("r_influence")

            if l_active is None or l_active >= params["M"]:
                print(f"[SKIP] {config_name} — l_active={l_active} not < M={params['M']}")
                continue

            async_name = config_name
            # Normal baseline: M = l_active, no partial updates — same per-step compute budget
            normal_params = {**params, "M": l_active, "l_active": None, "r_influence": None}
            normal_name = _build_config_name(None, normal_params)

            print(f"\n[PAIR]")
            print(f"  async  : {async_name}  (M={params['M']}, l_active={l_active})")
            print(f"  normal : {normal_name}  (M={l_active})")

            results = []
            for inst in instances:
                inst_label = f"{inst['name']}_dim{inst['dim']}_t{inst['type_instance']}"
                print(f"  [{inst_label}]")

                print(f"    -> async  (M={params['M']}, l_active={l_active}, r_influence={r_influence})")
                t0 = time.time()
                res_async = _run_variant(inst, params, l_active, r_influence, async_name, out_root, repo_root)
                score_str = f"{res_async[0]:.6f}" if res_async else "FAILED"
                print(f"       {score_str}  rank={res_async[1] if res_async else '?'}  ({time.time()-t0:.1f}s)")

                print(f"    -> normal (M={l_active}, full updates)")
                t0 = time.time()
                res_normal = _run_variant(inst, normal_params, None, None, normal_name, out_root, repo_root)
                score_str = f"{res_normal[0]:.6f}" if res_normal else "FAILED"
                print(f"       {score_str}  rank={res_normal[1] if res_normal else '?'}  ({time.time()-t0:.1f}s)")

                entry = dict(problem=inst["name"], dim=inst["dim"], t=inst["type_instance"])
                if res_async:
                    entry["async_score"], entry["async_rank"], _ = res_async
                if res_normal:
                    entry["normal_score"], entry["normal_rank"], _ = res_normal
                results.append(entry)

                report_path = Path(out_root) / async_name / "async_vs_normal.txt"
                _write_comparison(report_path, async_name, normal_name, results)

    print(f"\n[DONE] Elapsed: {time.time() - start_all:.2f}s")


if __name__ == "__main__":
    main()
