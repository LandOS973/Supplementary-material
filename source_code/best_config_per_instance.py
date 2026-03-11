#!/usr/bin/env python3
"""Find best config per instance based on last score in best_metrics.csv.

Scans results/config/*/{PROBLEM}_dim{N}_t{K}/best_metrics.csv and picks the
config with the highest score for each instance.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
import re
from typing import Dict, Tuple

from main_expe_overall import _rank_vs_global_ranking_excluding_ppo

INSTANCE_RE = re.compile(r"^(?P<problem>NK|QUBO)_dim(?P<dim>\d+)_t(?P<t>\d+)$")


def read_last_score(metrics_path: Path) -> Tuple[float | None, str | None]:
    """Return (score, metric_name) from last line. Prefer mean, then median, then best_fitness."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Best config per instance.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/config"),
        help="Root directory with config results (default: results/config)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/aggregation/best_config_per_instance.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    results_root = args.results_root
    if not results_root.is_dir():
        print(f"[ERR] Missing results root: {results_root}", file=sys.stderr)
        return 1

    best: Dict[Tuple[str, int, int], Tuple[str, float, str]] = {}
    total = 0
    used = 0

    for cfg_dir in sorted(results_root.iterdir()):
        if not cfg_dir.is_dir():
            continue
        config_name = cfg_dir.name
        for instance_dir in cfg_dir.iterdir():
            if not instance_dir.is_dir():
                continue
            m = INSTANCE_RE.match(instance_dir.name)
            if not m:
                continue
            problem = m.group("problem")
            dim = int(m.group("dim"))
            t = int(m.group("t"))
            metrics_path = instance_dir / "best_metrics.csv"
            if not metrics_path.is_file():
                # legacy name
                metrics_path = instance_dir / f"{problem}_best_metrics.csv"
            if not metrics_path.is_file():
                continue
            total += 1
            score, metric = read_last_score(metrics_path)
            if score is None:
                continue
            used += 1
            key = (problem, dim, t)
            current = best.get(key)
            if current is None:
                best[key] = (config_name, score, metric or "")
            else:
                _, best_score, _ = current
                if problem == "QUBO":
                    if score < best_score:
                        best[key] = (config_name, score, metric or "")
                else:
                    if score > best_score:
                        best[key] = (config_name, score, metric or "")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "problem",
                "dim",
                "type_instance",
                "best_config",
                "score",
                "metric",
                "global_rank",
                "global_total",
                "global_percent",
                "global_best_algo",
                "global_best_score",
            ]
        )
        repo_root = Path(__file__).resolve().parent.parent
        for (problem, dim, t), (config, score, metric) in sorted(best.items()):
            best_algo, best_score, my_rank, n_rank, my_pct, _my_cmp, _best_cmp = _rank_vs_global_ranking_excluding_ppo(
                str(repo_root), problem, dim, t, score
            )
            writer.writerow(
                [
                    problem,
                    dim,
                    t,
                    config,
                    score,
                    metric,
                    my_rank,
                    n_rank,
                    my_pct,
                    best_algo,
                    best_score,
                ]
            )

    print(f"[DONE] {args.out} (instances={len(best)}, scanned={total}, used={used})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
