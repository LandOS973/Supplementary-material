#!/usr/bin/env python3
"""Assemble final_scores_budget_<B>.csv from per-seed final_scores_seed<S>_budget_<B>.csv.

main_nevergrad.py / main_baseline_edas_and_tabu.py write one small CSV per (algo, t, seed)
task with the final score of each instance (already "higher = better"). This concatenates
them per result directory into the final_scores_budget_<B>.csv that main_table.py
(Wilcoxon) and main_global_ranking.py read. Only directories containing per-seed files
are touched, so legacy (n<=256) final_scores CSVs are never overwritten.

Usage:
    python3 additional_results/build_final_scores_from_seeds.py [--budget 50000]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "nevergrad"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=50000)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()

    pattern = f"final_scores_seed*_budget_{args.budget}.csv"
    dirs = sorted({p.parent for p in args.results_root.rglob(pattern)})
    for d in dirs:
        rows = []
        for seed_file in sorted(d.glob(pattern)):
            with seed_file.open(newline="") as handle:
                rows.extend(csv.DictReader(handle))
        rows.sort(key=lambda r: (int(r["instance"]), int(r["restart"])))
        out = d / f"final_scores_budget_{args.budget}.csv"
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["instance", "restart", "runtime", "score", "filename"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"{out.relative_to(args.results_root)}: {len(rows)} runs")
    print(f"{len(dirs)} dossiers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
