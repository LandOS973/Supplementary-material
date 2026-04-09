#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "nevergrad"
DEFAULT_BUDGET = 50000


def iter_leaf_dirs(root: Path, budget: int) -> Iterable[Path]:
    pattern = f"results_nevergrad_*_budget_{budget}_*.txt"
    parents = {path.parent for path in root.rglob(pattern)}
    for parent in sorted(parents):
        if parent.is_dir():
            yield parent


def resolve_xy_keys(fieldnames: List[str]) -> Tuple[str, str]:
    x_field = next(
        (key for key in ("runtime", "evaluations", "evaluation", "eval", "step", "budget") if key in fieldnames),
        fieldnames[0],
    )
    y_field = next(
        (key for key in ("mean", "score", "fitness", "best_fitness", "value") if key in fieldnames),
        fieldnames[1] if len(fieldnames) > 1 else fieldnames[0],
    )
    return x_field, y_field


def load_xy(path: Path) -> Tuple[List[float], List[float]]:
    x_vals: List[float] = []
    y_vals: List[float] = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if not reader.fieldnames:
            return x_vals, y_vals
        x_field, y_field = resolve_xy_keys(reader.fieldnames)
        for row in reader:
            try:
                x_vals.append(float(row.get(x_field, "")))
                y_vals.append(float(row.get(y_field, "")))
            except (TypeError, ValueError):
                continue
    return x_vals, y_vals


def export_leaf(type_dir: Path, budget: int, overwrite: bool) -> tuple[int, int, int]:
    out_path = type_dir / f"mean_curve_budget_{budget}.csv"
    if out_path.exists() and not overwrite:
        try:
            with out_path.open(newline="", encoding="utf-8") as handle:
                row_count = max(0, sum(1 for _ in handle) - 1)
            return 0, row_count, 1
        except OSError:
            return 0, 0, 1

    files = sorted(type_dir.glob(f"*_budget_{budget}_*.txt"))
    if not files:
        return 0, 0, 0

    sums: Dict[float, float] = {}
    counts: Dict[float, int] = {}

    for run_file in files:
        x_vals, y_vals = load_xy(run_file)
        for x_val, y_val in zip(x_vals, y_vals):
            sums[x_val] = sums.get(x_val, 0.0) + y_val
            counts[x_val] = counts.get(x_val, 0) + 1

    if not sums:
        return len(files), 0, 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["runtime", "mean", "count"])
        for runtime in sorted(sums.keys()):
            mean_val = sums[runtime] / counts[runtime]
            writer.writerow([repr(runtime), repr(mean_val), counts[runtime]])

    return len(files), len(sums), 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export mean curve CSV per algo/problem/dim/type from nevergrad run files."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Nevergrad results root (default: <project_root>/results/nevergrad)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="Budget to filter run files (default: 50000)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing mean_curve CSV files.",
    )
    args = parser.parse_args()

    results_root = args.results_root
    budget = int(args.budget)
    overwrite = not bool(args.no_overwrite)

    if not results_root.is_dir():
        raise SystemExit(f"Results root not found: {results_root}")

    leaf_dirs = 0
    exported = 0
    skipped_existing = 0
    files_seen = 0
    rows_written = 0

    for leaf in iter_leaf_dirs(results_root, budget):
        leaf_dirs += 1
        seen, rows, skipped = export_leaf(leaf, budget, overwrite=overwrite)
        if skipped:
            skipped_existing += 1
            continue
        if seen > 0:
            exported += 1
            files_seen += seen
            rows_written += rows

    print(
        f"Done. leaf_dirs={leaf_dirs} exported={exported} skipped_existing={skipped_existing} "
        f"files_seen={files_seen} rows_written={rows_written} budget={budget}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

