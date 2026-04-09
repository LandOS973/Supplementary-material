#!/usr/bin/env python3
"""Export final score rows for nevergrad run files.

For each leaf directory:
  results/nevergrad/<algo>/<problem>/<dim>/<type_instance>/

this script collects files matching:
  results_nevergrad_*_budget_<BUDGET>_*.txt

and writes:
  final_scores_budget_<BUDGET>.csv

with one row per run file (last runtime, last score).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "nevergrad"
DEFAULT_BUDGET = 50000

RUN_RE = re.compile(
    r"^results_nevergrad_"
    r"(?P<algo>.+?)_"
    r"(?P<problem>[A-Za-z0-9]+)_"
    r"(?P<dim>\d+)_"
    r"(?P<type_instance>\d+)_"
    r"budget_(?P<budget>\d+)_"
    r".*?(?:_i_(?P<instance>\d+))?"
    r"(?:_r_(?P<restart>\d+))?"
    r"\.txt$"
)


def _parse_float(text: str) -> float | None:
    try:
        return float(text.strip())
    except (TypeError, ValueError):
        return None


def read_last_runtime_score(path: Path) -> tuple[float, float] | None:
    """Return (runtime, score) from the last numeric row in the file."""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None

    for line in reversed(lines):
        raw = line.strip()
        if not raw:
            continue
        if raw.lower().startswith("runtime"):
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 2:
            continue
        runtime = _parse_float(parts[0])
        score = _parse_float(parts[1])
        if runtime is None or score is None:
            continue
        return runtime, score
    return None


def iter_leaf_dirs(root: Path, budget: int) -> Iterable[Path]:
    pattern = f"results_nevergrad_*_budget_{budget}_*.txt"
    parents = {path.parent for path in root.rglob(pattern)}
    for parent in sorted(parents):
        if parent.is_dir():
            yield parent


def export_leaf(type_dir: Path, budget: int) -> tuple[int, int]:
    pattern = f"results_nevergrad_*_budget_{budget}_*.txt"
    run_files = sorted(type_dir.glob(pattern))
    if not run_files:
        return 0, 0

    rows = []
    for run_file in run_files:
        parsed = read_last_runtime_score(run_file)
        if parsed is None:
            continue
        runtime, score = parsed
        instance = ""
        restart = ""
        match = RUN_RE.match(run_file.name)
        if match:
            instance = match.group("instance") or ""
            restart = match.group("restart") or ""
        rows.append(
            {
                "instance": instance,
                "restart": restart,
                "runtime": runtime,
                "score": score,
                "filename": run_file.name,
            }
        )

    if not rows:
        return len(run_files), 0

    rows.sort(
        key=lambda row: (
            int(row["instance"]) if str(row["instance"]).isdigit() else 10**9,
            int(row["restart"]) if str(row["restart"]).isdigit() else 10**9,
            row["filename"],
        )
    )

    out_path = type_dir / f"final_scores_budget_{budget}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance", "restart", "runtime", "score", "filename"])
        for row in rows:
            writer.writerow(
                [
                    row["instance"],
                    row["restart"],
                    row["runtime"],
                    row["score"],
                    row["filename"],
                ]
            )
    return len(run_files), len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export one CSV per algo/problem/dim/type with final scores per run file."
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
        help="Budget value to filter run files (default: 50000)",
    )
    args = parser.parse_args()

    results_root = args.results_root
    budget = int(args.budget)
    if not results_root.is_dir():
        raise SystemExit(f"Results root not found: {results_root}")

    leaf_count = 0
    exported_count = 0
    files_seen = 0
    rows_written = 0
    for leaf in iter_leaf_dirs(results_root, budget):
        leaf_count += 1
        seen, written = export_leaf(leaf, budget)
        if seen > 0:
            exported_count += 1
            files_seen += seen
            rows_written += written

    print(
        f"Done. leaf_dirs={leaf_count} exported={exported_count} "
        f"files_seen={files_seen} rows_written={rows_written} budget={budget}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
