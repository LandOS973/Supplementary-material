#!/usr/bin/env python3
"""Build global ranking files from nevergrad results.

Reads results from <project_root>/results/nevergrad by default,
computes the mean final score over 100 runs per instance, and writes
CSV rankings to additional_results/global_ranking.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "nevergrad"
DEFAULT_BUDGET = 50000
DEFAULT_EXPECTED_RUNS = 100


def iter_algo_dirs(results_root: Path) -> List[Path]:
    return sorted([p for p in results_root.iterdir() if p.is_dir()])


def find_instances(results_root: Path) -> Dict[Tuple[str, int, int], Dict[str, Path]]:
    """Return mapping instance -> algo -> directory for that instance.

    Instance key is (problem, dim, type_instance).
    """
    instances: Dict[Tuple[str, int, int], Dict[str, Path]] = {}
    for algo_dir in iter_algo_dirs(results_root):
        algo = algo_dir.name
        for problem_dir in algo_dir.iterdir():
            if not problem_dir.is_dir():
                continue
            problem = problem_dir.name.upper()
            if problem not in ("NK", "NK3", "QUBO"):
                continue
            for dim_dir in problem_dir.iterdir():
                if not dim_dir.is_dir():
                    continue
                try:
                    dim = int(dim_dir.name)
                except ValueError:
                    continue
                for type_dir in dim_dir.iterdir():
                    if not type_dir.is_dir():
                        continue
                    try:
                        type_instance = int(type_dir.name)
                    except ValueError:
                        continue
                    key = (problem, dim, type_instance)
                    instances.setdefault(key, {})[algo] = type_dir
    return instances


def read_last_score(path: Path) -> float | None:
    """Read last numeric score from a run file. Return None if not found."""
    # Scan from end for the last non-empty, non-header line
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line or line.lower().startswith("runtime"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            return float(parts[1])
        except ValueError:
            continue
    return None


def read_scores_from_final_csv(path: Path) -> List[float]:
    """Read final scores from pre-aggregated CSV (one row per run)."""
    scores: List[float] = []
    if not path.exists():
        return scores
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return scores
            score_key = None
            for candidate in ("score", "final_score", "value"):
                if candidate in reader.fieldnames:
                    score_key = candidate
                    break
            if score_key is None:
                return scores
            has_filename = "filename" in reader.fieldnames
            rows: List[tuple[str, float]] = []
            for row in reader:
                raw = row.get(score_key)
                if raw is None:
                    continue
                try:
                    score = float(raw)
                except ValueError:
                    continue
                if has_filename:
                    rows.append((str(row.get("filename") or ""), score))
                else:
                    scores.append(score)
            if has_filename:
                rows.sort(key=lambda item: item[0])
                scores = [score for _, score in rows]
    except OSError:
        return []
    return scores


def collect_scores(
    type_dir: Path,
    budget: int,
) -> tuple[List[float], int, int, str]:
    csv_path = type_dir / f"final_scores_budget_{budget}.csv"
    csv_scores = read_scores_from_final_csv(csv_path)
    if csv_scores:
        return csv_scores, len(csv_scores), 0, "csv"

    pattern = f"*_budget_{budget}_*.txt"
    files = sorted(type_dir.glob(pattern))
    bad = 0
    scores: List[float] = []
    for run_file in files:
        score = read_last_score(run_file)
        if score is None:
            bad += 1
            continue
        scores.append(score)
    return scores, len(files), bad, "txt"


def mean(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for v in values:
        total += v
        count += 1
    if count == 0:
        raise ValueError("Cannot compute mean of empty list")
    return total / count


def filename_for(problem: str, dim: int, type_instance: int) -> str:
    if problem.upper() == "QUBO":
        return f"UBQP_N_{dim}_K_{type_instance}_ranks.csv"
    if problem.upper() == "NK3":
        return f"NK3_N_{dim}_K_{type_instance}_ranks.csv"
    return f"NK_N_{dim}_K_{type_instance}_ranks.csv"


def build_rankings(
    results_root: Path,
    out_dir: Path,
    budget: int,
    expected_runs: int,
) -> int:
    algo_dirs = iter_algo_dirs(results_root)
    algos = [p.name for p in algo_dirs]
    instances = find_instances(results_root)

    if not instances:
        print(f"[WARN] No instances found under {results_root}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0
    for (problem, dim, type_instance), per_algo_dir in sorted(instances.items()):
        low_runs = []
        per_algo_score = {}
        for algo in algos:
            type_dir = per_algo_dir.get(algo)
            if type_dir is None:
                continue
            scores, total_runs, bad, source = collect_scores(type_dir, budget)
            if total_runs == 0:
                continue
            if total_runs < expected_runs:
                low_runs.append((f"{algo}:{source}", total_runs))
            if bad:
                low_runs.append((f"{algo}:bad", bad))
            if not scores:
                continue
            per_algo_score[algo] = mean(scores)

        if not per_algo_score:
            skipped += 1
            print(
                f"[WARN] Skip {problem} N={dim} K={type_instance}: no runs found",
                file=sys.stderr,
            )
            continue
        if low_runs:
            low_str = ", ".join(f"{a}={c}" for a, c in low_runs)
            print(
                f"[WARN] {problem} N={dim} K={type_instance}: low runs (<{expected_runs}) ({low_str})",
                file=sys.stderr,
            )

        # Sort by score descending (maximization)
        ranking = sorted(per_algo_score.items(), key=lambda kv: kv[1], reverse=True)
        out_path = out_dir / filename_for(problem, dim, type_instance)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name_algo", "score"])
            for algo, score in ranking:
                writer.writerow([algo, score])
        written += 1

    print(
        f"Done. Wrote {written} ranking files, skipped {skipped} instances.",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build global ranking CSVs.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Path to the nevergrad results root (default: <project_root>/results/nevergrad)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "global_ranking",
        help="Output directory for ranking CSVs",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="Budget to filter on",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=DEFAULT_EXPECTED_RUNS,
        help="Expected number of run files per algo/instance",
    )

    args = parser.parse_args()
    return build_rankings(
        results_root=args.results_root,
        out_dir=args.out_dir,
        budget=args.budget,
        expected_runs=args.expected_runs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
