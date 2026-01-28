#!/usr/bin/env python3
"""Generate a summary table image across all tested instances."""

from __future__ import annotations

import argparse
import csv
import re
import textwrap
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent


def parse_summary_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    return data


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def is_maximization_problem(problem_name: str) -> bool:
    return problem_name.upper() in ("NK", "NK3", "BLOCK")


def normalize_score(problem_name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if is_maximization_problem(problem_name):
        return value
    return -value if value < 0 else value


def parse_rank_value(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    match = re.search(r"(\d+)\s*/\s*(\d+)", value)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def load_best_summary(experiment_dir: Path, problem_name: str) -> dict[str, str] | None:
    summary_files = sorted(experiment_dir.glob(f"{problem_name}_*_best_summary.txt"))
    if not summary_files:
        return None
    best_summary = None
    best_score = None
    maximize = is_maximization_problem(problem_name)
    for path in summary_files:
        data = parse_summary_file(path)
        avg_score = parse_float(data.get("avg_score"))
        if avg_score is None:
            continue
        if best_score is None:
            best_score = avg_score
            best_summary = data
            continue
        if maximize and avg_score > best_score:
            best_score = avg_score
            best_summary = data
        elif not maximize and avg_score < best_score:
            best_score = avg_score
            best_summary = data
    return best_summary


def ranking_path(problem_name: str, dim: int, type_instance: int) -> Path:
    if problem_name.upper() == "QUBO":
        name = "UBQP"
    else:
        name = problem_name
    return ROOT / "additional_results" / "global_ranking" / f"{name}_N_{dim}_K_{type_instance}_ranks.csv"


def load_ranking(path: Path) -> List[tuple[str, float]]:
    if not path.exists():
        return []
    rows: List[tuple[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("name_algo") or row.get("name") or row.get("algo") or "").strip()
            score = parse_float(row.get("score"))
            if not name or score is None:
                continue
            rows.append((name, score))
    return rows


def discover_instances() -> List[tuple[str, int, int, Path]]:
    exp_root = ROOT / "results" / "experiments"
    if not exp_root.exists():
        return []
    instances: List[tuple[str, int, int, Path]] = []
    seen: set[tuple[str, int, int]] = set()
    for entry in exp_root.iterdir():
        if not entry.is_dir():
            continue
        match = re.match(r"^(?P<name>.+)_dim(?P<dim>\d+)_t(?P<t>\d+)$", entry.name)
        if not match:
            continue
        name = match.group("name")
        dim = int(match.group("dim"))
        type_instance = int(match.group("t"))
        key = (name, dim, type_instance)
        if key in seen:
            continue
        seen.add(key)
        instances.append((name, dim, type_instance, entry))
    return sorted(instances, key=lambda item: (item[0], item[1], item[2]))


def format_rank(rank: int | None, total: int | None) -> str:
    if rank is None or total is None:
        return "—"
    return f"{rank}/{total}"


def format_score(problem_name: str, value: float | None) -> str:
    if value is None:
        return "—"
    if problem_name.upper() in ("NK", "NK3"):
        return f"{value:.4f}"
    return f"{value:.1f}"


def wrap_name(name: str | None, width: int = 18) -> str:
    if not name:
        return "—"
    return textwrap.fill(name, width=width)


def build_rows(methods: List[str]) -> tuple[List[List[str]], List[str]]:
    rows: List[List[str]] = []
    for problem_name, dim, type_instance, exp_dir in discover_instances():
        summary = load_best_summary(exp_dir, problem_name)
        if summary is None:
            continue
        avg_score = parse_float(summary.get("avg_score"))
        rank_value = parse_rank_value(summary.get("ranking_my_rank"))
        my_rank, my_total = (rank_value or (None, None))

        ranking_file = ranking_path(problem_name, dim, type_instance)
        ranking = load_ranking(ranking_file)
        total = len(ranking) if ranking else None
        rank_map: dict[str, tuple[int, float]] = {}
        for idx, (name, score) in enumerate(ranking, start=1):
            rank_map[name] = (idx, score)

        row: List[str] = [problem_name, str(dim), str(type_instance)]

        our_score = normalize_score(problem_name, avg_score)
        row.extend([format_rank(my_rank, my_total), format_score(problem_name, our_score)])

        for method in methods:
            rank_entry = rank_map.get(method)
            if rank_entry:
                rank, score = rank_entry
                row.extend([format_rank(rank, total), format_score(problem_name, score)])
            else:
                row.extend(["—", "—"])

        best_name = "—"
        best_rank = None
        best_score = None
        for idx, (name, score) in enumerate(ranking, start=1):
            if name == "PPO-EDA":
                continue
            best_name = name
            best_rank = idx
            best_score = score
            break

        row.extend(
            [
                wrap_name(best_name),
                format_rank(best_rank, total),
                format_score(problem_name, best_score),
            ]
        )
        rows.append(row)

    return rows, ["Pb", "n", "t"] + ["Rank", "Score"] * (1 + len(methods)) + ["Name", "Rank", "Score"]


def plot_table(rows: List[List[str]], col_labels: List[str], output_png: Path, output_pdf: Path) -> None:
    if not rows:
        print("[WARN] No rows to render.")
        return

    n_rows = len(rows)
    fig_height = 1.6 + 0.35 * n_rows
    fig, ax = plt.subplots(figsize=(16, fig_height), dpi=220)
    ax.axis("off")

    col_widths = [0.06, 0.04, 0.04] + [0.06, 0.06] * 4 + [0.18, 0.06, 0.06]
    total_width = sum(col_widths)
    col_widths = [w / total_width for w in col_widths]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 0.9],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    for col_idx, width in enumerate(col_widths):
        for row_idx in range(n_rows + 1):
            cell = table.get_celld().get((row_idx, col_idx))
            if cell:
                cell.set_width(width)

    for row_idx in range(n_rows + 1):
        cell = table.get_celld().get((row_idx, 0))
        if cell:
            cell.set_height(0.9 / (n_rows + 1))

    x_edges = [0.0]
    for width in col_widths:
        x_edges.append(x_edges[-1] + width)

    def center_between(start: int, end: int) -> float:
        return (x_edges[start] + x_edges[end]) / 2

    ax.text(center_between(0, 3), 0.965, "Instances", ha="center", va="center", fontsize=10)
    ax.text(center_between(3, 11), 0.965, "Methods", ha="center", va="center", fontsize=10)
    ax.text(center_between(11, 14), 0.965, "Best method (others)", ha="center", va="center", fontsize=10)

    method_labels = ["reinforce SVGD", "PBIL", "MIMIC", "BOA"]
    method_spans = [(3, 5), (5, 7), (7, 9), (9, 11)]
    for label, (start, end) in zip(method_labels, method_spans):
        ax.text(center_between(start, end), 0.925, label, ha="center", va="center", fontsize=9)

    ax.hlines(0.9, 0, 1, transform=ax.transAxes, color="#333333", linewidth=0.8)
    ax.hlines(0.0, 0, 1, transform=ax.transAxes, color="#333333", linewidth=0.8)

    rows_problem = [row[0] for row in rows]
    for idx in range(1, n_rows):
        if rows_problem[idx] != rows_problem[idx - 1]:
            cell = table.get_celld().get((idx + 1, 0))
            if cell:
                ax.hlines(cell.get_y(), 0, 1, transform=ax.transAxes, color="#666666", linewidth=0.6)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved table to {output_png} and {output_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summary table for all instances.")
    parser.add_argument("--png", type=Path, default=ROOT / "courbes" / "summary_table.png")
    parser.add_argument("--pdf", type=Path, default=ROOT / "courbes" / "summary_table.pdf")
    args = parser.parse_args()

    methods = ["PBIL", "MIMIC", "BOA"]
    rows, col_labels = build_rows(methods)
    plot_table(rows, col_labels, args.png, args.pdf)


if __name__ == "__main__":
    main()
