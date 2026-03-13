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


def load_last_metrics_score(csv_path: Path) -> float | None:
    """Load the score from the last data row of a best_metrics.csv file."""
    if not csv_path.exists():
        return None

    def _data_lines(handle):
        for line in handle:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            yield line

    try:
        with csv_path.open() as handle:
            reader = csv.reader(_data_lines(handle))
            header = next(reader, None)
            if not header:
                return None
            last_row = None
            for row in reader:
                last_row = row
            if not last_row:
                return None
            row_dict = dict(zip(header, last_row))
            for key in ("mean", "avg_score", "score", "avg", "avg_fitness", "fitness", "best_fitness"):
                if key in row_dict and row_dict[key]:
                    return parse_float(row_dict[key])
    except Exception:
        return None
    return None


def load_svgd_score(config_dir: Path, problem_name: str, dim: int, type_instance: int) -> tuple[float | None, tuple[int, int] | None]:
    """Load SVGD score and optional rank from config directory.

    Returns (score, (rank, total)) where rank tuple may be None.
    """
    instance_dir = config_dir / f"{problem_name}_dim{dim}_t{type_instance}"
    if not instance_dir.exists():
        return None, None

    csv_path = instance_dir / "best_metrics.csv"
    csv_score = load_last_metrics_score(csv_path)
    if csv_score is None:
        return None, None
    return csv_score, None


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
    value = abs(value)
    if problem_name.upper() in ("NK", "NK3"):
        return f"{value:.4f}"
    return f"{value:.1f}"


def wrap_name(name: str | None, width: int = 18) -> str:
    if not name:
        return "—"
    return textwrap.fill(name, width=width)


def build_rows(methods: List[str], config_dir: Path) -> tuple[List[List[str]], List[str]]:
    rows: List[List[str]] = []

    # Gather instances from results/experiments
    instances = list(discover_instances())

    # Also gather instances present in the config directory (to ensure none missing)
    if config_dir.exists():
        for entry in config_dir.iterdir():
            # looking for directories like NK_dim64_t1
            match = re.match(r"^(?P<name>.+)_dim(?P<dim>\d+)_t(?P<t>\d+)$", entry.name)
            if not match:
                continue
            name = match.group("name")
            dim = int(match.group("dim"))
            type_instance = int(match.group("t"))
            key = (name, dim, type_instance)
            if key not in {(i[0], i[1], i[2]) for i in instances}:
                instances.append((name, dim, type_instance, entry))

    # sort instances for deterministic output
    instances = sorted(instances, key=lambda item: (item[0], item[1], item[2]))

    for problem_name, dim, type_instance, exp_dir in instances:
        # Exclude specific problematic instance per user request
        if problem_name.upper() == "BLOCK" and dim == 2064 and type_instance == 16:
            continue
        # Try to load summary from experiments folder if exists
        avg_score = None
        exp_summary = None
        if exp_dir and exp_dir.exists():
            exp_summary = load_best_summary(exp_dir, problem_name)
            if exp_summary:
                avg_score = parse_float(exp_summary.get("avg_score"))

        ranking_file = ranking_path(problem_name, dim, type_instance)
        ranking = load_ranking(ranking_file)
        ranking = [
            (name, score)
            for name, score in ranking
            if name not in {"PPO-EDA", "Tabu", "TABU"}
        ]

        row: List[str] = [problem_name, str(dim), str(type_instance)]

        # Load SVGD score and optional rank from config
        svgd_raw_score, svgd_rank_from_config = load_svgd_score(config_dir, problem_name, dim, type_instance)
        svgd_score = normalize_score(problem_name, svgd_raw_score)

        # Build combined ranking including SVGD
        combined = list(ranking)
        if svgd_score is not None:
            combined.append(("SVGD", svgd_score))
        combined.sort(key=lambda item: item[1], reverse=True)
        total = len(combined) if combined else None

        rank_map: dict[str, tuple[int, float]] = {}
        for idx, (name, score) in enumerate(combined, start=1):
            rank_map[name] = (idx, score)

        svgd_rank = rank_map.get("SVGD", (None, None))[0]
        row.extend([format_rank(svgd_rank, total), format_score(problem_name, svgd_score)])

        # Add other methods
        for method in methods:
            rank_entry = rank_map.get(method)
            if rank_entry:
                rank, score = rank_entry
                row.extend([format_rank(rank, total), format_score(problem_name, score)])
            else:
                row.extend(["—", "—"])

        # Best method (excluding SVGD, PBIL, MIMIC, BOA)
        best_name = "—"
        best_rank = None
        best_score = None
        if combined:
            excluded = {"SVGD"}
            excluded.update(methods)
            for idx, (name, score) in enumerate(combined, start=1):
                if name in excluded:
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


def write_csv(rows: List[List[str]], col_labels: List[str], output_csv: Path) -> None:
    if not rows:
        print("[WARN] No rows to write.")
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(col_labels)
        writer.writerows(rows)
    print(f"Saved table to {output_csv}")


def write_excel(rows: List[List[str]], col_labels: List[str], output_xlsx: Path) -> None:
    if not rows:
        print("[WARN] No rows to write.")
        return
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
    except ImportError:
        print("[WARN] openpyxl is not installed. Install it or use --format csv.")
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "summary"

    # Column groups (0-based indices)
    # Instances: Pb, n, t -> cols 1-3
    # Methods: SVGD, PBIL, MIMIC, BOA -> each has Rank/Score
    # Best method (others): Name, Rank, Score -> last 3
    n_cols = len(col_labels)
    if n_cols != 14:
        print(f"[WARN] Unexpected column count: {n_cols}, expected 14. Excel header merging may be off.")

    # Header rows
    ws.append([""] * n_cols)
    ws.append([""] * n_cols)
    ws.append(col_labels)

    # Group headers (row 1)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=3)
    ws.cell(row=1, column=1, value="Instances")
    ws.merge_cells(start_row=1, start_column=4, end_row=1, end_column=11)
    ws.cell(row=1, column=4, value="Methods")
    ws.merge_cells(start_row=1, start_column=12, end_row=1, end_column=14)
    ws.cell(row=1, column=12, value="Best method (others)")

    # Method headers (row 2)
    ws.merge_cells(start_row=2, start_column=4, end_row=2, end_column=5)
    ws.cell(row=2, column=4, value="SVGD")
    ws.merge_cells(start_row=2, start_column=6, end_row=2, end_column=7)
    ws.cell(row=2, column=6, value="PBIL")
    ws.merge_cells(start_row=2, start_column=8, end_row=2, end_column=9)
    ws.cell(row=2, column=8, value="MIMIC")
    ws.merge_cells(start_row=2, start_column=10, end_row=2, end_column=11)
    ws.cell(row=2, column=10, value="BOA")

    # Data rows
    for row in rows:
        ws.append(row)

    bold = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row_idx in range(1, 4):
        for col_idx in range(1, n_cols + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = bold
            cell.alignment = center

    for row_idx in range(4, 4 + len(rows)):
        for col_idx in range(1, n_cols + 1):
            ws.cell(row=row_idx, column=col_idx).alignment = center

    col_widths = [8, 6, 6] + [9, 9] * 4 + [24, 9, 9]
    for idx, width in enumerate(col_widths, start=1):
        ws.column_dimensions[chr(64 + idx)].width = width

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_xlsx)
    print(f"Saved table to {output_xlsx}")


def plot_table(rows: List[List[str]], col_labels: List[str], output_png: Path, output_pdf: Path) -> None:
    if not rows:
        print("[WARN] No rows to render.")
        return

    n_rows = len(rows)
    fig_height = max(8.0, 0.42 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(18, fig_height), dpi=220)
    ax.axis("off")
    ax.set_position([0.02, 0.02, 0.96, 0.96])

    col_widths = [0.06, 0.04, 0.04] + [0.06, 0.06] * 4 + [0.24, 0.06, 0.06]
    total_width = sum(col_widths)
    col_widths = [w / total_width for w in col_widths]

    n_cols = len(col_labels)
    table_bbox = [0, 0, 1, 0.82]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        bbox=table_bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for col_idx, width in enumerate(col_widths):
        for row_idx in range(n_rows + 1):
            cell = table.get_celld().get((row_idx, col_idx))
            if cell:
                cell.set_width(width)

    line_counts: List[int] = [1]
    for row in rows:
        max_lines = 1
        for value in row:
            max_lines = max(max_lines, str(value).count("\n") + 1)
        line_counts.append(max_lines)
    total_lines = sum(line_counts)
    bbox_height = table_bbox[3]
    for row_idx, line_count in enumerate(line_counts):
        row_height = bbox_height * (line_count / total_lines)
        for col_idx in range(n_cols):
            cell = table.get_celld().get((row_idx, col_idx))
            if cell:
                cell.set_height(row_height)
                if row_idx == 0:
                    cell.get_text().set_fontweight("bold")

    rows_problem = [row[0] for row in rows]
    for idx in range(1, n_rows):
        if rows_problem[idx] != rows_problem[idx - 1]:
            cell = table.get_celld().get((idx + 1, 0))
            if cell:
                ax.hlines(cell.get_y(), 0, 1, transform=ax.transAxes, color="#666666", linewidth=0.6)

    x_edges = [0.0]
    for width in col_widths:
        x_edges.append(x_edges[-1] + width)

    def center_between(start: int, end: int) -> float:
        return (x_edges[start] + x_edges[end]) / 2

    header_bottom = table_bbox[1] + table_bbox[3]
    header2_h = 0.06
    header1_h = 0.07
    header2_y = header_bottom
    header1_y = header_bottom + header2_h

    def draw_header_cell(x0: float, x1: float, y0: float, h: float, text: str, fontsize: int) -> None:
        rect = plt.Rectangle(
            (x0, y0),
            x1 - x0,
            h,
            fill=False,
            linewidth=0.8,
            edgecolor="#333333",
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        if text:
            ax.text(
                (x0 + x1) / 2,
                y0 + h / 2,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                transform=ax.transAxes,
            )

    draw_header_cell(x_edges[0], x_edges[3], header1_y, header1_h, "Instances", 11)
    draw_header_cell(x_edges[3], x_edges[11], header1_y, header1_h, "Methods", 11)
    draw_header_cell(x_edges[11], x_edges[14], header1_y, header1_h, "Best method (others)", 11)

    draw_header_cell(x_edges[0], x_edges[3], header2_y, header2_h, "", 9)
    method_labels = ["SVGD", "PBIL", "MIMIC", "BOA"]
    method_spans = [(3, 5), (5, 7), (7, 9), (9, 11)]
    for label, (start, end) in zip(method_labels, method_spans):
        draw_header_cell(x_edges[start], x_edges[end], header2_y, header2_h, label, 9)
    draw_header_cell(x_edges[11], x_edges[14], header2_y, header2_h, "", 9)

    ax.hlines(header_bottom, 0, 1, transform=ax.transAxes, color="#333333", linewidth=0.8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Saved table to {output_png} and {output_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summary table for all instances.")
    parser.add_argument("--png", type=Path, default=ROOT / "courbes" / "summary_table.png")
    parser.add_argument("--pdf", type=Path, default=ROOT / "courbes" / "summary_table.pdf")
    parser.add_argument("--csv", type=Path, default=ROOT / "courbes" / "summary_table.csv")
    parser.add_argument("--xlsx", type=Path, default=ROOT / "courbes" / "summary_table.xlsx")
    parser.add_argument("--format", choices=("all", "csv", "image", "excel"), default="all")
    args = parser.parse_args()

    # Ask for config
    config_name = input("Enter config name (e.g., krbf__advnormalizedfitness__M2__L14__eps0p025__g0p007__ds0p15__dm0p01): ").strip()
    config_dir = ROOT / "results" / "config" / config_name
    
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        return

    methods = ["PBIL", "MIMIC", "BOA"]
    rows, col_labels = build_rows(methods, config_dir)
    if args.format in ("all", "csv"):
        write_csv(rows, col_labels, args.csv)
    if args.format in ("all", "image"):
        plot_table(rows, col_labels, args.png, args.pdf)
    if args.format in ("all", "excel"):
        write_excel(rows, col_labels, args.xlsx)


if __name__ == "__main__":
    main()
