#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import pandas as pd
from openpyxl.chart import BarChart, Reference


GROUP_RE = re.compile(r"^(?P<problem>[A-Za-z0-9]+)_dim(?P<dim>\d+)_t(?P<type_instance>\d+)$")
DUMMY_RE = re.compile(r"^dummy(?P<dummy_blocks>\d+)$")


def _parse_summary_config(summary_path):
    cfg = {}
    try:
        with open(summary_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if not value:
                    continue
                lowered = value.lower()
                if lowered in ("none", "null", "n/a"):
                    parsed = None
                elif lowered in ("true", "false"):
                    parsed = lowered == "true"
                else:
                    try:
                        if "." in value:
                            parsed = float(value)
                        else:
                            parsed = int(value)
                    except ValueError:
                        parsed = value
                cfg[key] = parsed
    except OSError:
        return None
    return cfg


def _is_maximization_problem(problem_type: str) -> bool:
    return str(problem_type).upper() in ("NK", "BLOCK")


def _parse_group_from_path(path: Path):
    problem = None
    dim = None
    type_instance = None
    dummy_blocks = 0
    mode = "standard"
    no_interact = False

    parts = list(path.parts)
    for part in parts:
        match = GROUP_RE.match(part)
        if match:
            problem = match.group("problem")
            dim = int(match.group("dim"))
            type_instance = int(match.group("type_instance"))
        dummy_match = DUMMY_RE.match(part)
        if dummy_match:
            dummy_blocks = int(dummy_match.group("dummy_blocks"))
        if part == "no_interact":
            no_interact = True
        if part == "decay":
            mode = "decay"
    if problem is None or dim is None or type_instance is None:
        return None
    return {
        "problem": problem,
        "dim": dim,
        "type_instance": type_instance,
        "dummy_blocks": dummy_blocks,
        "no_interact": no_interact,
        "mode": mode,
    }


def _is_better(problem_type: str, score: float, best_score: float) -> bool:
    if best_score is None:
        return True
    if _is_maximization_problem(problem_type):
        return score > best_score
    return score < best_score


def main():
    parser = argparse.ArgumentParser(description="Review kernels and export results to Excel.")
    parser.add_argument(
        "--summary_root",
        type=str,
        default=None,
        help="Root folder containing *_best_summary.txt files (default: results/experiments).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output Excel path (default: courbes/review_kernels.xlsx).",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    summary_root = args.summary_root or os.path.join(repo_root, "results", "experiments")
    out_path = args.out or os.path.join(repo_root, "courbes", "review_kernels.xlsx")

    summary_root_path = Path(summary_root)
    summary_files = list(summary_root_path.rglob("*_best_summary.txt"))

    all_rows = []
    skipped = 0
    for path in summary_files:
        group = _parse_group_from_path(path)
        cfg = _parse_summary_config(path)
        if not group or not cfg:
            skipped += 1
            continue
        if str(group.get("problem", "")).upper() == "BLOCK":
            skipped += 1
            continue
        if group.get("mode") != "standard" or group.get("no_interact"):
            skipped += 1
            continue
        kernel = cfg.get("kernel")
        avg_score = cfg.get("avg_score")
        if kernel is None or avg_score is None:
            skipped += 1
            continue
        row = {
            **group,
            "kernel": str(kernel).lower(),
            "avg_score": float(avg_score),
            "summary_path": str(path),
        }
        if "m" in cfg:
            row["M"] = cfg.get("m")
        if "lambda" in cfg:
            row["lambda"] = cfg.get("lambda")
        if "epsilon_svgd" in cfg:
            row["epsilon_svgd"] = cfg.get("epsilon_svgd")
        if "gamma" in cfg:
            row["gamma"] = cfg.get("gamma")
        all_rows.append(row)

    if not all_rows:
        raise SystemExit(f"Aucun resume trouve dans {summary_root}.")

    all_df = pd.DataFrame(all_rows)
    all_df = all_df.drop(columns=["bandwith_kernel", "advantage"], errors="ignore")
    all_df = all_df.drop(columns=["dummy_blocks", "no_interact"], errors="ignore")

    best_rows = {}
    for _, row in all_df.iterrows():
        key = (
            row["problem"],
            int(row["dim"]),
            int(row["type_instance"]),
            int(row.get("dummy_blocks", 0)),
            bool(row.get("no_interact", False)),
            row.get("mode", "standard"),
        )
        current = best_rows.get(key)
        if current is None:
            best_rows[key] = row
            continue
        if _is_better(row["problem"], row["avg_score"], current["avg_score"]):
            best_rows[key] = row

    best_df = pd.DataFrame(best_rows.values())
    best_df = best_df.drop(columns=["dummy_blocks", "no_interact"], errors="ignore")
    total_instances = len(best_df)

    kernel_counts = (
        best_df.groupby("kernel")
        .size()
        .reindex(["hk", "jsd", "pk", "rbf"])
        .fillna(0)
        .astype(int)
        .reset_index(name="count")
    )
    kernel_counts["%"] = kernel_counts["count"] / total_instances * 100.0

    m_counts = (
        best_df.dropna(subset=["M"])
        .groupby("M")
        .size()
        .reset_index(name="count")
        .sort_values(by="M")
    )
    m_counts["%"] = m_counts["count"] / total_instances * 100.0

    lambda_counts = (
        best_df.dropna(subset=["lambda"])
        .groupby("lambda")
        .size()
        .reset_index(name="count")
        .sort_values(by="lambda")
    )
    lambda_counts["%"] = lambda_counts["count"] / total_instances * 100.0

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        best_df.sort_values(by=["problem", "dim", "type_instance", "kernel"]).to_excel(
            writer, sheet_name="best_by_instance", index=False
        )
        kernel_counts.to_excel(writer, sheet_name="kernel_share", index=False)
        all_df.sort_values(by=["problem", "dim", "type_instance", "kernel"]).to_excel(
            writer, sheet_name="all_kernels", index=False
        )
        m_counts.to_excel(writer, sheet_name="M_hist", index=False)
        lambda_counts.to_excel(writer, sheet_name="lambda_hist", index=False)

        workbook = writer.book

        kernel_sheet = writer.sheets["kernel_share"]
        kernel_chart = BarChart()
        kernel_chart.title = "Kernel (count)"
        kernel_chart.y_axis.title = "count"
        data = Reference(kernel_sheet, min_col=2, min_row=1, max_row=kernel_sheet.max_row)
        cats = Reference(kernel_sheet, min_col=1, min_row=2, max_row=kernel_sheet.max_row)
        kernel_chart.add_data(data, titles_from_data=True)
        kernel_chart.set_categories(cats)
        kernel_sheet.add_chart(kernel_chart, "E2")

        m_sheet = writer.sheets["M_hist"]
        m_chart = BarChart()
        m_chart.title = "M (count)"
        m_chart.y_axis.title = "count"
        data = Reference(m_sheet, min_col=2, min_row=1, max_row=m_sheet.max_row)
        cats = Reference(m_sheet, min_col=1, min_row=2, max_row=m_sheet.max_row)
        m_chart.add_data(data, titles_from_data=True)
        m_chart.set_categories(cats)
        m_sheet.add_chart(m_chart, "E2")

        lambda_sheet = writer.sheets["lambda_hist"]
        lambda_chart = BarChart()
        lambda_chart.title = "Lambda (count)"
        lambda_chart.y_axis.title = "count"
        data = Reference(lambda_sheet, min_col=2, min_row=1, max_row=lambda_sheet.max_row)
        cats = Reference(lambda_sheet, min_col=1, min_row=2, max_row=lambda_sheet.max_row)
        lambda_chart.add_data(data, titles_from_data=True)
        lambda_chart.set_categories(cats)
        lambda_sheet.add_chart(lambda_chart, "E2")

    print(f"Ecrit: {out_path}")
    print(f"Instances traitees: {total_instances}")
    if skipped:
        print(f"Resumes ignores: {skipped}")


if __name__ == "__main__":
    main()
