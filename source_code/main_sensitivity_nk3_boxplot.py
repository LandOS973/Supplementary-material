#!/usr/bin/env python3
"""
Plot sensitivity boxplots for NK3 from existing config results.

This script does NOT run experiments. It:
1) Asks for a config name that contains an "__M<value>" segment.
2) Searches for M=1..20 variants under results/config.
3) For each NK3 instance folder (NK3_dimN_tK), reads the LAST row of best_metrics.csv.
4) Builds boxplots using the stored quantiles (2%, 25%, 50%, 75%, 98%) and mean.
5) Writes a summary CSV + saves plots.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = ROOT / "results" / "config"
DEFAULT_M_VALUES = list(range(1, 17))


# -----------------------------
# Helpers
# -----------------------------

def _replace_m(config_name: str, new_m: int) -> str:
    parts = config_name.split("__")
    for idx, part in enumerate(parts):
        if re.fullmatch(r"M\d+", part):
            parts[idx] = f"M{new_m}"
            return "__".join(parts)

    updated = re.sub(r"(?:(?<=^)|(?<=__))M\d+(?=(?:__|$))", f"M{new_m}", config_name, count=1)
    if updated != config_name:
        return updated

    updated = re.sub(r"M\d+", f"M{new_m}", config_name, count=1)
    if updated != config_name:
        return updated

    raise ValueError("Config name must contain an '__M<value>' segment.")


def _iter_data_lines(path: Path) -> Iterable[str]:
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            yield line


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_last_metrics_row(csv_path: Path) -> Optional[Dict[str, str]]:
    if not csv_path.exists():
        return None
    try:
        reader = csv.reader(_iter_data_lines(csv_path))
        header = next(reader, None)
        if not header:
            return None
        last_row = None
        for row in reader:
            last_row = row
        if not last_row:
            return None
        return dict(zip(header, last_row))
    except Exception:
        return None


def _get_first_float(row: Dict[str, str], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in row and row[key] != "":
            value = _parse_float(row[key])
            if value is not None:
                return value
    return None


def _build_box_stats(row: Dict[str, str]) -> Optional[Dict[str, float]]:
    q1 = _get_first_float(row, ["25%", "q1"])
    med = _get_first_float(row, ["50%", "median"])
    q3 = _get_first_float(row, ["75%", "q3"])
    if q1 is None or med is None or q3 is None:
        return None

    whislo = _get_first_float(row, ["2%", "5%", "min"]) or q1
    whishi = _get_first_float(row, ["98%", "95%", "max"]) or q3
    mean = _get_first_float(row, ["mean", "avg_score", "avg", "score_mean"])

    return {
        "q1": q1,
        "med": med,
        "q3": q3,
        "whislo": whislo,
        "whishi": whishi,
        "mean": mean,
        "fliers": [],
    }


# -----------------------------
# Data loading
# -----------------------------

def load_nk3_stats_for_config(config_dir: Path) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Return {(N, K): stats} for a single config directory."""
    stats_by_instance: Dict[Tuple[int, int], Dict[str, float]] = {}
    for entry in config_dir.iterdir():
        if not entry.is_dir():
            continue
        match = re.match(r"^NK3_dim(?P<n>\d+)_t(?P<t>\d+)$", entry.name)
        if not match:
            continue
        n = int(match.group("n"))
        k = int(match.group("t"))
        row = _read_last_metrics_row(entry / "best_metrics.csv")
        if not row:
            continue
        stats = _build_box_stats(row)
        if not stats:
            continue
        stats_by_instance[(n, k)] = stats
    return stats_by_instance


# -----------------------------
# Plotting + CSV
# -----------------------------

def write_summary_csv(rows: List[List[str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "problem",
            "N",
            "K",
            "M",
            "mean",
            "median",
            "q1",
            "q3",
            "p2",
            "p98",
        ])
        writer.writerows(rows)


def plot_boxplots(stats_map: Dict[Tuple[int, int], Dict[int, Dict[str, float]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for (n, k), per_m in sorted(stats_map.items(), key=lambda item: (item[0][0], item[0][1])):
        m_values = sorted(per_m.keys())
        if not m_values:
            continue
        stats_list = []
        for m in m_values:
            stats = dict(per_m[m])
            stats["label"] = str(m)
            stats_list.append(stats)

        fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
        box_width = 0.6
        box = ax.bxp(
            stats_list,
            positions=m_values,
            showmeans=False,
            patch_artist=True,
            widths=box_width,
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            medianprops=dict(color="none", linewidth=0.0),
        )

        colors = cm.get_cmap("viridis", max(len(m_values), 1))
        for patch, idx in zip(box["boxes"], range(len(m_values))):
            patch.set_facecolor(colors(idx))
            patch.set_alpha(0.85)

        # Draw mean (black) bar across the box width
        half_w = box_width / 2
        for m, stats in zip(m_values, stats_list):
            mean = stats.get("mean")
            if mean is not None:
                ax.hlines(mean, m - half_w, m + half_w, colors="#000000", linewidth=2.0, zorder=3)

        ax.set_title(f"NK3 (N={n}, K={k})", fontsize=18, fontweight="bold", pad=20)
        ax.set_xlabel("Number of Agents (m)", fontsize=20, fontweight="bold")
        ax.set_ylabel("Fitness", fontsize=20, fontweight="bold")
        ax.set_xticks(m_values)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        ax.set_axisbelow(True)
        plt.tight_layout()

        plot_path = output_dir / f"boxplot_NK3_{n}_{k}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# -----------------------------
# Entry point
# -----------------------------

def main() -> None:
    config_input = input("Enter config name (with __M<value>__): ").strip()
    if not config_input:
        print("Error: missing config name.")
        return

    config_path = Path(config_input)
    if config_path.is_dir():
        config_name = config_path.name
    else:
        config_name = config_input

    # Discover variants across M
    variants: List[Tuple[int, Path]] = []
    for m in DEFAULT_M_VALUES:
        try:
            name = _replace_m(config_name, m)
        except ValueError as exc:
            print(f"Error: {exc}")
            return
        path = CONFIG_ROOT / name
        if path.exists():
            variants.append((m, path))

    if not variants:
        print(f"No config variants found under {CONFIG_ROOT}")
        return

    # Build stats per (N, K) and per M
    stats_map: Dict[Tuple[int, int], Dict[int, Dict[str, float]]] = {}
    summary_rows: List[List[str]] = []

    for m, cfg_dir in variants:
        per_instance = load_nk3_stats_for_config(cfg_dir)
        for (n, k), stats in per_instance.items():
            stats_map.setdefault((n, k), {})[m] = stats

            summary_rows.append([
                "NK3",
                str(n),
                str(k),
                str(m),
                f"{stats.get('mean'):.6f}" if stats.get("mean") is not None else "",
                f"{stats.get('med'):.6f}" if stats.get("med") is not None else "",
                f"{stats.get('q1'):.6f}" if stats.get("q1") is not None else "",
                f"{stats.get('q3'):.6f}" if stats.get("q3") is not None else "",
                f"{stats.get('whislo'):.6f}" if stats.get("whislo") is not None else "",
                f"{stats.get('whishi'):.6f}" if stats.get("whishi") is not None else "",
            ])

    output_dir = ROOT / "results" / "config" / config_name / "sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "sensitivity_nk3_from_config.csv"
    write_summary_csv(summary_rows, summary_csv)

    plot_boxplots(stats_map, output_dir / "boxplots")

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved boxplots to: {output_dir / 'boxplots'}")


if __name__ == "__main__":
    main()
