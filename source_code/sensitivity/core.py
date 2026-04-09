from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import cm


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "results" / "config"
DEFAULT_M_VALUES = list(range(1, 17))


def replace_m(config_name: str, new_m: int) -> str:
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


def resolve_config_name(config_input: str) -> str:
    config_input = str(config_input).strip()
    if not config_input:
        raise ValueError("missing config name.")
    config_path = Path(config_input)
    if config_path.is_dir():
        return config_path.name
    return config_input


def iter_data_lines(path: Path) -> Iterable[str]:
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            yield line


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_last_metrics_row(csv_path: Path) -> Optional[Dict[str, str]]:
    if not csv_path.exists():
        return None
    try:
        reader = csv.reader(iter_data_lines(csv_path))
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


def get_first_float(row: Dict[str, str], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in row and row[key] != "":
            value = parse_float(row[key])
            if value is not None:
                return value
    return None


def build_box_stats(row: Dict[str, str]) -> Optional[Dict[str, float]]:
    q1 = get_first_float(row, ["25%", "q1"])
    med = get_first_float(row, ["50%", "median"])
    q3 = get_first_float(row, ["75%", "q3"])
    if q1 is None or med is None or q3 is None:
        return None

    whislo = get_first_float(row, ["2%", "5%", "min"]) or q1
    whishi = get_first_float(row, ["98%", "95%", "max"]) or q3
    mean = get_first_float(row, ["mean", "avg_score", "avg", "score_mean"])

    return {
        "q1": q1,
        "med": med,
        "q3": q3,
        "whislo": whislo,
        "whishi": whishi,
        "mean": mean,
        "fliers": [],
    }


def abs_transform(stats: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for key, value in stats.items():
        if value is None:
            out[key] = value
        elif isinstance(value, (int, float)):
            out[key] = abs(value)
        else:
            out[key] = value
    if out.get("q1") is not None and out.get("q3") is not None and out["q1"] > out["q3"]:
        out["q1"], out["q3"] = out["q3"], out["q1"]
    if out.get("whislo") is not None and out.get("whishi") is not None and out["whislo"] > out["whishi"]:
        out["whislo"], out["whishi"] = out["whishi"], out["whislo"]
    return out


def find_variants(
    config_name: str,
    config_root: Path = CONFIG_ROOT,
    m_values: Iterable[int] = DEFAULT_M_VALUES,
) -> List[Tuple[int, Path]]:
    variants: List[Tuple[int, Path]] = []
    for m in m_values:
        name = replace_m(config_name, int(m))
        path = config_root / name
        if path.exists():
            variants.append((int(m), path))
    return variants


def load_stats_for_config(
    config_dir: Path,
    folder_pattern: str,
    key_groups: Tuple[str, str] = ("n", "t"),
    abs_values: bool = False,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    stats_by_instance: Dict[Tuple[int, int], Dict[str, float]] = {}
    regex = re.compile(folder_pattern)
    g0, g1 = key_groups
    for entry in config_dir.iterdir():
        if not entry.is_dir():
            continue
        match = regex.match(entry.name)
        if not match:
            continue
        k0 = int(match.group(g0))
        k1 = int(match.group(g1))
        row = read_last_metrics_row(entry / "best_metrics.csv")
        if not row:
            continue
        stats = build_box_stats(row)
        if not stats:
            continue
        if abs_values:
            stats = abs_transform(stats)
        stats_by_instance[(k0, k1)] = stats
    return stats_by_instance


def collect_stats_map(
    variants: Iterable[Tuple[int, Path]],
    loader: Callable[[Path], Dict[Tuple[int, int], Dict[str, float]]],
) -> Dict[Tuple[int, int], Dict[int, Dict[str, float]]]:
    stats_map: Dict[Tuple[int, int], Dict[int, Dict[str, float]]] = {}
    for m, cfg_dir in variants:
        per_instance = loader(cfg_dir)
        for key, stats in per_instance.items():
            stats_map.setdefault(key, {})[m] = stats
    return stats_map


def build_summary_rows(
    problem_label: str,
    stats_map: Dict[Tuple[int, int], Dict[int, Dict[str, float]]],
) -> List[List[str]]:
    rows: List[List[str]] = []
    for (k0, k1), per_m in sorted(stats_map.items(), key=lambda item: (item[0][0], item[0][1])):
        for m in sorted(per_m.keys()):
            stats = per_m[m]
            rows.append(
                [
                    problem_label,
                    str(k0),
                    str(k1),
                    str(m),
                    f"{stats.get('mean'):.6f}" if stats.get("mean") is not None else "",
                    f"{stats.get('med'):.6f}" if stats.get("med") is not None else "",
                    f"{stats.get('q1'):.6f}" if stats.get("q1") is not None else "",
                    f"{stats.get('q3'):.6f}" if stats.get("q3") is not None else "",
                    f"{stats.get('whislo'):.6f}" if stats.get("whislo") is not None else "",
                    f"{stats.get('whishi'):.6f}" if stats.get("whishi") is not None else "",
                ]
            )
    return rows


def write_summary_csv(rows: List[List[str]], output_csv: Path, key_labels: Tuple[str, str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "problem",
                key_labels[0],
                key_labels[1],
                "M",
                "mean",
                "median",
                "q1",
                "q3",
                "p2",
                "p98",
            ]
        )
        writer.writerows(rows)


def plot_boxplots(
    stats_map: Dict[Tuple[int, int], Dict[int, Dict[str, float]]],
    output_dir: Path,
    title_fn: Callable[[int, int], str],
    filename_fn: Callable[[int, int], str],
    x_label: str = "Number of Agents (m)",
    y_label: str = "Fitness",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for (k0, k1), per_m in sorted(stats_map.items(), key=lambda item: (item[0][0], item[0][1])):
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

        half_w = box_width / 2
        for m, stats in zip(m_values, stats_list):
            mean = stats.get("mean")
            if mean is not None:
                ax.hlines(mean, m - half_w, m + half_w, colors="#000000", linewidth=2.0, zorder=3)

        ax.set_title(title_fn(k0, k1), fontsize=18, fontweight="bold", pad=20)
        ax.set_xlabel(x_label, fontsize=20, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=20, fontweight="bold")
        ax.set_xticks(m_values)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        ax.set_axisbelow(True)
        plt.tight_layout()

        fig.savefig(output_dir / filename_fn(k0, k1), dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_sensitivity_from_config(
    config_input: str,
    *,
    problem_label: str,
    folder_pattern: str,
    summary_filename: str,
    key_labels: Tuple[str, str],
    title_fn: Callable[[int, int], str],
    filename_fn: Callable[[int, int], str],
    abs_values: bool = False,
    key_groups: Tuple[str, str] = ("n", "t"),
    m_values: Iterable[int] = DEFAULT_M_VALUES,
) -> Tuple[Path, Path]:
    config_name = resolve_config_name(config_input)
    variants = find_variants(config_name, CONFIG_ROOT, m_values)
    if not variants:
        raise ValueError(f"No config variants found under {CONFIG_ROOT}")

    def _loader(path: Path) -> Dict[Tuple[int, int], Dict[str, float]]:
        return load_stats_for_config(
            path,
            folder_pattern=folder_pattern,
            key_groups=key_groups,
            abs_values=abs_values,
        )

    stats_map = collect_stats_map(variants, _loader)
    rows = build_summary_rows(problem_label, stats_map)

    output_dir = ROOT / "results" / "config" / config_name / "sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / summary_filename
    write_summary_csv(rows, summary_csv, key_labels=key_labels)
    plot_dir = output_dir / "boxplots"
    plot_boxplots(
        stats_map,
        plot_dir,
        title_fn=title_fn,
        filename_fn=filename_fn,
    )
    return summary_csv, plot_dir
