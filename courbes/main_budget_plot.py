"""Plot interact vs no_interact curves for each budget from config."""

from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parent.parent


def _load_problem_config(problem_override: str | None = None) -> tuple[str, int, int]:
    config_path = ROOT / "config" / "config.yaml"
    problem_name = "QUBO"
    dim = 256
    type_instance = 4
    try:
        cfg = OmegaConf.load(config_path)
        problem_key = problem_override or cfg.get("problem")
        if not problem_key:
            defaults = cfg.get("defaults") or []
            for entry in defaults:
                if isinstance(entry, (dict, DictConfig)) and "problem" in entry:
                    problem_key = entry["problem"]
                    break
        problem_key = problem_key or "qubo"
        problem_path = ROOT / "config" / "problem" / f"{problem_key}.yaml"
        problem_cfg = OmegaConf.load(problem_path)
        problem_name = str(problem_cfg.get("name") or problem_cfg.get("type_problem") or problem_name)
        dim_value = problem_cfg.get("dim")
        if dim_value is None:
            dim_value = problem_cfg.get("n")
        if dim_value is not None:
            dim = int(dim_value)
        type_value = problem_cfg.get("type_instance")
        if type_value is None:
            type_value = problem_cfg.get("k")
        if type_value is not None:
            type_instance = int(type_value)
    except OSError:
        pass
    return problem_name, dim, type_instance


def _build_instance_context(problem_override: str | None = None) -> dict[str, Path | str | int]:
    problem_name, dim, type_instance = _load_problem_config(problem_override)
    return _build_instance_context_from_values(problem_name, dim, type_instance)


def _build_instance_context_from_values(
    problem_name: str, dim: int, type_instance: int
) -> dict[str, Path | str | int]:
    experiment_dir = ROOT / "results" / "experiments" / f"{problem_name}_dim{dim}_t{type_instance}"
    if problem_name.upper() == "QUBO":
        instance_name = f"UBQP_{dim}_{type_instance}"
    else:
        instance_name = f"{problem_name}_{dim}_{type_instance}"
    output_dir = ROOT / "courbes" / instance_name
    return {
        "problem_name": problem_name,
        "dim": dim,
        "type_instance": type_instance,
        "instance_name": instance_name,
        "experiment_dir": experiment_dir,
        "output_dir": output_dir,
    }


def _read_csv_without_comments(path: Path) -> csv.DictReader:
    with path.open(newline="") as handle:
        lines = [line for line in handle if not line.lstrip().startswith("#")]
    return csv.DictReader(io.StringIO("".join(lines)))


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _sort_by_x(x_vals: List[float], y_vals: List[float]) -> Tuple[List[float], List[float]]:
    pairs = sorted(zip(x_vals, y_vals), key=lambda pair: pair[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def load_metric_series(path: Path, x_field: str, y_field: str) -> Tuple[List[float], List[float]]:
    x_vals: List[float] = []
    y_vals: List[float] = []
    reader = _read_csv_without_comments(path)
    for row in reader:
        x_val = _parse_float(row.get(x_field))
        y_val = _parse_float(row.get(y_field))
        if x_val is None or y_val is None:
            continue
        x_vals.append(x_val)
        y_vals.append(y_val)
    if not x_vals:
        raise ValueError(f"No data for {y_field} in {path}")
    return _sort_by_x(x_vals, y_vals)


def load_metric_series_with_std(
    path: Path, x_field: str, mean_field: str = "mean", std_field: str = "std"
) -> Tuple[List[float], List[float], List[float] | None]:
    x_vals: List[float] = []
    mean_vals: List[float] = []
    std_vals: List[float] = []
    reader = _read_csv_without_comments(path)
    fieldnames = reader.fieldnames or []
    mean_key = mean_field if mean_field in fieldnames else "best_fitness"
    if mean_key not in fieldnames:
        mean_key = mean_field
    std_key = std_field if std_field in fieldnames else None
    for row in reader:
        x_val = _parse_float(row.get(x_field))
        mean_val = _parse_float(row.get(mean_key))
        std_val = _parse_float(row.get(std_key)) if std_key else None
        if x_val is None or mean_val is None:
            continue
        x_vals.append(x_val)
        mean_vals.append(mean_val)
        if std_key and std_val is not None:
            std_vals.append(std_val)
    if not x_vals:
        raise ValueError(f"No data for {mean_key} in {path}")
    if std_key:
        if not std_vals:
            raise ValueError(f"No data for {std_key} in {path}")
        pairs = sorted(zip(x_vals, mean_vals, std_vals), key=lambda pair: pair[0])
        return [p[0] for p in pairs], [p[1] for p in pairs], [p[2] for p in pairs]
    x_vals, mean_vals = _sort_by_x(x_vals, mean_vals)
    return x_vals, mean_vals, None


def load_quantile_stats_at_final_step(
    path: Path,
    *,
    x_field: str = "step",
    low_field: str = "2%",
    q1_field: str = "25%",
    median_field: str = "50%",
    q3_field: str = "75%",
    high_field: str = "98%",
    mean_field: str = "mean",
) -> dict[str, float] | None:
    reader = _read_csv_without_comments(path)
    fieldnames = set(reader.fieldnames or [])
    required = {x_field, low_field, q1_field, median_field, q3_field, high_field}
    if not required.issubset(fieldnames):
        return None

    rows: List[dict[str, str]] = []
    for row in reader:
        if row.get(x_field) is None:
            continue
        rows.append(row)
    if not rows:
        return None

    def _step_value(r: dict[str, str]) -> float:
        val = _parse_float(r.get(x_field))
        return val if val is not None else float("-inf")

    last_row = max(rows, key=_step_value)
    values = {
        "low": _parse_float(last_row.get(low_field)),
        "q1": _parse_float(last_row.get(q1_field)),
        "med": _parse_float(last_row.get(median_field)),
        "q3": _parse_float(last_row.get(q3_field)),
        "high": _parse_float(last_row.get(high_field)),
    }
    if any(val is None for val in values.values()):
        return None
    mean_val = _parse_float(last_row.get(mean_field))
    if mean_val is not None:
        values["mean"] = float(mean_val)
    return {key: float(val) for key, val in values.items()}


def style_axes(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9, width=0.6, length=3)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.5, alpha=0.4)


def _plot_metric_pair_values(
    x_left: List[float],
    y_left: List[float],
    x_right: List[float],
    y_right: List[float],
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    left_label: str,
    right_label: str,
    right_linestyle: str = "--",
    std_left: List[float] | None = None,
    std_right: List[float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=180)
    ax.plot(
        x_left,
        y_left,
        label=left_label,
        color="#1f77b4",
        linewidth=1.4,
        alpha=0.95,
    )
    ax.plot(
        x_right,
        y_right,
        label=right_label,
        color="#ff7f0e",
        linewidth=1.4,
        linestyle=right_linestyle,
        alpha=0.95,
    )
    if std_left is not None:
        lower = [y - s for y, s in zip(y_left, std_left)]
        upper = [y + s for y, s in zip(y_left, std_left)]
        ax.fill_between(x_left, lower, upper, color="#1f77b4", alpha=0.2, linewidth=0.0)
    if std_right is not None:
        lower = [y - s for y, s in zip(y_right, std_right)]
        upper = [y + s for y, s in zip(y_right, std_right)]
        ax.fill_between(x_right, lower, upper, color="#ff7f0e", alpha=0.2, linewidth=0.0)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Evaluations", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax, grid_axis="both")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def _plot_metric_triplet(
    normal_path: Path,
    decay_path: Path,
    no_interact_path: Path,
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    metric_field: str,
    square_values: bool = False,
) -> None:
    x_norm, y_norm = load_metric_series(normal_path, x_field="step", y_field=metric_field)
    x_decay, y_decay = load_metric_series(decay_path, x_field="step", y_field=metric_field)
    x_no, y_no = load_metric_series(no_interact_path, x_field="step", y_field=metric_field)
    if square_values:
        y_int = [val * val for val in y_int]
        y_no = [val * val for val in y_no]

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=180)
    ax.plot(
        x_norm,
        y_norm,
        label="NORMAL",
        color="#1f77b4",
        linewidth=1.4,
        alpha=0.95,
    )
    ax.plot(
        x_decay,
        y_decay,
        label="DECAY",
        color="#2ca02c",
        linewidth=1.4,
        linestyle="--",
        alpha=0.95,
    )
    ax.plot(
        x_no,
        y_no,
        label="NO_INTERACT",
        color="#ff7f0e",
        linewidth=1.4,
        linestyle=":",
        alpha=0.95,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Evaluations", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax, grid_axis="both")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def _plot_synthetic_boxplot(
    title: str,
    ylabel: str,
    output_path: Path,
    box_specs: List[tuple[str, dict[str, float]]],
) -> None:
    if not box_specs:
        return

    stats = []
    labels = []
    mean_values: List[float | None] = []
    for label, spec in box_specs:
        stats.append({
            "label": label,
            "whislo": spec["low"],
            "q1": spec["q1"],
            "med": spec["med"],
            "q3": spec["q3"],
            "whishi": spec["high"],
            "fliers": [],
        })
        labels.append(label)
        mean_values.append(spec.get("mean"))

    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=180)
    palette = {
        "NORMAL": "#1f77b4",
        "DECAY": "#2ca02c",
        "NO_INTERACT": "#ff7f0e",
    }
    box_width = 0.6
    artists = ax.bxp(
        stats,
        showfliers=False,
        patch_artist=True,
        widths=box_width,
        boxprops={"linewidth": 1.0, "edgecolor": "#2f2f2f"},
        whiskerprops={"linewidth": 1.0, "color": "#2f2f2f"},
        capprops={"linewidth": 1.0, "color": "#2f2f2f"},
        medianprops={"linewidth": 1.2, "color": "#111111"},
    )
    for box, label in zip(artists["boxes"], labels):
        box.set_facecolor(palette.get(label, "#9e9e9e"))
        box.set_alpha(0.5)
    half_width = box_width / 2
    for idx, mean_val in enumerate(mean_values, start=1):
        if mean_val is None:
            continue
        ax.hlines(mean_val, idx - half_width, idx + half_width, colors="#d62728", linewidth=1.4)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#d62728", linewidth=1.4, label="mean"),
            Line2D([0], [0], color="#111111", linewidth=1.2, label="median"),
        ],
        frameon=False,
        fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    style_axes(ax, grid_axis="y")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def _plot_metric_pair(
    left_path: Path,
    right_path: Path,
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    metric_field: str,
    left_label: str,
    right_label: str,
    right_linestyle: str = "--",
) -> None:
    x_left, y_left = load_metric_series(left_path, x_field="step", y_field=metric_field)
    x_right, y_right = load_metric_series(right_path, x_field="step", y_field=metric_field)

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=180)
    ax.plot(
        x_left,
        y_left,
        label=left_label,
        color="#1f77b4",
        linewidth=1.4,
        alpha=0.95,
    )
    ax.plot(
        x_right,
        y_right,
        label=right_label,
        color="#ff7f0e",
        linewidth=1.4,
        linestyle=right_linestyle,
        alpha=0.95,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Evaluations", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax, grid_axis="both")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def _plot_mean_with_variance_band(
    interact_path: Path,
    no_interact_path: Path,
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    mean_field: str,
    std_field: str,
    std_scale: float = 1.0,
) -> None:
    x_int, y_int = load_metric_series(interact_path, x_field="step", y_field=mean_field)
    x_no, y_no = load_metric_series(no_interact_path, x_field="step", y_field=mean_field)
    x_int_std, y_int_std = load_metric_series(interact_path, x_field="step", y_field=std_field)
    x_no_std, y_no_std = load_metric_series(no_interact_path, x_field="step", y_field=std_field)

    if x_int != x_int_std or x_no != x_no_std:
        print(f"[WARN] Steps mismatch for variance band in {output_path}. Skipping.")
        return

    std_int = [val * std_scale for val in y_int_std]
    std_no = [val * std_scale for val in y_no_std]
    lower_int = [m - s for m, s in zip(y_int, std_int)]
    upper_int = [m + s for m, s in zip(y_int, std_int)]
    lower_no = [m - s for m, s in zip(y_no, std_no)]
    upper_no = [m + s for m, s in zip(y_no, std_no)]

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=180)
    ax.plot(x_int, y_int, label="interact", color="#1f77b4", linewidth=1.4, alpha=0.95)
    ax.fill_between(x_int, lower_int, upper_int, color="#1f77b4", alpha=0.25, linewidth=0)
    ax.plot(x_int, lower_int, color="#1f77b4", alpha=0.35, linewidth=0.6)
    ax.plot(x_int, upper_int, color="#1f77b4", alpha=0.35, linewidth=0.6)
    ax.plot(x_no, y_no, label="no_interact", color="#ff7f0e", linewidth=1.4, linestyle="--", alpha=0.95)
    ax.fill_between(x_no, lower_no, upper_no, color="#ff7f0e", alpha=0.25, linewidth=0)
    ax.plot(x_no, lower_no, color="#ff7f0e", alpha=0.35, linewidth=0.6, linestyle="--")
    ax.plot(x_no, upper_no, color="#ff7f0e", alpha=0.35, linewidth=0.6, linestyle="--")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Evaluations", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax, grid_axis="both")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def _iter_budget_dirs(experiment_dir: Path) -> List[Path]:
    budgets: List[Path] = []
    if not experiment_dir.exists():
        return budgets
    for path in experiment_dir.iterdir():
        if path.is_dir() and path.name.isdigit():
            budgets.append(path)
    return sorted(budgets, key=lambda p: int(p.name))


def _discover_budget_instances() -> List[dict[str, Path | str | int]]:
    exp_root = ROOT / "results" / "experiments"
    if not exp_root.exists():
        return []
    contexts: List[dict[str, Path | str | int]] = []
    seen: set[tuple[str, int, int]] = set()
    for entry in exp_root.iterdir():
        if not entry.is_dir():
            continue
        match = re.match(r"^(?P<name>.+)_dim(?P<dim>\d+)_t(?P<t>\d+)$", entry.name)
        if not match:
            continue
        if not _iter_budget_dirs(entry):
            continue
        name = match.group("name")
        dim = int(match.group("dim"))
        type_instance = int(match.group("t"))
        key = (name, dim, type_instance)
        if key in seen:
            continue
        seen.add(key)
        contexts.append(_build_instance_context_from_values(name, dim, type_instance))
    return sorted(contexts, key=lambda ctx: (str(ctx["problem_name"]), int(ctx["dim"]), int(ctx["type_instance"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot interact vs no_interact curves for each budget.")
    parser.add_argument("--problem", type=str, default=None, help="Problem config name (ex: qubo, nk, blockwise).")
    args = parser.parse_args()

    contexts = _discover_budget_instances()
    if not contexts:
        contexts = [_build_instance_context(args.problem)]

    for context in contexts:
        instance_name = context["instance_name"]
        print(f"[INFO] Plotting budgets for instance: {instance_name}")
        experiment_dir = Path(context["experiment_dir"])
        output_dir = Path(context["output_dir"])
        problem_name = str(context["problem_name"])
        dim = int(context["dim"])
        type_instance = int(context["type_instance"])

        budget_dirs = _iter_budget_dirs(experiment_dir)
        if not budget_dirs:
            print(f"[WARN] No budget directories found in {experiment_dir}.")
            continue

        for budget_dir in budget_dirs:
            budget = budget_dir.name
            interact_path = budget_dir / "interact.csv"
            no_interact_path = budget_dir / "no_interact.csv"
            if not interact_path.exists():
                print(f"Skipping budget {budget}: missing file: {interact_path}")
                continue

            budget_output_dir = output_dir / budget
            if no_interact_path.exists():
                _plot_metric_pair(
                    interact_path,
                    no_interact_path,
                    title=f"Average Score: interact vs no_interact ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                    ylabel="Average score",
                    output_path=budget_output_dir / "avg_score_interact_vs_no_interact.png",
                    metric_field="mean",
                    left_label="interact",
                    right_label="no_interact",
                )
                _plot_metric_pair(
                    interact_path,
                    no_interact_path,
                    title=f"Average Entropy: interact vs no_interact ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                    ylabel="Average entropy",
                    output_path=budget_output_dir / "entropy_interact_vs_no_interact.png",
                    metric_field="avg_entropy",
                    left_label="interact",
                    right_label="no_interact",
                )
                _plot_metric_pair(
                    interact_path,
                    no_interact_path,
                    title=f"Average Hamming: interact vs no_interact ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                    ylabel="Average hamming",
                    output_path=budget_output_dir / "hamming_interact_vs_no_interact.png",
                    metric_field="avg_hamming",
                    left_label="interact",
                    right_label="no_interact",
                )
                _plot_metric_pair(
                    interact_path,
                    no_interact_path,
                    title=f"Average L1: interact vs no_interact ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                    ylabel="Average L1",
                    output_path=budget_output_dir / "l1_interact_vs_no_interact.png",
                    metric_field="avg_l1",
                    left_label="interact",
                    right_label="no_interact",
                )
                try:
                    x_int, y_int, std_int = load_metric_series_with_std(interact_path, x_field="step")
                    x_no, y_no, std_no = load_metric_series_with_std(no_interact_path, x_field="step")
                except ValueError as exc:
                    print(f"[WARN] {exc}.")
                else:
                    if std_int is not None and std_no is not None:
                        _plot_metric_pair_values(
                            x_int,
                            y_int,
                            x_no,
                            y_no,
                            title=f"Average Score: interact vs no_interact ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                            ylabel="Average score",
                            output_path=budget_output_dir / "avg_score_interact_vs_no_interact_std.png",
                            left_label="interact",
                            right_label="no_interact",
                            std_left=std_int,
                            std_right=std_no,
                        )
                        _plot_metric_pair_values(
                            x_int,
                            std_int,
                            x_no,
                            std_no,
                            title=f"Std: interact vs no_interact ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                            ylabel="Std",
                            output_path=budget_output_dir / "std_interact_vs_no_interact.png",
                            left_label="interact",
                            right_label="no_interact",
                        )
                    else:
                        print(f"[WARN] Missing std column in {interact_path} or {no_interact_path}.")
            else:
                print(f"Skipping interact vs no_interact for budget {budget}: missing file {no_interact_path}")

            decay_path = budget_dir / "decay.csv"
            if decay_path.exists():
                _plot_metric_pair(
                    interact_path,
                    decay_path,
                    title=f"Average Score: NORMAL vs DECAY ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                    ylabel="Average score",
                    output_path=budget_output_dir / "avg_score_interact_vs_decay.png",
                    metric_field="mean",
                    left_label="NORMAL",
                    right_label="DECAY",
                )
                _plot_metric_pair(
                    interact_path,
                    decay_path,
                    title=f"Average Entropy: NORMAL vs DECAY ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                    ylabel="Average entropy",
                    output_path=budget_output_dir / "entropy_interact_vs_decay.png",
                    metric_field="avg_entropy",
                    left_label="NORMAL",
                    right_label="DECAY",
                )
                _plot_metric_pair(
                    interact_path,
                    decay_path,
                    title=f"Average Hamming: NORMAL vs DECAY ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                    ylabel="Average hamming",
                    output_path=budget_output_dir / "hamming_interact_vs_decay.png",
                    metric_field="avg_hamming",
                    left_label="NORMAL",
                    right_label="DECAY",
                )
                _plot_metric_pair(
                    interact_path,
                    decay_path,
                    title=f"Average L1: NORMAL vs DECAY ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                    ylabel="Average L1",
                    output_path=budget_output_dir / "l1_interact_vs_decay.png",
                    metric_field="avg_l1",
                    left_label="NORMAL",
                    right_label="DECAY",
                )
                try:
                    x_int, y_int, std_int = load_metric_series_with_std(interact_path, x_field="step")
                    x_dec, y_dec, std_dec = load_metric_series_with_std(decay_path, x_field="step")
                except ValueError as exc:
                    print(f"[WARN] {exc}.")
                else:
                    if std_int is not None and std_dec is not None:
                        _plot_metric_pair_values(
                            x_int,
                            y_int,
                            x_dec,
                            y_dec,
                            title=f"Average Score: NORMAL vs DECAY ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                            ylabel="Average score",
                            output_path=budget_output_dir / "avg_score_interact_vs_decay_std.png",
                            left_label="NORMAL",
                            right_label="DECAY",
                            std_left=std_int,
                            std_right=std_dec,
                        )
                        _plot_metric_pair_values(
                            x_int,
                            std_int,
                            x_dec,
                            std_dec,
                            title=f"Std: NORMAL vs DECAY ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                            ylabel="Std",
                            output_path=budget_output_dir / "std_interact_vs_decay.png",
                            left_label="NORMAL",
                            right_label="DECAY",
                        )
                    else:
                        print(f"[WARN] Missing std column in {interact_path} or {decay_path}.")

            if no_interact_path.exists() and decay_path.exists():
                _plot_metric_triplet(
                    interact_path,
                    decay_path,
                    no_interact_path,
                    title=f"Average Score: NORMAL vs DECAY vs NO_INTERACT ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                    ylabel="Average score",
                    output_path=budget_output_dir / "avg_score_normal_vs_decay_vs_no_interact.png",
                    metric_field="mean",
                )
                _plot_metric_triplet(
                    interact_path,
                    decay_path,
                    no_interact_path,
                    title=f"Average Entropy: NORMAL vs DECAY vs NO_INTERACT ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                    ylabel="Average entropy",
                    output_path=budget_output_dir / "entropy_normal_vs_decay_vs_no_interact.png",
                    metric_field="avg_entropy",
                )
                _plot_metric_triplet(
                    interact_path,
                    decay_path,
                    no_interact_path,
                    title=f"Average Hamming: NORMAL vs DECAY vs NO_INTERACT ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                    ylabel="Average hamming",
                    output_path=budget_output_dir / "hamming_normal_vs_decay_vs_no_interact.png",
                    metric_field="avg_hamming",
                )
                _plot_metric_triplet(
                    interact_path,
                    decay_path,
                    no_interact_path,
                    title=f"Average L1: NORMAL vs DECAY vs NO_INTERACT ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                    ylabel="Average L1",
                    output_path=budget_output_dir / "l1_normal_vs_decay_vs_no_interact.png",
                    metric_field="avg_l1",
                )

            box_specs: List[tuple[str, dict[str, float]]] = []
            normal_stats = load_quantile_stats_at_final_step(interact_path)
            if normal_stats:
                box_specs.append(("NORMAL", normal_stats))
            decay_stats = load_quantile_stats_at_final_step(decay_path) if decay_path.exists() else None
            if decay_stats:
                box_specs.append(("DECAY", decay_stats))
            no_stats = load_quantile_stats_at_final_step(no_interact_path) if no_interact_path.exists() else None
            if no_stats:
                box_specs.append(("NO_INTERACT", no_stats))

            _plot_synthetic_boxplot(
                title=f"Final score distribution ({problem_name} N={dim}, K={type_instance}, budget={budget})",
                ylabel="Score",
                output_path=budget_output_dir / "boxplot_final_score.png",
                box_specs=box_specs,
            )


if __name__ == "__main__":
    main()
