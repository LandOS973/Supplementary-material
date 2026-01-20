#!/usr/bin/env python3
"""Plot comparison curves for QUBO (N=256, K=4)."""

from __future__ import annotations

import csv
from itertools import cycle
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RANKING_PATH = ROOT / "additional_results/global_ranking/UBQP_N_256_K_4_ranks.csv"
DEFAULT_KERNEL = "rbf"
MY_DATA_PATH = ROOT / "results/experiments/QUBO_dim256_t4/QUBO_rbf_best_metrics.csv"
EXPERIMENT_DIR = ROOT / "results/experiments/QUBO_dim256_t4"
COMPETITOR_DIR = Path("/home/landos/Downloads/resultAlgos/results_nevergrad_final")
INSTANCE_NAME = "UBQP_256_4"
OUTPUT_PATH = ROOT / "courbes" / INSTANCE_NAME / "comparison_qubo_256_t4.png"
KERNELS_OUTPUT_DIR = ROOT / "courbes" / INSTANCE_NAME / "Kernels"


def load_top_algorithms(
    ranking_path: Path,
    limit: int = 10,
    skip: Iterable[str] = ("PPO-EDA",),
) -> List[str]:
    algos: List[str] = []
    skip_set = set(skip)
    with ranking_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("name_algo") or row.get("algo") or row.get("name")
            if not name:
                continue
            if name in skip_set:
                continue
            algos.append(name)
            if len(algos) >= limit:
                break
    if len(algos) < limit:
        raise ValueError(
            f"Expected at least {limit} algorithms, found {len(algos)} in {ranking_path}"
        )
    return algos


def resolve_xy_keys(
    fieldnames: List[str], x_key: str | None, y_key: str | None
) -> Tuple[str, str]:
    if x_key and x_key in fieldnames:
        x_field = x_key
    else:
        x_field = next(
            (key for key in ("runtime", "evaluations", "evaluation", "eval", "step", "budget") if key in fieldnames),
            fieldnames[0],
        )

    if y_key and y_key in fieldnames:
        y_field = y_key
    else:
        y_field = next(
            (key for key in ("mean", "score", "fitness", "best_fitness", "value") if key in fieldnames),
            fieldnames[1] if len(fieldnames) > 1 else fieldnames[0],
        )
    return x_field, y_field


def load_xy_from_csv(
    path: Path, has_header: bool, x_key: str | None = None, y_key: str | None = None
) -> Tuple[List[float], List[float]]:
    x_vals: List[float] = []
    y_vals: List[float] = []
    with path.open(newline="") as handle:
        if has_header:
            reader = csv.DictReader(handle, skipinitialspace=True)
            if not reader.fieldnames:
                raise ValueError(f"Missing header in {path}")
            x_field, y_field = resolve_xy_keys(reader.fieldnames, x_key, y_key)
            for row in reader:
                try:
                    x_vals.append(float(row.get(x_field, "")))
                    y_vals.append(float(row.get(y_field, "")))
                except (ValueError, TypeError):
                    continue
        else:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    x_vals.append(float(row[0]))
                    y_vals.append(float(row[1]))
                except ValueError:
                    continue
    if not x_vals:
        raise ValueError(f"No valid data in {path}")
    return sort_by_x(x_vals, y_vals)


def sort_by_x(x_vals: List[float], y_vals: List[float]) -> Tuple[List[float], List[float]]:
    pairs = sorted(zip(x_vals, y_vals), key=lambda pair: pair[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def find_competitor_files(algo: str) -> Tuple[List[Path], bool]:
    csv_path = COMPETITOR_DIR / f"{algo}.csv"
    if csv_path.exists():
        return [csv_path], False

    algo_dir = COMPETITOR_DIR / algo
    if not algo_dir.exists():
        raise FileNotFoundError(f"Missing competitor directory: {algo_dir}")

    candidate_files: List[Path] = []
    for problem in ("UBQP", "QUBO", "qubo", "ubqp"):
        candidate_dir = algo_dir / problem / "256" / "4"
        if candidate_dir.exists():
            candidate_files.extend(candidate_dir.glob("*.txt"))

    if not candidate_files:
        patterns = (
            f"**/results_nevergrad_{algo}_UBQP_256_4_*.txt",
            f"**/results_nevergrad_{algo}_QUBO_256_4_*.txt",
            f"**/*{algo}*UBQP*256*4*.txt",
            f"**/*{algo}*QUBO*256*4*.txt",
        )
        for pattern in patterns:
            candidate_files.extend(algo_dir.glob(pattern))

    if not candidate_files:
        raise FileNotFoundError(
            f"Missing competitor data file for {algo} under {algo_dir}"
        )

    return sorted(candidate_files), True


def load_mean_across_runs(paths: List[Path]) -> Tuple[List[float], List[float]]:
    sums: dict[float, float] = {}
    counts: dict[float, int] = {}
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            if not reader.fieldnames:
                continue
            x_field, y_field = resolve_xy_keys(reader.fieldnames, "runtime", "mean")
            for row in reader:
                try:
                    x_val = float(row.get(x_field, ""))
                    y_val = float(row.get(y_field, ""))
                except (ValueError, TypeError):
                    continue
                sums[x_val] = sums.get(x_val, 0.0) + y_val
                counts[x_val] = counts.get(x_val, 0) + 1

    if not sums:
        raise ValueError("No valid data in competitor run files")

    x_vals = sorted(sums.keys())
    y_vals = [sums[x_val] / counts[x_val] for x_val in x_vals]
    return x_vals, y_vals


def plot_comparison() -> None:
    algos = load_top_algorithms(RANKING_PATH, limit=10, skip=("PPO-EDA",))
    best_kernel = find_best_kernel_summary(EXPERIMENT_DIR)
    my_data_path = EXPERIMENT_DIR / f"QUBO_{best_kernel}_best_metrics.csv"
    if not my_data_path.exists():
        my_data_path = MY_DATA_PATH

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    color_cycle = cycle(plt.cm.tab20.colors)

    for algo in algos:
        paths, has_header = find_competitor_files(algo)
        if has_header:
            if len(paths) > 1:
                x_vals, y_vals = load_mean_across_runs(paths)
            else:
                x_vals, y_vals = load_xy_from_csv(
                    paths[0], has_header=True, x_key="runtime", y_key="mean"
                )
        else:
            x_vals, y_vals = load_xy_from_csv(paths[0], has_header=False)
        ax.plot(
            x_vals,
            y_vals,
            label=algo,
            color=next(color_cycle),
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=2,
            antialiased=True,
        )

    my_x, my_y = load_xy_from_csv(
        my_data_path, has_header=True, x_key="step", y_key="best_fitness"
    )
    my_filtered = [(x, y) for x, y in zip(my_x, my_y) if x >= 100]
    if not my_filtered:
        raise ValueError("No reinforce svgd data at step >= 100")
    my_x, my_y = zip(*my_filtered)
    ax.plot(
        my_x,
        my_y,
        label=f"reinforce svgd ({best_kernel})",
        color="green",
        linewidth=2.0,
        zorder=3,
        antialiased=True,
    )

    ax.set_title("Comparison on QUBO (N=256, K=4)")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Average score")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"Saved plot to {OUTPUT_PATH}")


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


def find_best_kernel_summary(summary_dir: Path) -> str:
    summary_files = sorted(summary_dir.glob("QUBO_*_best_summary.txt"))
    if not summary_files:
        return DEFAULT_KERNEL

    best_kernel = DEFAULT_KERNEL
    best_score = None
    for path in summary_files:
        data = parse_summary_file(path)
        kernel = data.get("Kernel")
        avg_score = parse_float(data.get("avg_score"))
        if kernel is None or avg_score is None:
            continue
        if best_score is None or avg_score < best_score:
            best_score = avg_score
            best_kernel = kernel
    return best_kernel
    try:
        return float(value)
    except ValueError:
        return None


def compute_metric_means(path: Path, fields: Iterable[str]) -> dict[str, float]:
    totals = {field: 0.0 for field in fields}
    counts = {field: 0 for field in fields}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for field in fields:
                value = parse_float(row.get(field))
                if value is None:
                    continue
                totals[field] += value
                counts[field] += 1

    means: dict[str, float] = {}
    for field in fields:
        if counts[field] == 0:
            raise ValueError(f"Missing metric {field} in {path}")
        means[field] = totals[field] / counts[field]
    return means


def load_metric_series(path: Path, x_field: str, y_field: str) -> Tuple[List[float], List[float]]:
    x_vals: List[float] = []
    y_vals: List[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_val = parse_float(row.get(x_field))
            y_val = parse_float(row.get(y_field))
            if x_val is None or y_val is None:
                continue
            x_vals.append(x_val)
            y_vals.append(y_val)
    if not x_vals:
        raise ValueError(f"No data for {y_field} in {path}")
    return sort_by_x(x_vals, y_vals)


def style_axes(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9, width=0.6, length=3)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.5, alpha=0.4)


def plot_kernel_bar(
    kernels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    output_path: Path,
    log_scale: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=180)
    colors = plt.cm.Set2.colors[: len(kernels)]
    ax.bar(
        kernels,
        values,
        color=colors,
        edgecolor="#333333",
        linewidth=0.4,
        width=0.6,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Kernel", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if log_scale:
        ax.set_yscale("log")
    style_axes(ax, grid_axis="y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_kernel_hyperparams(kernels: List[str], params: dict[str, List[float]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), dpi=180)
    axes = axes.flatten()
    for idx, (param, values) in enumerate(params.items()):
        ax = axes[idx]
        ax.bar(
            kernels,
            values,
            color=plt.cm.Set3.colors[: len(kernels)],
            edgecolor="#333333",
            linewidth=0.4,
            width=0.6,
        )
        ax.set_title(param, fontsize=11)
        ax.set_xlabel("Kernel", fontsize=9)
        if param == "gamma":
            ax.set_yscale("log")
        style_axes(ax, grid_axis="y")

    fig.suptitle("Grid Search Hyperparameters", fontsize=13, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = KERNELS_OUTPUT_DIR / "qubo_dim256_t4_hyperparams.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_kernel_metric_curves(
    kernels: List[str],
    metric_field: str,
    title: str,
    ylabel: str,
    filename: str,
    show_markers: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.4), dpi=180)
    color_cycle = cycle(plt.cm.tab10.colors)
    for kernel in kernels:
        metrics_path = EXPERIMENT_DIR / f"QUBO_{kernel}_best_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file {metrics_path}")
        x_vals, y_vals = load_metric_series(metrics_path, x_field="step", y_field=metric_field)
        mark_every = max(1, len(x_vals) // 25)
        ax.plot(
            x_vals,
            y_vals,
            label=kernel,
            color=next(color_cycle),
            linewidth=1.2 if not show_markers else 1.2,
            alpha=0.95,
            marker="o" if show_markers else None,
            markersize=2.4,
            markerfacecolor="white",
            markeredgewidth=0.6,
            markevery=mark_every,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Evaluations", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    style_axes(ax, grid_axis="both")
    output_path = KERNELS_OUTPUT_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_kernel_comparison() -> None:
    summary_files = sorted(EXPERIMENT_DIR.glob("QUBO_*_best_summary.txt"))
    if not summary_files:
        raise FileNotFoundError(f"No summary files found in {EXPERIMENT_DIR}")

    kernel_data: dict[str, dict[str, str]] = {}
    for path in summary_files:
        data = parse_summary_file(path)
        kernel = data.get("Kernel")
        if not kernel:
            continue
        kernel_data[kernel] = data

    if not kernel_data:
        raise ValueError(f"No kernel entries found in {EXPERIMENT_DIR}")

    kernels = sorted(kernel_data.keys())
    hyperparams = {"M": [], "lambda": [], "epsilon_svgd": [], "gamma": []}

    for kernel in kernels:
        summary = kernel_data[kernel]
        for param in hyperparams:
            value = parse_float(summary.get(param))
            if value is None:
                raise ValueError(f"Missing {param} for kernel {kernel}")
            hyperparams[param].append(value)

    plot_kernel_hyperparams(kernels, hyperparams)
    plot_kernel_metric_curves(
        kernels,
        metric_field="mean",
        title="Average Score per Kernel (Evolution)",
        ylabel="Average score",
        filename="qubo_dim256_t4_curve_avg_score.png",
        show_markers=False,
    )
    plot_kernel_metric_curves(
        kernels,
        metric_field="avg_l1",
        title="Average L1 per Kernel (Evolution)",
        ylabel="Average L1",
        filename="qubo_dim256_t4_curve_avg_l1.png",
    )
    plot_kernel_metric_curves(
        kernels,
        metric_field="avg_hamming",
        title="Average Hamming per Kernel (Evolution)",
        ylabel="Average Hamming",
        filename="qubo_dim256_t4_curve_avg_hamming.png",
    )
    plot_kernel_metric_curves(
        kernels,
        metric_field="avg_entropy",
        title="Average Entropy per Kernel (Evolution)",
        ylabel="Average Entropy",
        filename="qubo_dim256_t4_curve_avg_entropy.png",
    )


def main() -> None:
    plot_comparison()
    plot_kernel_comparison()


if __name__ == "__main__":
    main()
