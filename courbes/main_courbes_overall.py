#!/usr/bin/env python3
"""
Plot curves/boxplots for a given config under results/config/<ConfigName>.
Outputs plots inside each instance directory.
"""

from __future__ import annotations

import csv
import os
import re
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt

from main_plot import (
    _normalize_score_sign,
    clip_series_to_budget,
    aggregate_quantile_stats,
    find_competitor_files,
    load_mean_across_runs,
    load_metric_series,
    load_metric_series_with_std,
    load_quantile_stats_at_final_step,
    load_top_algorithms,
    load_xy_from_csv,
    parse_float,
    normalize_box_stats,
    plot_interact_vs_no_interact_series,
    plot_pair_series,
    plot_synthetic_boxplot,
)


ROOT = Path(__file__).resolve().parent.parent
INSTANCE_DIR_RE = re.compile(r"^(?P<problem>QUBO|NK|NK3)_dim(?P<dim>\d+)_t(?P<t>\d+)$")
MAX_BUDGET = 50000


def _build_instance_list(config_dir: Path):
    instances = []
    for child in sorted(config_dir.iterdir()):
        if not child.is_dir():
            continue
        match = INSTANCE_DIR_RE.match(child.name)
        if not match:
            continue
        instances.append(
            dict(
                path=child,
                problem=match.group("problem"),
                dim=int(match.group("dim")),
                type_instance=int(match.group("t")),
            )
        )
    return instances


def _read_score_at_budget(path: Path, budget: int) -> float | None:
    """Return score at exact budget if present, else the last available score."""
    try:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            runtime_idx = 0
            score_idx = 1
            first_data_row: list[str] | None = None
            if header:
                lower = [cell.strip().lower() for cell in header]
                if "runtime" in lower or "score" in lower:
                    if "runtime" in lower:
                        runtime_idx = lower.index("runtime")
                    if "score" in lower:
                        score_idx = lower.index("score")
                else:
                    # Headerless file: treat first row as data if numeric.
                    if parse_float(header[0].strip()) is not None and parse_float(header[1].strip()) is not None:
                        first_data_row = header

            last_score: float | None = None
            score_at_budget: float | None = None
            if first_data_row is not None:
                runtime = parse_float(first_data_row[runtime_idx].strip())
                score = parse_float(first_data_row[score_idx].strip())
                if runtime is not None and score is not None:
                    last_score = score
                    if abs(runtime - budget) < 1e-9:
                        score_at_budget = score
            for row in reader:
                if len(row) <= max(runtime_idx, score_idx):
                    continue
                runtime = parse_float(row[runtime_idx].strip())
                score = parse_float(row[score_idx].strip())
                if runtime is None or score is None:
                    continue
                last_score = score
                if abs(runtime - budget) < 1e-9:
                    score_at_budget = score
            return score_at_budget if score_at_budget is not None else last_score
    except OSError:
        return None


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("Empty list")
    if p <= 0:
        return sorted_values[0]
    if p >= 1:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _aggregate_final_scores_from_runs(paths: list[Path], budget: int) -> dict[str, float] | None:
    scores: list[float] = []
    for path in paths:
        score = _read_score_at_budget(path, budget)
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    scores.sort()
    return {
        "low": _percentile(scores, 0.02),
        "q1": _percentile(scores, 0.25),
        "med": _percentile(scores, 0.50),
        "q3": _percentile(scores, 0.75),
        "high": _percentile(scores, 0.98),
        "mean": sum(scores) / len(scores),
    }


def _plot_comparison_vs_algos(instance, config_name: str, my_metrics: Path, output_dir: Path):
    problem = instance["problem"]
    dim = instance["dim"]
    t = instance["type_instance"]
    if problem.upper() == "QUBO":
        ranking_path = ROOT / "additional_results" / "global_ranking" / f"UBQP_N_{dim}_K_{t}_ranks.csv"
    else:
        ranking_path = ROOT / "additional_results" / "global_ranking" / f"{problem}_N_{dim}_K_{t}_ranks.csv"
    if not ranking_path.exists():
        print(f"[WARN] Missing ranking file {ranking_path}.")
        return

    algos = load_top_algorithms(ranking_path, limit=10, skip=("PPO-EDA", "Tabu", "TABU", "tabu"))
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    color_cycle = cycle(plt.cm.tab20.colors)

    for algo in algos:
        try:
            paths, has_header = find_competitor_files(algo, dim, t, problem)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}. Skipping.")
            continue
        if has_header:
            if len(paths) > 1:
                x_vals, y_vals = load_mean_across_runs(paths)
            else:
                x_vals, y_vals = load_xy_from_csv(paths[0], has_header=True, x_key="runtime", y_key="mean")
        else:
            x_vals, y_vals = load_xy_from_csv(paths[0], has_header=False)
        x_vals, y_vals = clip_series_to_budget(x_vals, y_vals, max_budget=MAX_BUDGET)
        y_vals = _normalize_score_sign(problem, y_vals)
        ax.plot(
            x_vals,
            y_vals,
            label=algo,
            color=next(color_cycle),
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=2,
        )

    my_x, my_y = load_xy_from_csv(my_metrics, has_header=True, x_key="step", y_key="best_fitness")
    my_filtered = [(x, y) for x, y in zip(my_x, my_y) if x >= 100]
    if my_filtered:
        my_x, my_y = zip(*my_filtered)
        my_x, my_y = clip_series_to_budget(list(my_x), list(my_y), max_budget=MAX_BUDGET)
        my_y = _normalize_score_sign(problem, list(my_y))
        ax.plot(
            my_x,
            my_y,
            label="SVGD-EDA",
            color="green",
            linewidth=2.0,
            zorder=3,
        )

    ax.set_title("")
    ax.set_xlabel("Evaluations", fontsize=16)
    ax.set_ylabel("Average score", fontsize=16)
    ax.legend(fontsize=11, ncol=2, frameon=False)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    output_path = output_dir / f"comparison_{problem.lower()}_{dim}_t{t}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def _plot_top5_boxplot(instance, my_metrics: Path, output_dir: Path):
    problem = instance["problem"]
    dim = instance["dim"]
    t = instance["type_instance"]
    if problem.upper() == "QUBO":
        ranking_path = ROOT / "additional_results" / "global_ranking" / f"UBQP_N_{dim}_K_{t}_ranks.csv"
    else:
        ranking_path = ROOT / "additional_results" / "global_ranking" / f"{problem}_N_{dim}_K_{t}_ranks.csv"
    if not ranking_path.exists():
        print(f"[WARN] Missing ranking file {ranking_path}.")
        return

    my_stats = load_quantile_stats_at_final_step(my_metrics, x_field="step")
    if not my_stats:
        print(f"[WARN] Missing quantiles in {my_metrics}. Skipping top-5 boxplot.")
        return

    my_label = "SVGD-EDA"
    box_specs = [(my_label, normalize_box_stats(problem, my_stats))]

    algos = load_top_algorithms(ranking_path, limit=5, skip=("PPO-EDA", "Tabu", "TABU", "tabu"))
    for algo in algos:
        try:
            paths, has_header = find_competitor_files(algo, dim, t, problem)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}. Skipping.")
            continue
        if not has_header:
            run_stats = _aggregate_final_scores_from_runs(paths, MAX_BUDGET)
            if not run_stats:
                print(f"[WARN] Missing quantiles for {algo}. Skipping.")
                continue
            stats = normalize_box_stats(problem, run_stats)
        else:
            stats = aggregate_quantile_stats(paths, problem_name=problem)
            if not stats:
                run_stats = _aggregate_final_scores_from_runs(paths, MAX_BUDGET)
                if not run_stats:
                    print(f"[WARN] Missing quantiles for {algo}. Skipping.")
                    continue
                stats = normalize_box_stats(problem, run_stats)
        if not stats:
            print(f"[WARN] Missing quantiles for {algo}. Skipping.")
            continue
        box_specs.append((algo, stats))

    if len(box_specs) <= 1:
        print("[WARN] No competitor stats available for top-5 boxplot.")
        return

    palette = {my_label: "#2ca02c"}
    color_cycle = cycle(plt.cm.tab20.colors)
    for label, _ in box_specs:
        if label == my_label:
            continue
        palette[label] = next(color_cycle)

    output_path = output_dir / "boxplot_final_score_vs_top5.png"
    plot_synthetic_boxplot(
        title="",
        ylabel="Score",
        output_path=output_path,
        box_specs=box_specs,
        palette=palette,
    )


def _plot_interact_vs_no_interact(instance, interact_path: Path, no_interact_path: Path, output_dir: Path):
    problem = instance["problem"]
    dim = instance["dim"]
    t = instance["type_instance"]
    maximize = problem.upper() in ("NK", "BLOCK")

    x_int, y_int, std_int = load_metric_series_with_std(
        interact_path, x_field="step", maximize=maximize
    )
    x_no, y_no, std_no = load_metric_series_with_std(
        no_interact_path, x_field="step", maximize=maximize
    )
    if problem.upper() in ("QUBO", "UBQP"):
        y_int = _normalize_score_sign(problem, y_int)
        y_no = _normalize_score_sign(problem, y_no)

    plot_interact_vs_no_interact_series(
        x_int,
        y_int,
        x_no,
        y_no,
        title=f"Average Score: interact vs no_interact ({problem} N={dim}, K={t}, budget=50000)",
        ylabel="Average score",
        output_path=output_dir / "avg_score_interact_vs_no_interact.png",
        std_int=std_int,
        std_no=std_no,
    )

    if std_int is not None and std_no is not None:
        plot_interact_vs_no_interact_series(
            x_int,
            std_int,
            x_no,
            std_no,
            title=f"Std: interact vs no_interact ({problem} N={dim}, K={t}, budget=50000)",
            ylabel="Std",
            output_path=output_dir / "std_interact_vs_no_interact.png",
        )

    # L1 + Hamming
    x_int_l1, y_int_l1 = load_metric_series(interact_path, x_field="step", y_field="avg_l1")
    x_no_l1, y_no_l1 = load_metric_series(no_interact_path, x_field="step", y_field="avg_l1")
    plot_pair_series(
        x_int_l1,
        y_int_l1,
        x_no_l1,
        y_no_l1,
        title=f"L1: interact vs no_interact ({problem} N={dim}, K={t})",
        ylabel="Average L1",
        output_path=output_dir / "l1_interact_vs_no_interact.png",
        label_a="interact",
        label_b="no_interact",
        color_a="#1f77b4",
        color_b="#ff7f0e",
    )

    x_int_h, y_int_h = load_metric_series(interact_path, x_field="step", y_field="avg_hamming")
    x_no_h, y_no_h = load_metric_series(no_interact_path, x_field="step", y_field="avg_hamming")
    plot_pair_series(
        x_int_h,
        y_int_h,
        x_no_h,
        y_no_h,
        title=f"Hamming: interact vs no_interact ({problem} N={dim}, K={t})",
        ylabel="Average hamming",
        output_path=output_dir / "hamming_interact_vs_no_interact.png",
        label_a="interact",
        label_b="no_interact",
        color_a="#1f77b4",
        color_b="#ff7f0e",
    )

    x_int_e, y_int_e = load_metric_series(interact_path, x_field="step", y_field="avg_entropy")
    x_no_e, y_no_e = load_metric_series(no_interact_path, x_field="step", y_field="avg_entropy")
    plot_pair_series(
        x_int_e,
        y_int_e,
        x_no_e,
        y_no_e,
        title=f"Entropy: interact vs no_interact ({problem} N={dim}, K={t})",
        ylabel="Average entropy",
        output_path=output_dir / "entropy_interact_vs_no_interact.png",
        label_a="interact",
        label_b="no_interact",
        color_a="#1f77b4",
        color_b="#ff7f0e",
    )

    # Boxplot: final score normal vs no_interact
    box_specs = []
    stats_int = load_quantile_stats_at_final_step(interact_path)
    if stats_int:
        box_specs.append(("NORMAL", normalize_box_stats(problem, stats_int)))
    stats_no = load_quantile_stats_at_final_step(no_interact_path)
    if stats_no:
        box_specs.append(("NO_INTERACT", normalize_box_stats(problem, stats_no)))
    if box_specs:
        plot_synthetic_boxplot(
            title=f"Final score distribution (normal vs no_interact) {problem} N={dim}, K={t}",
            ylabel="Score",
            output_path=output_dir / "boxplot_final_score_normal_vs_no_interact.png",
            box_specs=box_specs,
        )


def _plot_normal_vs_no_repulsion(instance, normal_path: Path, no_repulsion_path: Path, output_dir: Path):
    problem = instance["problem"]
    dim = instance["dim"]
    t = instance["type_instance"]
    maximize = problem.upper() in ("NK", "BLOCK")

    x_norm, y_norm, std_norm = load_metric_series_with_std(
        normal_path, x_field="step", maximize=maximize
    )
    x_nr, y_nr, std_nr = load_metric_series_with_std(
        no_repulsion_path, x_field="step", maximize=maximize
    )
    if problem.upper() in ("QUBO", "UBQP"):
        y_norm = _normalize_score_sign(problem, y_norm)
        y_nr = _normalize_score_sign(problem, y_nr)

    plot_pair_series(
        x_norm,
        y_norm,
        x_nr,
        y_nr,
        title=f"Average Score: normal vs no_repulsion ({problem} N={dim}, K={t})",
        ylabel="Average score",
        output_path=output_dir / "avg_score_normal_vs_no_repulsion.png",
        label_a="normal",
        label_b="no_repulsion",
        color_a="#1f77b4",
        color_b="#ff7f0e",
    )

    if std_norm is not None and std_nr is not None:
        plot_pair_series(
            x_norm,
            std_norm,
            x_nr,
            std_nr,
            title=f"Std: normal vs no_repulsion ({problem} N={dim}, K={t})",
            ylabel="Std",
            output_path=output_dir / "std_normal_vs_no_repulsion.png",
            label_a="normal",
            label_b="no_repulsion",
            color_a="#1f77b4",
            color_b="#ff7f0e",
        )

    x_norm_l1, y_norm_l1 = load_metric_series(normal_path, x_field="step", y_field="avg_l1")
    x_nr_l1, y_nr_l1 = load_metric_series(no_repulsion_path, x_field="step", y_field="avg_l1")
    plot_pair_series(
        x_norm_l1,
        y_norm_l1,
        x_nr_l1,
        y_nr_l1,
        title=f"L1: normal vs no_repulsion ({problem} N={dim}, K={t})",
        ylabel="Average L1",
        output_path=output_dir / "l1_normal_vs_no_repulsion.png",
        label_a="normal",
        label_b="no_repulsion",
        color_a="#1f77b4",
        color_b="#ff7f0e",
    )

    x_norm_h, y_norm_h = load_metric_series(normal_path, x_field="step", y_field="avg_hamming")
    x_nr_h, y_nr_h = load_metric_series(no_repulsion_path, x_field="step", y_field="avg_hamming")
    plot_pair_series(
        x_norm_h,
        y_norm_h,
        x_nr_h,
        y_nr_h,
        title=f"Hamming: normal vs no_repulsion ({problem} N={dim}, K={t})",
        ylabel="Average hamming",
        output_path=output_dir / "hamming_normal_vs_no_repulsion.png",
        label_a="normal",
        label_b="no_repulsion",
        color_a="#1f77b4",
        color_b="#ff7f0e",
    )

    x_norm_e, y_norm_e = load_metric_series(normal_path, x_field="step", y_field="avg_entropy")
    x_nr_e, y_nr_e = load_metric_series(no_repulsion_path, x_field="step", y_field="avg_entropy")
    plot_pair_series(
        x_norm_e,
        y_norm_e,
        x_nr_e,
        y_nr_e,
        title=f"Entropy: normal vs no_repulsion ({problem} N={dim}, K={t})",
        ylabel="Average entropy",
        output_path=output_dir / "entropy_normal_vs_no_repulsion.png",
        label_a="normal",
        label_b="no_repulsion",
        color_a="#1f77b4",
        color_b="#ff7f0e",
    )

    box_specs = []
    stats_norm = load_quantile_stats_at_final_step(normal_path)
    if stats_norm:
        box_specs.append(("NORMAL", normalize_box_stats(problem, stats_norm)))
    stats_nr = load_quantile_stats_at_final_step(no_repulsion_path)
    if stats_nr:
        box_specs.append(("NO_REPULSION", normalize_box_stats(problem, stats_nr)))
    if box_specs:
        plot_synthetic_boxplot(
            title=f"Final score distribution (normal vs no_repulsion) {problem} N={dim}, K={t}",
            ylabel="Score",
            output_path=output_dir / "boxplot_final_score_normal_vs_no_repulsion.png",
            box_specs=box_specs,
        )


def main():
    config_name = input(
        "Config name (ex: kjsd__advperagentrankweighted__M4__L24__eps0p01__g0p0005__ds0p05__dm0p05): "
    ).strip()
    if not config_name:
        raise SystemExit("Config name required.")

    config_dir = ROOT / "results" / "config" / config_name
    if not config_dir.exists():
        raise SystemExit(f"Config directory not found: {config_dir}")

    instances = _build_instance_list(config_dir)
    if not instances:
        raise SystemExit("No instances found under this config.")

    for inst in instances:
        inst_dir = inst["path"]
        my_metrics = inst_dir / "best_metrics.csv"
        if not my_metrics.exists():
            print(f"[WARN] Missing {my_metrics}. Skipping.")
            continue

        output_dir = inst_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Plotting {inst_dir.name} -> {output_dir}")
        try:
            _plot_comparison_vs_algos(inst, config_name, my_metrics, output_dir)
        except Exception as exc:
            print(f"[WARN] comparison failed for {inst_dir.name}: {exc}")
        try:
            _plot_top5_boxplot(inst, my_metrics, output_dir)
        except Exception as exc:
            print(f"[WARN] top5 boxplot failed for {inst_dir.name}: {exc}")

        no_interact_metrics = inst_dir / "no_interact" / "best_metrics.csv"
        if no_interact_metrics.exists():
            try:
                _plot_interact_vs_no_interact(inst, my_metrics, no_interact_metrics, output_dir)
            except Exception as exc:
                print(f"[WARN] interact/no_interact failed for {inst_dir.name}: {exc}")
        else:
            print(f"[WARN] Missing {no_interact_metrics}. Skipping interact/no_interact plots.")

        no_repulsion_metrics = inst_dir / "no_repulsion" / "best_metrics.csv"
        if no_repulsion_metrics.exists():
            try:
                _plot_normal_vs_no_repulsion(inst, my_metrics, no_repulsion_metrics, output_dir)
            except Exception as exc:
                print(f"[WARN] normal/no_repulsion failed for {inst_dir.name}: {exc}")
        else:
            print(f"[WARN] Missing {no_repulsion_metrics}. Skipping normal/no_repulsion plots.")


if __name__ == "__main__":
    main()
