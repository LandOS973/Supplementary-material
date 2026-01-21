#!/usr/bin/env python3
"""Plot interact vs no_interact curves for each budget from config."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
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
        dim = int(problem_cfg.get("dim") or problem_cfg.get("n") or dim)
        type_instance = int(problem_cfg.get("type_instance") or problem_cfg.get("k") or type_instance)
    except OSError:
        pass
    return problem_name, dim, type_instance


def _build_instance_context(problem_override: str | None = None) -> dict[str, Path | str | int]:
    problem_name, dim, type_instance = _load_problem_config(problem_override)
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


def style_axes(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9, width=0.6, length=3)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.5, alpha=0.4)


def _plot_interact_vs_no_interact(
    interact_path: Path,
    no_interact_path: Path,
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    metric_field: str,
) -> None:
    x_int, y_int = load_metric_series(interact_path, x_field="step", y_field=metric_field)
    x_no, y_no = load_metric_series(no_interact_path, x_field="step", y_field=metric_field)

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=180)
    ax.plot(
        x_int,
        y_int,
        label="interact",
        color="#1f77b4",
        linewidth=1.4,
        alpha=0.95,
    )
    ax.plot(
        x_no,
        y_no,
        label="no_interact",
        color="#ff7f0e",
        linewidth=1.4,
        linestyle="--",
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


def _iter_budget_dirs(experiment_dir: Path) -> List[Path]:
    budgets: List[Path] = []
    if not experiment_dir.exists():
        return budgets
    for path in experiment_dir.iterdir():
        if path.is_dir() and path.name.isdigit():
            budgets.append(path)
    return sorted(budgets, key=lambda p: int(p.name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot interact vs no_interact curves for each budget.")
    parser.add_argument("--problem", type=str, default=None, help="Problem config name (ex: qubo, nk, blockwise).")
    args = parser.parse_args()

    context = _build_instance_context(args.problem)
    experiment_dir = Path(context["experiment_dir"])
    output_dir = Path(context["output_dir"])
    problem_name = str(context["problem_name"])
    dim = int(context["dim"])
    type_instance = int(context["type_instance"])

    budget_dirs = _iter_budget_dirs(experiment_dir)
    if not budget_dirs:
        raise FileNotFoundError(f"No budget directories found in {experiment_dir}")

    for budget_dir in budget_dirs:
        budget = budget_dir.name
        interact_path = budget_dir / "interact.csv"
        no_interact_path = budget_dir / "no_interact.csv"
        if not interact_path.exists() or not no_interact_path.exists():
            missing = []
            if not interact_path.exists():
                missing.append(str(interact_path))
            if not no_interact_path.exists():
                missing.append(str(no_interact_path))
            print(f"Skipping budget {budget}: missing file(s): {', '.join(missing)}")
            continue

        budget_output_dir = output_dir / budget
        _plot_interact_vs_no_interact(
            interact_path,
            no_interact_path,
            title=f"Average Score: interact vs no_interact ({problem_name} N={dim}, K={type_instance}, budget={budget})",
            ylabel="Average score",
            output_path=budget_output_dir / "avg_score_interact_vs_no_interact.png",
            metric_field="mean",
        )
        _plot_interact_vs_no_interact(
            interact_path,
            no_interact_path,
            title=f"Average Entropy: interact vs no_interact ({problem_name} N={dim}, K={type_instance}, budget={budget})",
            ylabel="Average entropy",
            output_path=budget_output_dir / "entropy_interact_vs_no_interact.png",
            metric_field="avg_entropy",
        )


if __name__ == "__main__":
    main()
