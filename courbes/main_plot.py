#!/usr/bin/env python3
"""Plot comparison curves based on the active config instance."""

from __future__ import annotations

import csv
from itertools import cycle
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KERNEL = "rbf"
COMPETITOR_DIRS = [
    Path("/home/landos/Downloads/resultAlgos/results_nevergrad_final"),
    Path("/home/landos/Downloads/resultAlgos/results_EDAs_final"),
]


def _is_maximization_problem(problem_name: str) -> bool:
    return problem_name.upper() in ("NK", "BLOCK")


def _normalize_score_sign(problem_name: str, values: List[float]) -> List[float]:
    if not values:
        return values
    maximize = _is_maximization_problem(problem_name)
    if maximize:
        if max(values) <= 0:
            return [-val for val in values]
        return values
    # minimization problems (QUBO/UBQP): scores expected to be negative
    if min(values) >= 0:
        return [-val for val in values]
    return values


def _load_problem_config() -> tuple[str, int, int]:
    config_path = ROOT / "config" / "config.yaml"
    problem_name = "QUBO"
    dim = 128
    type_instance = 4
    try:
        cfg = OmegaConf.load(config_path)
        problem_key = cfg.get("problem")
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


def _build_instance_context() -> dict[str, Path | str | int]:
    problem_name, dim, type_instance = _load_problem_config()
    experiment_dir = ROOT / "results" / "experiments" / f"{problem_name}_dim{dim}_t{type_instance}"
    if problem_name.upper() == "QUBO":
        instance_name = f"UBQP_{dim}_{type_instance}"
        ranking_path = ROOT / "additional_results" / "global_ranking" / f"UBQP_N_{dim}_K_{type_instance}_ranks.csv"
    else:
        instance_name = f"{problem_name}_{dim}_{type_instance}"
        ranking_path = ROOT / "additional_results" / "global_ranking" / f"{problem_name}_N_{dim}_K_{type_instance}_ranks.csv"
    output_dir = ROOT / "courbes" / instance_name
    output_path = output_dir / f"comparison_{problem_name.lower()}_{dim}_t{type_instance}.png"
    return {
        "problem_name": problem_name,
        "dim": dim,
        "type_instance": type_instance,
        "instance_name": instance_name,
        "ranking_path": ranking_path,
        "experiment_dir": experiment_dir,
        "output_dir": output_dir,
        "my_data_path": experiment_dir / f"{problem_name}_{DEFAULT_KERNEL}_best_metrics.csv",
        "output_path": output_path,
        "kernels_output_dir": output_dir / "Kernels",
        "interact_vs_no_interact_dir": output_dir / "10000",
        "decay_vs_normal_dir": output_dir / "decay",
    }


def _build_instance_context_from_values(problem_name: str, dim: int, type_instance: int) -> dict[str, Path | str | int]:
    experiment_dir = ROOT / "results" / "experiments" / f"{problem_name}_dim{dim}_t{type_instance}"
    if problem_name.upper() == "QUBO":
        instance_name = f"UBQP_{dim}_{type_instance}"
        ranking_path = ROOT / "additional_results" / "global_ranking" / f"UBQP_N_{dim}_K_{type_instance}_ranks.csv"
    else:
        instance_name = f"{problem_name}_{dim}_{type_instance}"
        ranking_path = ROOT / "additional_results" / "global_ranking" / f"{problem_name}_N_{dim}_K_{type_instance}_ranks.csv"
    output_dir = ROOT / "courbes" / instance_name
    output_path = output_dir / f"comparison_{problem_name.lower()}_{dim}_t{type_instance}.png"
    return {
        "problem_name": problem_name,
        "dim": dim,
        "type_instance": type_instance,
        "instance_name": instance_name,
        "ranking_path": ranking_path,
        "experiment_dir": experiment_dir,
        "output_dir": output_dir,
        "my_data_path": experiment_dir / f"{problem_name}_{DEFAULT_KERNEL}_best_metrics.csv",
        "output_path": output_path,
        "kernels_output_dir": output_dir / "Kernels",
        "interact_vs_no_interact_dir": output_dir / "10000",
        "decay_vs_normal_dir": output_dir / "decay",
    }


def _discover_experiment_instances() -> List[dict[str, Path | str | int]]:
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
        name = match.group("name")
        dim = int(match.group("dim"))
        type_instance = int(match.group("t"))
        key = (name, dim, type_instance)
        if key in seen:
            continue
        seen.add(key)
        contexts.append(_build_instance_context_from_values(name, dim, type_instance))
    return sorted(contexts, key=lambda ctx: (str(ctx["problem_name"]), int(ctx["dim"]), int(ctx["type_instance"])))


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


def sort_by_x_with_std(
    x_vals: List[float], y_vals: List[float], std_vals: List[float]
) -> Tuple[List[float], List[float], List[float]]:
    pairs = sorted(zip(x_vals, y_vals, std_vals), key=lambda pair: pair[0])
    return [p[0] for p in pairs], [p[1] for p in pairs], [p[2] for p in pairs]


def find_competitor_files(
    algo: str,
    dim: int,
    type_instance: int,
    problem_name: str | None = None,
) -> Tuple[List[Path], bool]:
    for root in COMPETITOR_DIRS:
        csv_path = root / f"{algo}.csv"
        if csv_path.exists():
            return [csv_path], False

    algo_dir = None
    for root in COMPETITOR_DIRS:
        candidate = root / algo
        if candidate.exists():
            algo_dir = candidate
            break
    if algo_dir is None:
        roots = ", ".join(str(root) for root in COMPETITOR_DIRS)
        raise FileNotFoundError(f"Missing competitor directory for {algo} in {roots}")

    if problem_name:
        normalized = problem_name.upper()
        if normalized in ("QUBO", "UBQP"):
            problem_variants = ["UBQP", "QUBO", "qubo", "ubqp"]
        else:
            problem_variants = [normalized, normalized.lower()]
    else:
        problem_variants = ["UBQP", "QUBO", "qubo", "ubqp"]

    candidate_files: List[Path] = []
    for problem in problem_variants:
        candidate_dir = algo_dir / problem / str(dim) / str(type_instance)
        if candidate_dir.exists():
            candidate_files.extend(candidate_dir.glob("*.txt"))

    if not candidate_files:
        patterns: List[str] = []
        for problem in problem_variants:
            patterns.extend(
                (
                    f"**/results_nevergrad_{algo}_{problem}_{dim}_{type_instance}_*.txt",
                    f"**/*{algo}*{problem}*{dim}*{type_instance}*.txt",
                )
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


def plot_comparison(context: dict[str, Path | str | int]) -> None:
    ranking_path = context["ranking_path"]
    experiment_dir = context["experiment_dir"]
    my_data_path = context["my_data_path"]
    output_path = context["output_path"]
    problem_name = str(context["problem_name"])
    dim = int(context["dim"])
    type_instance = int(context["type_instance"])

    algos = load_top_algorithms(Path(ranking_path), limit=10, skip=("PPO-EDA",))
    best_kernel = find_best_kernel_summary(Path(experiment_dir), problem_name)
    best_metrics_path = Path(experiment_dir) / f"{problem_name}_{best_kernel}_best_metrics.csv"
    if best_metrics_path.exists():
        my_data_path = best_metrics_path

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    color_cycle = cycle(plt.cm.tab20.colors)

    for algo in algos:
        try:
            paths, has_header = find_competitor_files(algo, dim, type_instance, problem_name)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}. Skipping.")
            continue
        if has_header:
            if len(paths) > 1:
                x_vals, y_vals = load_mean_across_runs(paths)
            else:
                x_vals, y_vals = load_xy_from_csv(
                    paths[0], has_header=True, x_key="runtime", y_key="mean"
                )
        else:
            x_vals, y_vals = load_xy_from_csv(paths[0], has_header=False)
        y_vals = _normalize_score_sign(problem_name, y_vals)
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
        Path(my_data_path), has_header=True, x_key="step", y_key="best_fitness"
    )
    my_filtered = [(x, y) for x, y in zip(my_x, my_y) if x >= 100]
    if not my_filtered:
        raise ValueError("No reinforce svgd data at step >= 100")
    my_x, my_y = zip(*my_filtered)
    my_y = _normalize_score_sign(problem_name, list(my_y))
    ax.plot(
        my_x,
        my_y,
        label=f"reinforce svgd ({best_kernel})",
        color="green",
        linewidth=2.0,
        zorder=3,
        antialiased=True,
    )

    ax.set_title(f"Comparison on {problem_name} (N={dim}, K={type_instance})")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Average score")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(Path(output_path))
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_comparison_decay(context: dict[str, Path | str | int]) -> None:
    ranking_path = context["ranking_path"]
    experiment_dir = Path(context["experiment_dir"])
    decay_dir = experiment_dir / "decay"
    if not decay_dir.exists():
        print(f"[WARN] No decay directory found in {decay_dir}.")
        return
    output_dir = Path(context["decay_vs_normal_dir"])
    problem_name = str(context["problem_name"])
    dim = int(context["dim"])
    type_instance = int(context["type_instance"])

    algos = load_top_algorithms(Path(ranking_path), limit=10, skip=("PPO-EDA",))
    best_kernel = find_best_kernel_summary(decay_dir, problem_name)
    best_metrics_path = decay_dir / f"{problem_name}_{best_kernel}_best_metrics.csv"
    if not best_metrics_path.exists():
        raise FileNotFoundError(f"Missing decay metrics file {best_metrics_path}")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    color_cycle = cycle(plt.cm.tab20.colors)

    for algo in algos:
        try:
            paths, has_header = find_competitor_files(algo, dim, type_instance, problem_name)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}. Skipping.")
            continue
        if has_header:
            if len(paths) > 1:
                x_vals, y_vals = load_mean_across_runs(paths)
            else:
                x_vals, y_vals = load_xy_from_csv(
                    paths[0], has_header=True, x_key="runtime", y_key="mean"
                )
        else:
            x_vals, y_vals = load_xy_from_csv(paths[0], has_header=False)
        y_vals = _normalize_score_sign(problem_name, y_vals)
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
        best_metrics_path, has_header=True, x_key="step", y_key="best_fitness"
    )
    my_filtered = [(x, y) for x, y in zip(my_x, my_y) if x >= 100]
    if not my_filtered:
        raise ValueError("No reinforce svgd decay data at step >= 100")
    my_x, my_y = zip(*my_filtered)
    my_y = _normalize_score_sign(problem_name, list(my_y))
    ax.plot(
        my_x,
        my_y,
        label=f"reinforce svgd decay ({best_kernel})",
        color="red",
        linewidth=2.0,
        zorder=3,
        antialiased=True,
    )

    ax.set_title(f"Comparison (decay) on {problem_name} (N={dim}, K={type_instance})")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Average score")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"comparison_{problem_name.lower()}_{dim}_t{type_instance}_decay.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


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


def find_best_kernel_summary(
    summary_dir: Path, problem_name: str, maximize: bool | None = None
) -> str:
    summary_files = sorted(summary_dir.glob(f"{problem_name}_*_best_summary.txt"))
    if not summary_files:
        return DEFAULT_KERNEL

    if maximize is None:
        maximize = _is_maximization_problem(problem_name)
    best_kernel = DEFAULT_KERNEL
    best_score = None
    for path in summary_files:
        data = parse_summary_file(path)
        kernel = data.get("Kernel")
        avg_score = parse_float(data.get("avg_score"))
        if kernel is None or avg_score is None:
            continue
        if best_score is None:
            best_score = avg_score
            best_kernel = kernel
            continue
        if maximize and avg_score > best_score:
            best_score = avg_score
            best_kernel = kernel
        elif not maximize and avg_score < best_score:
            best_score = avg_score
            best_kernel = kernel
    return best_kernel


def list_kernels_from_metrics(metrics_dir: Path, problem_name: str) -> List[str]:
    prefix = f"{problem_name}_"
    suffix = "_best_metrics"
    kernels: List[str] = []
    for path in metrics_dir.glob(f"{problem_name}_*_best_metrics.csv"):
        stem = path.stem
        if not stem.startswith(prefix) or not stem.endswith(suffix):
            continue
        kernel = stem[len(prefix) : -len(suffix)]
        if kernel:
            kernels.append(kernel)
    return sorted(set(kernels))


def select_best_kernel_from_summaries(
    summary_dir: Path,
    problem_name: str,
    kernels: Iterable[str],
    maximize: bool | None = None,
) -> str | None:
    best_kernel = None
    best_score = None
    if maximize is None:
        maximize = _is_maximization_problem(problem_name)
    for kernel in kernels:
        summary_path = summary_dir / f"{problem_name}_{kernel}_best_summary.txt"
        if not summary_path.exists():
            continue
        data = parse_summary_file(summary_path)
        avg_score = parse_float(data.get("avg_score"))
        if avg_score is None:
            continue
        if best_score is None:
            best_score = avg_score
            best_kernel = kernel
            continue
        if maximize and avg_score > best_score:
            best_score = avg_score
            best_kernel = kernel
        elif not maximize and avg_score < best_score:
            best_score = avg_score
            best_kernel = kernel
    return best_kernel


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
        fieldnames = reader.fieldnames or []
        invert_best_fitness = False
        if y_field not in fieldnames and "best_fitness" in fieldnames:
            y_field = "best_fitness"
            invert_best_fitness = True
        for row in reader:
            x_val = parse_float(row.get(x_field))
            y_val = parse_float(row.get(y_field))
            if x_val is None or y_val is None:
                continue
            if invert_best_fitness:
                y_val = -y_val
            x_vals.append(x_val)
            y_vals.append(y_val)
    if not x_vals:
        raise ValueError(f"No data for {y_field} in {path}")
    return sort_by_x(x_vals, y_vals)


def load_metric_series_with_std(
    path: Path,
    x_field: str,
    mean_field: str = "mean",
    std_field: str = "std",
    maximize: bool | None = None,
) -> Tuple[List[float], List[float], List[float] | None]:
    x_vals: List[float] = []
    mean_vals: List[float] = []
    std_vals: List[float] = []
    rows: List[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    mean_key = mean_field if mean_field in fieldnames else "best_fitness"
    if mean_key not in fieldnames:
        mean_key = mean_field
    std_key = std_field if std_field in fieldnames else None
    invert_best_fitness = False
    if mean_key == "best_fitness" and maximize is True:
        invert_best_fitness = True
    for row in rows:
        x_val = parse_float(row.get(x_field))
        mean_val = parse_float(row.get(mean_key))
        std_val = parse_float(row.get(std_key)) if std_key else None
        if x_val is None or mean_val is None:
            continue
        if invert_best_fitness:
            mean_val = -mean_val
        x_vals.append(x_val)
        mean_vals.append(mean_val)
        if std_key and std_val is not None:
            std_vals.append(std_val)
    if not x_vals:
        raise ValueError(f"No data for {mean_key} in {path}")
    if std_key:
        if not std_vals:
            raise ValueError(f"No data for {std_key} in {path}")
        x_vals, mean_vals, std_vals = sort_by_x_with_std(x_vals, mean_vals, std_vals)
        return x_vals, mean_vals, std_vals
    x_vals, mean_vals = sort_by_x(x_vals, mean_vals)
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
) -> dict[str, float] | None:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        fieldnames = [name.strip() for name in (reader.fieldnames or []) if name]
        required = {x_field, low_field, q1_field, median_field, q3_field, high_field}
        if not required.issubset(set(fieldnames)):
            return None
        rows: List[dict[str, str]] = []
        for row in reader:
            cleaned = {key.strip(): value for key, value in row.items() if key}
            if cleaned.get(x_field) is None:
                continue
            rows.append(cleaned)

    if not rows:
        return None

    def _step_value(row: dict[str, str]) -> float:
        value = parse_float(row.get(x_field))
        return value if value is not None else float("-inf")

    last_row = max(rows, key=_step_value)
    values = {
        "low": parse_float(last_row.get(low_field)),
        "q1": parse_float(last_row.get(q1_field)),
        "med": parse_float(last_row.get(median_field)),
        "q3": parse_float(last_row.get(q3_field)),
        "high": parse_float(last_row.get(high_field)),
    }
    if any(value is None for value in values.values()):
        return None
    return {key: float(value) for key, value in values.items()}


def infer_quantile_x_field(path: Path) -> str:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        fieldnames = [name.strip() for name in (reader.fieldnames or []) if name]
    for key in ("runtime", "step", "evaluations", "evaluation", "eval", "budget"):
        if key in fieldnames:
            return key
    return "step"


def normalize_box_stats(problem_name: str, stats: dict[str, float]) -> dict[str, float]:
    values = [stats["low"], stats["q1"], stats["med"], stats["q3"], stats["high"]]
    normalized = _normalize_score_sign(problem_name, values)
    if normalized[0] > normalized[-1]:
        normalized = list(reversed(normalized))
    return {
        "low": normalized[0],
        "q1": normalized[1],
        "med": normalized[2],
        "q3": normalized[3],
        "high": normalized[4],
    }


def aggregate_quantile_stats(
    paths: List[Path],
    *,
    problem_name: str,
    x_field: str | None = None,
) -> dict[str, float] | None:
    stats_list: List[dict[str, float]] = []
    for path in paths:
        field = x_field or infer_quantile_x_field(path)
        stats = load_quantile_stats_at_final_step(path, x_field=field)
        if not stats:
            continue
        stats_list.append(normalize_box_stats(problem_name, stats))

    if not stats_list:
        return None

    keys = ("low", "q1", "med", "q3", "high")
    return {key: sum(item[key] for item in stats_list) / len(stats_list) for key in keys}


def style_axes(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9, width=0.6, length=3)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.5, alpha=0.4)


def plot_synthetic_boxplot(
    title: str,
    ylabel: str,
    output_path: Path,
    box_specs: List[tuple[str, dict[str, float]]],
    *,
    palette: dict[str, str] | None = None,
) -> None:
    if not box_specs:
        return

    stats = []
    labels = []
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

    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=180)
    default_palette = {
        "NORMAL": "#1f77b4",
        "DECAY": "#2ca02c",
        "NO_INTERACT": "#ff7f0e",
    }
    palette = palette or default_palette
    artists = ax.bxp(
        stats,
        showfliers=False,
        patch_artist=True,
        boxprops={"linewidth": 1.0, "edgecolor": "#2f2f2f"},
        whiskerprops={"linewidth": 1.0, "color": "#2f2f2f"},
        capprops={"linewidth": 1.0, "color": "#2f2f2f"},
        medianprops={"linewidth": 1.2, "color": "#111111"},
    )
    for box, label in zip(artists["boxes"], labels):
        box.set_facecolor(palette.get(label, "#9e9e9e"))
        box.set_alpha(0.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=10)
    style_axes(ax, grid_axis="y")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_interact_vs_no_interact_series(
    x_int: List[float],
    y_int: List[float],
    x_no: List[float],
    y_no: List[float],
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    std_int: List[float] | None = None,
    std_no: List[float] | None = None,
) -> None:
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
    if std_int is not None:
        lower = [y - s for y, s in zip(y_int, std_int)]
        upper = [y + s for y, s in zip(y_int, std_int)]
        ax.fill_between(x_int, lower, upper, color="#1f77b4", alpha=0.2, linewidth=0.0)
    if std_no is not None:
        lower = [y - s for y, s in zip(y_no, std_no)]
        upper = [y + s for y, s in zip(y_no, std_no)]
        ax.fill_between(x_no, lower, upper, color="#ff7f0e", alpha=0.2, linewidth=0.0)
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


def plot_pair_series(
    x_a: List[float],
    y_a: List[float],
    x_b: List[float],
    y_b: List[float],
    title: str,
    ylabel: str,
    output_path: Path,
    label_a: str,
    label_b: str,
    color_a: str,
    color_b: str,
    *,
    std_a: List[float] | None = None,
    std_b: List[float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=180)
    ax.plot(
        x_a,
        y_a,
        label=label_a,
        color=color_a,
        linewidth=1.4,
        alpha=0.95,
    )
    ax.plot(
        x_b,
        y_b,
        label=label_b,
        color=color_b,
        linewidth=1.4,
        linestyle="--",
        alpha=0.95,
    )
    if std_a is not None:
        lower = [y - s for y, s in zip(y_a, std_a)]
        upper = [y + s for y, s in zip(y_a, std_a)]
        ax.fill_between(x_a, lower, upper, color=color_a, alpha=0.2, linewidth=0.0)
    if std_b is not None:
        lower = [y - s for y, s in zip(y_b, std_b)]
        upper = [y + s for y, s in zip(y_b, std_b)]
        ax.fill_between(x_b, lower, upper, color=color_b, alpha=0.2, linewidth=0.0)
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


def plot_kernel_hyperparams(
    kernels: List[str], params: dict[str, List[float]], output_dir: Path, problem_name: str, dim: int, type_instance: int
) -> None:
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
    output_path = output_dir / f"{problem_name.lower()}_dim{dim}_t{type_instance}_hyperparams.png"
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
    *,
    experiment_dir: Path,
    output_dir: Path,
    problem_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.4), dpi=180)
    color_cycle = cycle(plt.cm.tab10.colors)
    for kernel in kernels:
        metrics_path = experiment_dir / f"{problem_name}_{kernel}_best_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file {metrics_path}")
        try:
            x_vals, y_vals = load_metric_series(metrics_path, x_field="step", y_field=metric_field)
        except ValueError as exc:
            print(f"[WARN] {exc}. Skipping {kernel}.")
            continue
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
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_kernel_comparison(context: dict[str, Path | str | int]) -> None:
    experiment_dir = Path(context["experiment_dir"])
    output_dir = Path(context["kernels_output_dir"])
    problem_name = str(context["problem_name"])
    dim = int(context["dim"])
    type_instance = int(context["type_instance"])
    summary_files = sorted(experiment_dir.glob(f"{problem_name}_*_best_summary.txt"))
    if not summary_files:
        raise FileNotFoundError(f"No summary files found in {experiment_dir}")

    kernel_data: dict[str, dict[str, str]] = {}
    for path in summary_files:
        data = parse_summary_file(path)
        kernel = data.get("Kernel")
        if not kernel:
            continue
        kernel_data[kernel] = data

    if not kernel_data:
        raise ValueError(f"No kernel entries found in {experiment_dir}")

    kernels = sorted(kernel_data.keys())
    hyperparams = {"M": [], "lambda": [], "epsilon_svgd": [], "gamma": []}

    for kernel in kernels:
        summary = kernel_data[kernel]
        for param in hyperparams:
            value = parse_float(summary.get(param))
            if value is None:
                raise ValueError(f"Missing {param} for kernel {kernel}")
            hyperparams[param].append(value)

    plot_kernel_hyperparams(kernels, hyperparams, output_dir, problem_name, dim, type_instance)
    plot_kernel_metric_curves(
        kernels,
        metric_field="mean",
        title="Average Score per Kernel (Evolution)",
        ylabel="Average score",
        filename=f"{problem_name.lower()}_dim{dim}_t{type_instance}_curve_avg_score.png",
        show_markers=False,
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        problem_name=problem_name,
    )
    plot_kernel_metric_curves(
        kernels,
        metric_field="avg_l1",
        title="Average L1 per Kernel (Evolution)",
        ylabel="Average L1",
        filename=f"{problem_name.lower()}_dim{dim}_t{type_instance}_curve_avg_l1.png",
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        problem_name=problem_name,
    )
    plot_kernel_metric_curves(
        kernels,
        metric_field="avg_hamming",
        title="Average Hamming per Kernel (Evolution)",
        ylabel="Average Hamming",
        filename=f"{problem_name.lower()}_dim{dim}_t{type_instance}_curve_avg_hamming.png",
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        problem_name=problem_name,
    )
    plot_kernel_metric_curves(
        kernels,
        metric_field="avg_entropy",
        title="Average Entropy per Kernel (Evolution)",
        ylabel="Average Entropy",
        filename=f"{problem_name.lower()}_dim{dim}_t{type_instance}_curve_avg_entropy.png",
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        problem_name=problem_name,
    )


def plot_kernel_interact_vs_no_interact(context: dict[str, Path | str | int]) -> None:
    experiment_dir = Path(context["experiment_dir"])
    output_dir = Path(context["interact_vs_no_interact_dir"])
    problem_name = str(context["problem_name"])
    dim = int(context["dim"])
    type_instance = int(context["type_instance"])
    interact_dir = experiment_dir
    no_interact_dir = experiment_dir / "no_interact"
    no_interact_kernels = list_kernels_from_metrics(no_interact_dir, problem_name)
    if not no_interact_kernels:
        print(f"[WARN] No no_interact metrics found in {no_interact_dir}.")
        return

    no_interact_kernel = select_best_kernel_from_summaries(
        no_interact_dir, problem_name, no_interact_kernels
    )
    if no_interact_kernel is None:
        no_interact_kernel = no_interact_kernels[0]

    best_kernel = find_best_kernel_summary(interact_dir, problem_name)

    interact_path = interact_dir / f"{problem_name}_{best_kernel}_best_metrics.csv"
    no_interact_path = no_interact_dir / f"{problem_name}_{no_interact_kernel}_best_metrics.csv"
    if not interact_path.exists() or not no_interact_path.exists():
        missing = []
        if not interact_path.exists():
            missing.append(str(interact_path))
        if not no_interact_path.exists():
            missing.append(str(no_interact_path))
        raise FileNotFoundError(f"Missing metrics file(s): {', '.join(missing)}")

    maximize = _is_maximization_problem(problem_name)
    x_int, y_int, std_int = load_metric_series_with_std(
        interact_path, x_field="step", maximize=maximize
    )
    x_no, y_no, std_no = load_metric_series_with_std(
        no_interact_path, x_field="step", maximize=maximize
    )

    title = (
        f"Average Score: interact vs no_interact ({problem_name} N={dim}, K={type_instance}, budget=10000)"
    )
    plot_interact_vs_no_interact_series(
        x_int,
        y_int,
        x_no,
        y_no,
        title=title,
        ylabel="Average score",
        output_path=output_dir / "avg_score_interact_vs_no_interact.png",
    )
    if std_int is not None and std_no is not None:
        plot_interact_vs_no_interact_series(
            x_int,
            y_int,
            x_no,
            y_no,
            title=title,
            ylabel="Average score",
            output_path=output_dir / "avg_score_interact_vs_no_interact_std.png",
            std_int=std_int,
            std_no=std_no,
        )
        plot_interact_vs_no_interact_series(
            x_int,
            std_int,
            x_no,
            std_no,
            title=f"Std: interact vs no_interact ({problem_name} N={dim}, K={type_instance}, budget=10000)",
            ylabel="Std",
            output_path=output_dir / "std_interact_vs_no_interact.png",
        )
    else:
        print(f"[WARN] Missing std column in {interact_path} or {no_interact_path}.")


def plot_decay_comparison(context: dict[str, Path | str | int]) -> None:
    experiment_dir = Path(context["experiment_dir"])
    decay_dir = experiment_dir / "decay"
    if not decay_dir.exists():
        print(f"[WARN] No decay directory found in {decay_dir}.")
        return

    output_dir = Path(context["decay_vs_normal_dir"])
    problem_name = str(context["problem_name"])
    dim = int(context["dim"])
    type_instance = int(context["type_instance"])

    best_kernel_normal = find_best_kernel_summary(experiment_dir, problem_name)
    best_kernel_decay = find_best_kernel_summary(decay_dir, problem_name)

    normal_path = experiment_dir / f"{problem_name}_{best_kernel_normal}_best_metrics.csv"
    decay_path = decay_dir / f"{problem_name}_{best_kernel_decay}_best_metrics.csv"
    if not normal_path.exists() or not decay_path.exists():
        missing = []
        if not normal_path.exists():
            missing.append(str(normal_path))
        if not decay_path.exists():
            missing.append(str(decay_path))
        raise FileNotFoundError(f"Missing metrics file(s): {', '.join(missing)}")

    maximize = _is_maximization_problem(problem_name)
    x_norm, y_norm, std_norm = load_metric_series_with_std(
        normal_path, x_field="step", maximize=maximize
    )
    x_decay, y_decay, std_decay = load_metric_series_with_std(
        decay_path, x_field="step", maximize=maximize
    )

    title = (
        f"Average Score: normal vs decay ({problem_name} N={dim}, K={type_instance}, budget=10000)"
    )
    label_norm = f"normal ({best_kernel_normal})"
    label_decay = f"decay ({best_kernel_decay})"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_pair_series(
        x_norm,
        y_norm,
        x_decay,
        y_decay,
        title=title,
        ylabel="Average score",
        output_path=output_dir / "avg_score_normal_vs_decay.png",
        label_a=label_norm,
        label_b=label_decay,
        color_a="#2ca02c",
        color_b="#d62728",
    )

    if std_norm is not None and std_decay is not None:
        plot_pair_series(
            x_norm,
            y_norm,
            x_decay,
            y_decay,
            title=title,
            ylabel="Average score",
            output_path=output_dir / "avg_score_normal_vs_decay_std.png",
            label_a=label_norm,
            label_b=label_decay,
            color_a="#2ca02c",
            color_b="#d62728",
            std_a=std_norm,
            std_b=std_decay,
        )
        plot_pair_series(
            x_norm,
            std_norm,
            x_decay,
            std_decay,
            title=f"Std: normal vs decay ({problem_name} N={dim}, K={type_instance}, budget=10000)",
            ylabel="Std",
            output_path=output_dir / "std_normal_vs_decay.png",
            label_a=label_norm,
            label_b=label_decay,
            color_a="#2ca02c",
            color_b="#d62728",
        )
    else:
        print(f"[WARN] Missing std column in {normal_path} or {decay_path}.")

    try:
        x_norm_h, y_norm_h = load_metric_series(normal_path, x_field="step", y_field="avg_hamming")
        x_decay_h, y_decay_h = load_metric_series(decay_path, x_field="step", y_field="avg_hamming")
    except ValueError as exc:
        print(f"[WARN] {exc}.")
    else:
        plot_pair_series(
            x_norm_h,
            y_norm_h,
            x_decay_h,
            y_decay_h,
            title=f"Average Hamming: normal vs decay ({problem_name} N={dim}, K={type_instance}, budget=10000)",
            ylabel="Average hamming",
            output_path=output_dir / "hamming_normal_vs_decay.png",
            label_a=label_norm,
            label_b=label_decay,
            color_a="#2ca02c",
            color_b="#d62728",
        )

    try:
        x_norm_l1, y_norm_l1 = load_metric_series(normal_path, x_field="step", y_field="avg_l1")
        x_decay_l1, y_decay_l1 = load_metric_series(decay_path, x_field="step", y_field="avg_l1")
    except ValueError as exc:
        print(f"[WARN] {exc}.")
    else:
        plot_pair_series(
            x_norm_l1,
            y_norm_l1,
            x_decay_l1,
            y_decay_l1,
            title=f"Average L1: normal vs decay ({problem_name} N={dim}, K={type_instance}, budget=10000)",
            ylabel="Average L1",
            output_path=output_dir / "l1_normal_vs_decay.png",
            label_a=label_norm,
            label_b=label_decay,
            color_a="#2ca02c",
            color_b="#d62728",
        )


def plot_final_score_boxplot(context: dict[str, Path | str | int]) -> None:
    experiment_dir = Path(context["experiment_dir"])
    output_dir = Path(context["output_dir"])
    problem_name = str(context["problem_name"])
    dim = int(context["dim"])
    type_instance = int(context["type_instance"])

    normal_kernel = find_best_kernel_summary(experiment_dir, problem_name)
    normal_path = experiment_dir / f"{problem_name}_{normal_kernel}_best_metrics.csv"

    decay_dir = experiment_dir / "decay"
    decay_path = None
    if decay_dir.exists():
        decay_kernel = find_best_kernel_summary(decay_dir, problem_name)
        decay_path = decay_dir / f"{problem_name}_{decay_kernel}_best_metrics.csv"

    no_interact_dir = experiment_dir / "no_interact"
    no_interact_path = None
    if no_interact_dir.exists():
        no_interact_kernels = list_kernels_from_metrics(no_interact_dir, problem_name)
        if no_interact_kernels:
            no_interact_kernel = select_best_kernel_from_summaries(
                no_interact_dir, problem_name, no_interact_kernels
            )
            if no_interact_kernel is None:
                no_interact_kernel = no_interact_kernels[0]
            no_interact_path = no_interact_dir / f"{problem_name}_{no_interact_kernel}_best_metrics.csv"

    box_specs: List[tuple[str, dict[str, float]]] = []
    if normal_path.exists():
        stats = load_quantile_stats_at_final_step(normal_path)
        if stats:
            box_specs.append(("NORMAL", stats))
    if decay_path and decay_path.exists():
        stats = load_quantile_stats_at_final_step(decay_path)
        if stats:
            box_specs.append(("DECAY", stats))
    if no_interact_path and no_interact_path.exists():
        stats = load_quantile_stats_at_final_step(no_interact_path)
        if stats:
            box_specs.append(("NO_INTERACT", stats))

    plot_synthetic_boxplot(
        title=f"Final score distribution ({problem_name} N={dim}, K={type_instance})",
        ylabel="Score",
        output_path=output_dir / "boxplot_final_score.png",
        box_specs=box_specs,
    )


def plot_top5_competitors_boxplot(context: dict[str, Path | str | int]) -> None:
    ranking_path = context["ranking_path"]
    experiment_dir = Path(context["experiment_dir"])
    output_dir = Path(context["output_dir"])
    problem_name = str(context["problem_name"])
    dim = int(context["dim"])
    type_instance = int(context["type_instance"])

    best_kernel = find_best_kernel_summary(experiment_dir, problem_name)
    my_metrics_path = experiment_dir / f"{problem_name}_{best_kernel}_best_metrics.csv"
    if not my_metrics_path.exists():
        print(f"[WARN] Missing metrics file {my_metrics_path}. Skipping top-5 boxplot.")
        return

    my_stats = load_quantile_stats_at_final_step(my_metrics_path, x_field="step")
    if not my_stats:
        print(f"[WARN] Missing quantiles in {my_metrics_path}. Skipping top-5 boxplot.")
        return
    my_label = f"OURS ({best_kernel})"
    box_specs: List[tuple[str, dict[str, float]]] = [
        (my_label, normalize_box_stats(problem_name, my_stats))
    ]

    algos = load_top_algorithms(Path(ranking_path), limit=5, skip=("PPO-EDA",))
    for algo in algos:
        try:
            paths, has_header = find_competitor_files(algo, dim, type_instance, problem_name)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}. Skipping.")
            continue
        if not has_header:
            print(f"[WARN] Missing quantile headers for {algo}. Skipping.")
            continue
        stats = aggregate_quantile_stats(paths, problem_name=problem_name)
        if not stats:
            print(f"[WARN] Missing quantiles for {algo}. Skipping.")
            continue
        box_specs.append((algo, stats))

    if len(box_specs) <= 1:
        print("[WARN] No competitor stats available for top-5 boxplot.")
        return

    palette: dict[str, str] = {my_label: "#2ca02c"}
    color_cycle = cycle(plt.cm.tab20.colors)
    for label, _ in box_specs:
        if label == my_label:
            continue
        palette[label] = next(color_cycle)

    plot_synthetic_boxplot(
        title=f"Final score distribution: ours vs top-5 ({problem_name} N={dim}, K={type_instance})",
        ylabel="Score",
        output_path=output_dir / "boxplot_final_score_vs_top5.png",
        box_specs=box_specs,
        palette=palette,
    )


def main() -> None:
    contexts = _discover_experiment_instances()
    if not contexts:
        contexts = [_build_instance_context()]
    for context in contexts:
        instance_name = context["instance_name"]
        print(f"[INFO] Plotting instance: {instance_name}")
        try:
            plot_comparison(context)
        except Exception as exc:
            print(f"[WARN] comparison failed for {instance_name}: {exc}")
        try:
            plot_comparison_decay(context)
        except Exception as exc:
            print(f"[WARN] comparison_decay failed for {instance_name}: {exc}")
        try:
            plot_kernel_comparison(context)
        except Exception as exc:
            print(f"[WARN] kernel comparison failed for {instance_name}: {exc}")
        try:
            plot_kernel_interact_vs_no_interact(context)
        except Exception as exc:
            print(f"[WARN] interact_vs_no_interact failed for {instance_name}: {exc}")
        try:
            plot_decay_comparison(context)
        except Exception as exc:
            print(f"[WARN] decay comparison failed for {instance_name}: {exc}")
        try:
            plot_final_score_boxplot(context)
        except Exception as exc:
            print(f"[WARN] final score boxplot failed for {instance_name}: {exc}")
        try:
            plot_top5_competitors_boxplot(context)
        except Exception as exc:
            print(f"[WARN] top-5 competitor boxplot failed for {instance_name}: {exc}")


if __name__ == "__main__":
    main()
