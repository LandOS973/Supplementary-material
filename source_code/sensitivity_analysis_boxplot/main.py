#!/usr/bin/env python3
import argparse
import csv
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _to_number_if_possible(value: str) -> Any:
    if value is None:
        return value
    v = value.strip()
    if v == "":
        return value
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return value


def _sort_key_for_value(value: Any) -> Tuple[int, Any]:
    if isinstance(value, (int, float)):
        return (0, value)
    return (1, str(value))


def summarize_csv(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {input_path}")
        fieldnames = reader.fieldnames
        if "M" not in fieldnames or "fitness" not in fieldnames:
            raise ValueError(
                f"CSV must contain 'M' and 'fitness' columns: {input_path}"
            )
        descriptor_cols = [c for c in fieldnames if c not in ("M", "fitness")]

        groups: Dict[Tuple[Any, ...], List[float]] = {}
        for row in reader:
            key_values: List[Any] = []
            for col in descriptor_cols:
                key_values.append(_to_number_if_possible(row.get(col, "")))
            key_values.append(_to_number_if_possible(row.get("M", "")))
            key = tuple(key_values)

            try:
                fitness = float(row.get("fitness", ""))
            except ValueError:
                continue

            groups.setdefault(key, []).append(fitness)

    rows_out: List[Dict[str, Any]] = []
    for key, values in groups.items():
        descriptor_vals = key[:-1]
        m_val = key[-1]
        median_val = statistics.median(values)
        row_out: Dict[str, Any] = {}
        for col, val in zip(descriptor_cols, descriptor_vals):
            row_out[col] = val
        row_out["M"] = m_val
        row_out["fitness_median"] = median_val
        row_out["fitness_min"] = min(values)
        row_out["fitness_max"] = max(values)
        row_out["count"] = len(values)
        rows_out.append(row_out)

    def sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
        parts: List[Any] = []
        for col in descriptor_cols:
            parts.append(_sort_key_for_value(row[col]))
        parts.append(_sort_key_for_value(row["M"]))
        return tuple(parts)

    rows_out.sort(key=sort_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_fields = descriptor_cols + [
        "M",
        "fitness_median",
        "fitness_min",
        "fitness_max",
        "count",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    default_nk = base_dir / "sensitivity_nk_boxplot_results.csv"
    default_qubo = (
        base_dir.parent
        / "sensitivity_analysis_qubo_boxplot"
        / "sensitivity_qubo_boxplot_results.csv"
    )
    default_out_nk = base_dir / "sensitivity_nk_boxplot_summary.csv"
    default_out_qubo = (
        base_dir.parent
        / "sensitivity_analysis_qubo_boxplot"
        / "sensitivity_qubo_boxplot_summary.csv"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Summarize sensitivity boxplot CSVs by instance type and M (median/min/max)."
        )
    )
    parser.add_argument(
        "--nk",
        default=str(default_nk),
        help="Path to NK CSV input",
    )
    parser.add_argument(
        "--qubo",
        default=str(default_qubo),
        help="Path to QUBO CSV input",
    )
    parser.add_argument(
        "--out-nk",
        default=str(default_out_nk),
        help="Path to NK CSV output",
    )
    parser.add_argument(
        "--out-qubo",
        default=str(default_out_qubo),
        help="Path to QUBO CSV output",
    )
    args = parser.parse_args()

    summarize_csv(Path(args.nk), Path(args.out_nk))
    summarize_csv(Path(args.qubo), Path(args.out_qubo))


if __name__ == "__main__":
    main()
