"""
Generate a LaTeX table of sensitivity scores per M from existing CSV summaries.

The script expects sensitivity CSVs under:
  results/config/<config_name>/sensitivity/

It reads any of:
  - sensitivity_qubo_from_config.csv
  - sensitivity_nk_from_config.csv
  - sensitivity_nk3_from_config.csv

Output:
  results/config/<config_name>/sensitivity/sensitivity_m_scores.tex
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = ROOT / "results" / "config"


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _first_key(row: Dict[str, str], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    return None


def _read_sensitivity_csv(path: Path) -> Dict[Tuple[str, int, int], Dict[int, float]]:
    data: Dict[Tuple[str, int, int], Dict[int, float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            problem = _first_key(row, ["problem", "Problem"])
            if not problem:
                continue
            n_str = _first_key(row, ["n", "N"])
            t_str = _first_key(row, ["type_instance", "type", "K", "k"])
            m_str = _first_key(row, ["M", "m"])
            mean_str = _first_key(row, ["mean", "avg", "avg_score", "score_mean"])

            n = _parse_int(n_str)
            t = _parse_int(t_str)
            m = _parse_int(m_str)
            mean = _parse_float(mean_str)
            if n is None or t is None or m is None or mean is None:
                continue

            key = (problem, n, t)
            data.setdefault(key, {})[m] = mean
    return data


def _format_score(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _build_latex_table(data: Dict[Tuple[str, int, int], Dict[int, float]]) -> str:
    if not data:
        raise ValueError("No data found to build table.")

    m_values = sorted({m for scores in data.values() for m in scores.keys()})
    problem_order = ["NK", "NK3", "QUBO"]
    problems_in_data = {key[0] for key in data.keys()}
    ordered_problems = [p for p in problem_order if p in problems_in_data]
    ordered_problems.extend(sorted(problems_in_data - set(ordered_problems)))

    grouped: Dict[str, Dict[int, List[Tuple[int, Dict[int, float]]]]] = {}
    for (problem, n, t), scores in data.items():
        grouped.setdefault(problem, {}).setdefault(n, []).append((t, scores))

    for problem in grouped:
        for n in grouped[problem]:
            grouped[problem][n].sort(key=lambda item: item[0])

    col_spec = "lrr" + "".join(["c" for _ in m_values])
    header_cols = ["Problem", "Dim (n)", "Type/K"] + [f"m={m}" for m in m_values]

    lines: List[str] = []
    lines.append("% Table: Sensitivity scores per M")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Sensitivity analysis: average scores for each $M$ value. Bold values highlight the best (maximum) score per instance.}")
    lines.append("\\label{tab:sensitivity_analysis}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" " + " & ".join(header_cols) + " \\\\")
    lines.append("\\midrule")

    col_end = 3 + len(m_values)
    for p_idx, problem in enumerate(ordered_problems):
        if problem not in grouped:
            continue
        n_map = grouped[problem]
        n_values = sorted(n_map.keys())
        problem_rows = sum(len(n_map[n]) for n in n_values)
        first_problem_row = True

        for n_idx, n in enumerate(n_values):
            t_list = n_map[n]
            n_rows = len(t_list)
            first_n_row = True

            for t, scores in t_list:
                best_val = max(scores.values()) if scores else None
                row_cells: List[str] = []

                if first_problem_row:
                    row_cells.append(f"\\multirow{{{problem_rows}}}{{*}}{{{problem}}}")
                else:
                    row_cells.append("")

                if first_n_row:
                    row_cells.append(f"\\multirow{{{n_rows}}}{{*}}{{{n}}}")
                else:
                    row_cells.append("")

                row_cells.append(str(t))

                for m in m_values:
                    val = scores.get(m)
                    if val is None:
                        cell = "--"
                    else:
                        formatted = _format_score(val)
                        if best_val is not None and abs(val - best_val) <= 1e-12:
                            formatted = f"\\best{{{formatted}}}"
                        cell = formatted
                    row_cells.append(cell)

                lines.append(" " + " & ".join(row_cells) + " \\\\")
                first_problem_row = False
                first_n_row = False

            if n_idx < len(n_values) - 1:
                lines.append(f"\\cmidrule(lr){{2-{col_end}}}")

        if p_idx < len(ordered_problems) - 1:
            lines.append("\\midrule[1.2pt]")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        config_name = input("Config name: ").strip()

    if not config_name:
        print("Error: config name is required.")
        return

    sensitivity_dir = CONFIG_ROOT / config_name / "sensitivity"
    if not sensitivity_dir.exists():
        print(f"Error: sensitivity directory not found: {sensitivity_dir}")
        return

    data: Dict[Tuple[str, int, int], Dict[int, float]] = {}
    csv_files = [
        sensitivity_dir / "sensitivity_qubo_from_config.csv",
        sensitivity_dir / "sensitivity_nk_from_config.csv",
        sensitivity_dir / "sensitivity_nk3_from_config.csv",
    ]

    found_any = False
    for csv_path in csv_files:
        if csv_path.exists():
            found_any = True
            per_file = _read_sensitivity_csv(csv_path)
            for key, scores in per_file.items():
                data.setdefault(key, {}).update(scores)

    if not found_any:
        print(f"No sensitivity CSV files found in {sensitivity_dir}")
        return

    if not data:
        print("No rows parsed from sensitivity CSVs.")
        return

    latex = _build_latex_table(data)

    output_tex = sensitivity_dir / "sensitivity_m_scores.tex"
    output_tex.write_text(latex, encoding="utf-8")

    print(f"Wrote LaTeX table to: {output_tex}")


if __name__ == "__main__":
    main()
