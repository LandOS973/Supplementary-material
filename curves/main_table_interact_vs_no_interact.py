"""Build a per-instance table: Interact vs No-Interact scores + gap (%).

Reads results from:
  results/config/<ConfigName>/<InstanceName>/best_metrics.csv
  results/config/<ConfigName>/<InstanceName>/no_interact/best_metrics.csv

Outputs CSV and LaTeX (with bold on best score).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parent.parent
INSTANCE_RE = re.compile(r"^(?P<problem>QUBO|NK)_dim(?P<dim>\d+)_t(?P<t>\d+)$")
DEFAULT_CONFIG_NAME = "krbf__advglobalrankweighted__M7__L13__eps0p08__g0p015__ds0p03__dm0p01"


def _read_last_metric_score(metrics_path: Path) -> float | None:
    try:
        lines = [line.strip() for line in metrics_path.read_text().splitlines() if line.strip()]
        if len(lines) < 2:
            return None
        header = [h.strip() for h in lines[0].split(",")]
        last = [v.strip() for v in lines[-1].split(",")]

        def pick(col: str):
            if col in header:
                idx = header.index(col)
                if idx < len(last) and last[idx] != "":
                    return last[idx]
            return None

        for col in ("mean", "median", "best_fitness"):
            val = pick(col)
            if val is not None:
                return float(val)
        return None
    except Exception:
        return None


def _is_maximize(problem: str) -> bool:
    return problem.upper() == "NK"


def _format_score(problem: str, value: float | None) -> str:
    if value is None:
        return "—"
    value = abs(value)
    if problem.upper() == "NK":
        return f"{value:.4f}"
    return f"{value:.2f}"


def _bold(text: str) -> str:
    return f"\\best{{{text}}}"


def _gap_percent(problem: str, score_interact: float, score_no: float) -> float:
    if not _is_maximize(problem):
        score_interact = -score_interact
        score_no = -score_no
    if score_no == 0:
        return 0.0
    return (score_interact - score_no) / abs(score_no) * 100.0


def _iter_instances(config_dir: Path) -> List[Tuple[str, int, int]]:
    instances: List[Tuple[str, int, int]] = []
    if not config_dir.exists():
        return instances
    for entry in config_dir.iterdir():
        if not entry.is_dir():
            continue
        m = INSTANCE_RE.match(entry.name)
        if not m:
            continue
        instances.append((m.group("problem"), int(m.group("dim")), int(m.group("t"))))
    order = {"QUBO": 0, "NK": 1}
    return sorted(instances, key=lambda x: (order.get(x[0], 99), x[1], x[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Interact vs no_interact table from config results.")
    parser.add_argument("--config", type=str, default=None, help="Config name (otherwise prompted).")
    parser.add_argument("--csv", type=Path, default=ROOT / "curves" / "table_interact_vs_no_interact.csv")
    parser.add_argument("--tex", type=Path, default=ROOT / "curves" / "table_interact_vs_no_interact.tex")
    args = parser.parse_args()

    config_name = args.config or (
        input(
            f"Config name (ex: {DEFAULT_CONFIG_NAME}) [default: {DEFAULT_CONFIG_NAME}]: "
        ).strip()
        or DEFAULT_CONFIG_NAME
    )

    config_dir = ROOT / "results" / "config" / config_name
    if not config_dir.exists():
        raise SystemExit(f"Config directory not found: {config_dir}")

    rows: List[List[str]] = []
    rows_csv: List[List[str]] = []
    rows_meta: List[Tuple[str, int, int]] = []
    header = ["Problem", "Dim (n)", "Type/K", "Score (Interact)", "Score (No Interact)", "Gap (%)"]

    for problem, dim, t in _iter_instances(config_dir):
        inst_dir = config_dir / f"{problem}_dim{dim}_t{t}"
        interact_metrics = inst_dir / "best_metrics.csv"
        no_metrics = inst_dir / "no_interact" / "best_metrics.csv"
        if not interact_metrics.exists() or not no_metrics.exists():
            continue
        score_interact = _read_last_metric_score(interact_metrics)
        score_no = _read_last_metric_score(no_metrics)
        if score_interact is None or score_no is None:
            continue

        gap = _gap_percent(problem, score_interact, score_no)
        gap_str = f"{gap:+.2f}%"
        gap_str_tex = gap_str.replace("%", "\\%")

        if _is_maximize(problem):
            best_is_interact = score_interact >= score_no
        else:
            best_is_interact = score_interact <= score_no

        score_interact_str = _format_score(problem, score_interact)
        score_no_str = _format_score(problem, score_no)

        rows.append(
            [
                problem,
                str(dim),
                str(t),
                _bold(score_interact_str) if best_is_interact else score_interact_str,
                _bold(score_no_str) if not best_is_interact else score_no_str,
                gap_str_tex,
            ]
        )
        rows_csv.append(
            [
                problem,
                str(dim),
                str(t),
                score_interact_str,
                score_no_str,
                gap_str,
            ]
        )
        rows_meta.append((problem, dim, t))

    if not rows:
        print("[WARN] No rows found. Check that best_metrics.csv exist for interact and no_interact.")
        return

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows_csv:
            f.write(",".join(r) + "\n")

    args.tex.parent.mkdir(parents=True, exist_ok=True)
    with args.tex.open("w") as f:
        f.write("% Table: Interact vs No-Interact\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write(
            "\\caption{Detailed comparison between Interacting Agents (SVGD-EDA) and Independent Agents. "
            "Bold values indicate the best average score between the two methods.}\n"
        )
        f.write("\\label{tab:interact_vs_no_interact}\n")
        f.write("\\begin{tabular}{lrrccc}\n")
        f.write("\\toprule\n")
        f.write("Problem & Dim (n) & Type/K & Score (Interact) & Score (No Interact) & Gap (\\%) \\\\\n")
        f.write("\\midrule\n")
        prev_problem = None
        prev_dim = None
        for meta, r in zip(rows_meta, rows):
            problem, dim, _t = meta
            if prev_problem is not None and (problem != prev_problem or dim != prev_dim):
                f.write("\\midrule\n")
            f.write("{} & {} & {} & {} & {} & {} \\\\\n".format(*r))
            prev_problem = problem
            prev_dim = dim
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved CSV to {args.csv}")
    print(f"Saved LaTeX to {args.tex}")


if __name__ == "__main__":
    main()
