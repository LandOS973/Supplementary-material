#!/usr/bin/env python3
"""
Rebuild overall_summary.xlsx by scanning results/config/* folders.
"""

import os
from pathlib import Path

from openpyxl import Workbook

from main_expe_overall import _collect_config_stats


SUMMARY_HEADERS = [
    "config_name",
    "kernel",
    "advantage",
    "M",
    "lambda_",
    "epsilon_svgd",
    "gamma",
    "decay_start_ratio",
    "decay_min_factor",
    "nasbench_avg_score",
    "mean_rank",
    "median_rank",
    "std_percent",
    "top1_count",
    "top3_count",
    "top5_count",
    "top10_count",
    "top_1_nk",
    "top_1_nk3",
    "top_1_qubo",
    "win_rate_mean",
    "mean_hamming_norm",
    "mean_l1_norm",
    "hasRawScore",
    "n_instances",
    "n_ranked",
]


def _infer_params_from_name(config_name: str) -> dict:
    # Parse config_name like: kjsd__advperagentrankweighted__M3__L20__eps0p005__g0p0005__ds0p05__dm0p05
    parts = config_name.split("__")
    out = {
        "kernel": None,
        "advantage": None,
        "M": None,
        "lambda_": None,
        "epsilon_svgd": None,
        "gamma": None,
        "decay_start_ratio": None,
        "decay_min_factor": None,
        "bandwith_kernel": None,
    }
    def parse_float(token: str):
        token = token.replace("p", ".").replace("m", "-")
        try:
            return float(token)
        except Exception:
            return None
    for p in parts:
        if p.startswith("k"):
            out["kernel"] = p[1:]
        elif p.startswith("adv"):
            out["advantage"] = p[3:]
        elif p.startswith("M"):
            try:
                out["M"] = int(p[1:])
            except Exception:
                pass
        elif p.startswith("L"):
            try:
                out["lambda_"] = int(p[1:])
            except Exception:
                pass
        elif p.startswith("eps"):
            out["epsilon_svgd"] = parse_float(p[3:])
        elif p.startswith("g"):
            out["gamma"] = parse_float(p[1:])
        elif p.startswith("ds"):
            out["decay_start_ratio"] = parse_float(p[2:])
        elif p.startswith("dm"):
            out["decay_min_factor"] = parse_float(p[2:])
        elif p.startswith("bw"):
            out["bandwith_kernel"] = parse_float(p[2:])
    return out


def _has_raw_score(cfg_dir: Path) -> int:
    # Any raw_scores.csv under the config directory counts.
    return 1 if any(cfg_dir.rglob("raw_scores.csv")) else 0


def _nasbench_avg_score(cfg_dir: Path):
    metrics_path = cfg_dir / "nasbench" / "best_metrics.csv"
    if not metrics_path.is_file():
        return None
    try:
        lines = [line.strip() for line in metrics_path.read_text().splitlines() if line.strip()]
        if len(lines) < 2:
            return None
        header = [h.strip() for h in lines[0].split(",")]
        last = [v.strip() for v in lines[-1].split(",")]

        def pick(col: str):
            if col in header:
                idx = header.index(col)
                if idx < len(last):
                    return last[idx]
            return None

        val = pick("mean") or pick("median") or pick("best_fitness")
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_root = os.path.join(repo_root, "results", "config")
    out_xlsx = os.path.join(out_root, "overall_summary.xlsx")

    rows = []
    for cfg_dir in sorted(Path(out_root).iterdir()):
        if not cfg_dir.is_dir():
            continue
        config_name = cfg_dir.name
        params = _infer_params_from_name(config_name)
        if not params.get("kernel") or params.get("M") is None or params.get("lambda_") is None:
            continue
        stats = _collect_config_stats(str(cfg_dir), config_name, params, repo_root)
        stats["nasbench_avg_score"] = _nasbench_avg_score(cfg_dir)
        stats["hasRawScore"] = _has_raw_score(cfg_dir)
        rows.append(stats)

    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    ws.append(SUMMARY_HEADERS)
    for row in rows:
        ws.append([row.get(h) for h in SUMMARY_HEADERS])

    wb.save(out_xlsx)
    print(f"[DONE] Rebuilt: {out_xlsx} ({len(rows)} configs)")


if __name__ == "__main__":
    main()
