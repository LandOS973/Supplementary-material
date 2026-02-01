#!/usr/bin/env python3
"""
Rebuild overall_summary.xlsx by scanning results/config/* folders.
"""

import os
from pathlib import Path

from openpyxl import Workbook

from main_expe_overall import SUMMARY_HEADERS, _collect_config_stats


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
