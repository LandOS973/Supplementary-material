#!/usr/bin/env python3
"""
Batch runner to test greedy-final behavior across NK / NK3 / QUBO on multiple budgets.
Console-only output (no visualization files).
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path


BUDGETS = [1000, 5000, 10000, 30000, 40000, 50000]

EXPERIMENTS = [
    {
        "label": "NK | petite dim | petite rugosite",
        "problem": "nk",
        "dim": 64,
        "type_instance": 1,
    },
    {
        "label": "NK3 | dim moyenne | rugosite moyenne",
        "problem": "nk3",
        "dim": 128,
        "type_instance": 4,
    },
    {
        "label": "QUBO | grande dim | type 5",
        "problem": "qubo",
        "dim": 256,
        "type_instance": 5,
    },
]


def _extract_average_test_score(output: str):
    match = re.search(r"average_test_score:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_greedy_summary(output: str):
    pattern = re.compile(
        r"Instance\s+\d+\s+=> nb d'agent qui ameliorent le score\s+([-\d.]+)\s+\(([-+]?\d*\.?\d+)\s+de score(?:\s+normalise)?\s+en moyenne\)"
    )
    counts = []
    gains = []
    for match in pattern.finditer(output):
        try:
            counts.append(float(match.group(1)))
            gains.append(float(match.group(2)))
        except ValueError:
            continue
    if not counts:
        return None, None
    mean_count = sum(counts) / len(counts)
    mean_gain = sum(gains) / len(gains)
    return mean_count, mean_gain


def run_once(script_dir: Path, exp: dict, budget: int) -> dict:
    cmd = [
        sys.executable,
        "main.py",
        f"problem={exp['problem']}",
        f"problem.dim={exp['dim']}",
        f"problem.type_instance={exp['type_instance']}",
        f"budget={budget}",
        "seed=0",
        "verbose=false",
        "visualization=false",
        "agent.enable_greedy_final=true",
    ]

    print("=" * 100)
    print(
        f"[RUN] {exp['label']} | problem={exp['problem']} dim={exp['dim']} "
        f"type={exp['type_instance']} | budget={budget}"
    )
    print(" ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=script_dir,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    dt = time.time() - t0
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    print(f"[END] return_code={proc.returncode} runtime={dt:.2f}s")
    greedy_agents, greedy_gain = _extract_greedy_summary(proc.stdout or "")
    return {
        "label": exp["label"],
        "problem": exp["problem"],
        "dim": exp["dim"],
        "type_instance": exp["type_instance"],
        "budget": budget,
        "return_code": proc.returncode,
        "runtime_s": dt,
        "avg_score": _extract_average_test_score(proc.stdout or ""),
        "avg_greedy_agents": greedy_agents,
        "avg_greedy_gain": greedy_gain,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    failures = 0
    results = []

    for exp in EXPERIMENTS:
        print("\n" + "#" * 100)
        print(f"# TEST SUITE: {exp['label']}")
        print("#" * 100)
        for budget in BUDGETS:
            res = run_once(script_dir, exp, budget)
            results.append(res)
            if res["return_code"] != 0:
                failures += 1

    print("\n" + "=" * 100)
    if failures == 0:
        print("Tous les runs se sont termines sans erreur.")
    else:
        print(f"Runs en erreur: {failures}")

    print("=" * 100)
    print("RAPPORT FINAL")
    print("=" * 100)
    for exp in EXPERIMENTS:
        print(
            f"\n[{exp['label']}] problem={exp['problem']} dim={exp['dim']} type={exp['type_instance']}"
        )
        exp_results = [r for r in results if r["label"] == exp["label"]]
        exp_results.sort(key=lambda r: r["budget"])
        for r in exp_results:
            score_str = f"{r['avg_score']:.6f}" if r["avg_score"] is not None else "N/A"
            greedy_agents = (
                f"{r['avg_greedy_agents']:.2f}"
                if r["avg_greedy_agents"] is not None
                else "N/A"
            )
            greedy_gain = (
                f"{r['avg_greedy_gain']:+.4f}"
                if r["avg_greedy_gain"] is not None
                else "N/A"
            )
            print(
                f"  budget={r['budget']:>6} | score={score_str:>10} | "
                f"greedy_agents_moy={greedy_agents:>5} | greedy_gain_moy={greedy_gain}"
            )


if __name__ == "__main__":
    main()
