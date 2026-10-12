"""Temps de calcul exacts des configs SVGD-EDA (PPSN) et SE-SVGD-EDA (EvoCOP) sur les 56 classes.

Reprend exactement le code des expériences (main_expe_overall._run_once, PPO compris) : une classe =
10 instances × 10 restarts lancés en un seul lot sur le GPU, budget 50 000 évaluations par run.
Chaque classe est chronométrée avec synchronisation CUDA, après un run de chauffe (initialisation CUDA).
N'écrit rien dans results/config ; sorties dans results/timing/ (CSV + résumé Markdown avec le matériel).

    python source_code/expe/main_timing.py -g jeanzay/grids/timing.json [--only QUBO_dim64_t0 ...] [--repeats 1]
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXPE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPE_DIR.parent)); sys.path.insert(0, str(EXPE_DIR))
import main_expe_overall as m   # noqa: E402

REPO_ROOT = EXPE_DIR.parents[1]


def _hardware() -> dict:
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?") \
        if Path("/proc/cpuinfo").exists() else platform.processor()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "aucun (CPU)"
    mem = f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} Go" if torch.cuda.is_available() else "—"
    return {"noeud": socket.gethostname(), "job Slurm": os.environ.get("SLURM_JOB_ID", "—"), "GPU": gpu,
            "mémoire GPU": mem, "CPU": cpu, "CPU alloués": os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count())),
            "torch": torch.__version__, "CUDA": torch.version.cuda or "—", "python": platform.python_version()}


def _instances() -> list[dict]:
    root = REPO_ROOT / "source_code" / "instances"
    n = m.DEFAULTS["nb_instances_test"]
    return (m._discover_qubo_instances(root / "QUBO", n) + m._discover_nk_instances(root / "nk", n)
            + m._discover_nk3_instances(root / "nk3", n))


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_once(inst: dict, p: dict) -> float:
    ctx = m._load_instances(inst, m.DEFAULTS["device"])          # chargement des instances hors chrono
    _sync(); t0 = time.perf_counter()
    m._run_once(ctx, p["kernel"], p["advantage"], p["M"], p["lambda_"], p["epsilon_svgd"], p["gamma"],
                p["decay_start_ratio"], p["decay_min_factor"], device=m.DEFAULTS["device"],
                nb_restarts=m.DEFAULTS["nb_restarts"], ppo_params=p)
    _sync()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-g", "--grid", default=str(REPO_ROOT / "jeanzay" / "grids" / "timing.json"))
    ap.add_argument("--only", nargs="*", default=None, help="Classes à chronométrer (défaut : les 56)")
    ap.add_argument("--repeats", type=int, default=1, help="Répétitions par classe (on garde la médiane)")
    args = ap.parse_args()

    hw = _hardware()
    print("=== Matériel ==="); [print(f"  {k}: {v}") for k, v in hw.items()]
    configs = [(name, p) for g in m._load_grids(args.grid) for name, p in m._expand_grid(g)]
    insts = [i for i in _instances() if not args.only
             or f"{i['name']}_dim{i['dim']}_t{i['type_instance']}" in args.only]
    print(f"=== {len(configs)} configs × {len(insts)} classes, {args.repeats} répétition(s) ===")
    m._set_seeds(m.DEFAULTS["seed"])

    # Chauffe : initialisation CUDA / compilation des noyaux, non comptée
    print("chauffe…", flush=True); _time_once(insts[0], configs[0][1])

    out_dir = REPO_ROOT / "results" / "timing"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = hw["job Slurm"] if hw["job Slurm"] != "—" else time.strftime("%Y%m%d_%H%M%S")
    rows = []
    for cfg, p in configs:
        print(f"\n[CONFIG] {cfg}", flush=True)
        for inst in insts:
            name = f"{inst['name']}_dim{inst['dim']}_t{inst['type_instance']}"
            ts = [_time_once(inst, p) for _ in range(args.repeats)]
            t = float(np.median(ts))
            rows.append({"config": cfg, "classe": name, "probleme": inst["name"], "n": inst["dim"],
                         "temps_s": round(t, 3), "repetitions": " ".join(f"{x:.3f}" for x in ts)})
            print(f"  {name:18s} {t:9.2f} s", flush=True)

    csv_path = out_dir / f"timing_{tag}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    md = ["# Temps de calcul", "", "## Matériel", ""] + [f"- **{k}** : {v}" for k, v in hw.items()] + [
        "", "## Protocole", "",
        "- Une classe = 10 instances × 10 restarts exécutés en un seul lot sur le GPU, 50 000 évaluations par run.",
        "- Chrono : `time.perf_counter()` avec `torch.cuda.synchronize()` avant et après ; chargement des "
        "instances exclu ; un run de chauffe non compté.",
        f"- Répétitions par classe : {args.repeats} (médiane).", "", "## Résultats", ""]
    for cfg, _ in configs:
        sub = [r for r in rows if r["config"] == cfg]
        tot = sum(r["temps_s"] for r in sub)
        md += [f"### `{cfg}`", "", f"- Total : **{tot:.1f} s** ({tot / 60:.1f} min) pour {len(sub)} classes ; "
               f"moyenne {tot / len(sub):.2f} s par classe (100 runs), soit {tot / len(sub) / 100 * 1000:.0f} ms "
               "par run en équivalent séquentiel.", "", "| problème | n | temps moyen par classe (s) |", "|---|---|---|"]
        for pb in ("QUBO", "NK", "NK3"):
            for n in sorted({r["n"] for r in sub if r["probleme"] == pb}):
                v = [r["temps_s"] for r in sub if r["probleme"] == pb and r["n"] == n]
                md.append(f"| {pb} | {n} | {np.mean(v):.2f} |")
        md.append("")
    (out_dir / f"timing_{tag}.md").write_text("\n".join(md) + "\n")
    print(f"\nÉcrit : {csv_path} et {csv_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
