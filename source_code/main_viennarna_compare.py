"""
Compare SVGD-EDA vs a Nevergrad baseline on a ViennaRNA target.

Reads all parameters from config_viennarna.yaml (problem + agent sections).
Appends one row per run to results/viennarna_comparison.csv:
  target_name, budget, score_svgd, nevergrad_algo, score_nevergrad, gap, time_svgd_s, time_nevergrad_s
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from main_viennarna import (
    _load_kernel_config,
    get_Score_trajectories_viennarna_cuda,
)
from main_nevergrad_viennarna import _run_restart
from problems.viennarna import (
    ETERNA100_TSV_URL,
    load_target_from_eterna100,
    normalize_target_struct,
)

try:
    import RNA
except Exception:
    RNA = None

try:
    import nevergrad as ng
except Exception:
    ng = None


# ─── Instances à comparer ───────────────────────────────────────────────────
TARGETS = [
    "Simple Hairpin",
    "Kudzu","Chicken feet",
    "Mat - Martian 2", "Fractal 2","Still Life (Sunflower In A Vase)","Quasispecies 2-2 Loop Challenge","Simple Single Bond","Pokeball","Anemone"
]
# ────────────────────────────────────────────────────────────────────────────

_DEFAULT_TARGET_STRUCT = "(" * 25 + "." * 50 + ")" * 25

CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "viennarna_comparison.csv"
CSV_COLUMNS = [
    "target_name",
    "budget",
    "score_svgd",
    "nevergrad_algo",
    "score_nevergrad",
    "gap",
    "time_svgd_s",
    "time_nevergrad_s",
]


def _append_row(row: dict) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    replaced = False
    if CSV_PATH.exists():
        with CSV_PATH.open(newline="") as f:
            reader = csv.DictReader(f)
            for existing in reader:
                if (existing["target_name"] == row["target_name"]
                        and existing["nevergrad_algo"] == row["nevergrad_algo"]):
                    rows.append(row)
                    replaced = True
                else:
                    rows.append(existing)
    if not replaced:
        rows.append(row)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _run_one_target(
    target_name: str,
    target_source: str,
    cfg: DictConfig,
    agent_val,
    algo_name: str,
    budget: int,
    seed: int,
    nb_instances: int,
    nb_restarts: int,
    verbose: bool,
    num_workers: int,
    lambda_: int,
    M: int,
    kernel_cfg: dict,
    epsilon_svgd: float,
    svgd_gamma: float,
    advantage_cfg,
    decay_enabled: bool,
    decay_start_ratio: float,
    decay_min_factor: float,
    device: str,
) -> None:
    target_struct_cfg = OmegaConf.select(cfg, "problem.target_struct")
    if target_struct_cfg:
        target_struct = normalize_target_struct(str(target_struct_cfg))
        target_resolved_name = "cfg_target_struct"
    else:
        target_struct, target_resolved_name = load_target_from_eterna100(
            target_name=target_name,
            source=target_source,
            fallback_target=_DEFAULT_TARGET_STRUCT,
            verbose=verbose,
        )

    dim = len(target_struct)
    print(f"\n{'='*50}")
    print(f"Target: {target_resolved_name} (len={dim}) | budget={budget} | seed={seed}")
    print(f"SVGD: nb_instances={nb_instances} | Nevergrad: {algo_name} nb_restarts={nb_restarts}")

    # ------------------------------------------------------------------ SVGD
    strategy = FactoryStrategyEA().createStrategyEA(
        "PPO-EDA",
        dim,
        lambda_,
        device,
        [4] * dim,
        M,
        learning_rate=epsilon_svgd,
        epsilon_svgd=epsilon_svgd,
        enable_visualization=False,
        svgd_gamma=svgd_gamma,
        decay_start_ratio=decay_start_ratio,
        decay_min_factor=decay_min_factor,
        decay_enabled=decay_enabled,
        advantage_cfg=advantage_cfg,
        kernel_config=kernel_cfg,
        no_interact=bool(agent_val("no_interact") or False),
        no_repulsion=bool(agent_val("no_repulsion") or False),
        is_nk3=False,
    ).to(device)

    print(f"\n[1/2] SVGD-EDA  M={M} lambda={lambda_} ...")
    t0 = time.perf_counter()
    scores_svgd = get_Score_trajectories_viennarna_cuda(
        target_struct=target_struct,
        strategy=strategy,
        nb_instances=nb_instances,
        budget=budget,
        size_popEA=lambda_,
        device=device,
        verbose=verbose,
        num_workers=num_workers,
        enable_visualization=False,
        name_file=None,
        return_history=False,
    )
    time_svgd = time.perf_counter() - t0
    score_svgd = float(np.mean(scores_svgd))
    print(f"    score={score_svgd:.4f}  time={time_svgd:.1f}s")

    # --------------------------------------------------------------- Nevergrad
    print(f"\n[2/2] {algo_name}  nb_restarts={nb_restarts} ...")
    t0 = time.perf_counter()
    ng_scores = []
    with tqdm(total=nb_restarts * budget, unit="eval", desc=algo_name, disable=not verbose) as pbar:
        for i in range(nb_restarts):
            best, _ = _run_restart(
                target_struct=target_struct,
                dim=dim,
                budget=budget,
                step_record=100,
                algo_name=algo_name,
                seed=seed + i,
                progress_bar=pbar,
            )
            ng_scores.append(best)
    time_ng = time.perf_counter() - t0
    score_ng = float(np.mean(ng_scores))
    print(f"    score={score_ng:.4f}  time={time_ng:.1f}s")

    # ------------------------------------------------------------------ CSV
    gap = round(score_svgd - score_ng, 6)
    row = {
        "target_name": target_resolved_name,
        "budget": budget,
        "score_svgd": round(score_svgd, 6),
        "nevergrad_algo": algo_name,
        "score_nevergrad": round(score_ng, 6),
        "gap": gap,
        "time_svgd_s": round(time_svgd, 1),
        "time_nevergrad_s": round(time_ng, 1),
    }
    _append_row(row)

    winner = "SVGD" if gap > 0 else algo_name
    print(f"SVGD         : {score_svgd:.4f}  ({time_svgd:.1f}s)")
    print(f"{algo_name:<13}: {score_ng:.4f}  ({time_ng:.1f}s)")
    print(f"Gap          : {gap:+.6f}  ({winner} wins)")
    print(f"Saved to     : {CSV_PATH}")


@hydra.main(config_path="../config", config_name="config_viennarna", version_base=None)
def main(cfg: DictConfig) -> None:
    if RNA is None:
        raise RuntimeError("Install ViennaRNA Python bindings.")
    if ng is None:
        raise RuntimeError("Install nevergrad.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    def agent_val(key):
        try:
            return OmegaConf.select(cfg, f"agent.{key}")
        except Exception:
            return None

    target_source = str(OmegaConf.select(cfg, "problem.target_source") or ETERNA100_TSV_URL)
    budget = int(cfg.get("budget", 100000))
    seed = int(cfg.seed)
    nb_instances = int(cfg.nb_instances_test)
    nb_restarts = int(cfg.get("nb_restarts", 1))
    verbose = bool(cfg.get("verbose", True))

    algo_name = str(
        OmegaConf.select(cfg, "problem.nevergrad_algo")
        or cfg.get("nevergrad_algo")
        or "DiscreteDE"
    )

    num_workers_cfg = OmegaConf.select(cfg, "problem.num_workers") or cfg.get("num_workers")
    num_workers = max(1, int(num_workers_cfg)) if num_workers_cfg is not None else max(1, mp.cpu_count() - 1)

    # Resolve SVGD agent params once (dim-independent)
    lambda_ = int(agent_val("lambda") or cfg.get("lambda") or cfg.get("lambda_") or 10)
    M = int(agent_val("M") or cfg.get("M") or 1)
    kernel_name = str(agent_val("kernel") or cfg.get("kernel") or "rbf").lower()
    kernel_cfg = _load_kernel_config(kernel_name, repo_root)
    bandwith_override = agent_val("bandwith_kernel") or cfg.get("bandwith_kernel")
    if bandwith_override is not None:
        kernel_cfg["bandwith_kernel"] = bandwith_override
    kernel_cfg["debug_svgd"] = False
    epsilon_svgd = float(agent_val("epsilon_svgd") or cfg.get("epsilon_svgd") or kernel_cfg.get("epsilon_svgd") or 0.1)
    svgd_gamma = float(agent_val("gamma") or cfg.get("gamma") or kernel_cfg.get("gamma") or 0.01)
    advantage_cfg = agent_val("advantage") or cfg.get("advantage") or "globalrankweighted"
    if isinstance(advantage_cfg, DictConfig):
        advantage_cfg = OmegaConf.to_container(advantage_cfg, resolve=True)
    decay_enabled = bool(agent_val("decay") or cfg.get("decay") or False)
    decay_start_ratio = float(agent_val("decay_start_ratio") or cfg.get("decay_start_ratio") or (0.0 if decay_enabled else 0.8))
    decay_min_factor = float(
        agent_val("min_factor") or agent_val("decay_min_factor")
        or cfg.get("min_factor") or cfg.get("decay_min_factor")
        or (0.05 if decay_enabled else 0.1)
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"Running {len(TARGETS)} targets | budget={budget} | algo={algo_name}")

    for target_name in TARGETS:
        _run_one_target(
            target_name=target_name,
            target_source=target_source,
            cfg=cfg,
            agent_val=agent_val,
            algo_name=algo_name,
            budget=budget,
            seed=seed,
            nb_instances=nb_instances,
            nb_restarts=nb_restarts,
            verbose=verbose,
            num_workers=num_workers,
            lambda_=lambda_,
            M=M,
            kernel_cfg=kernel_cfg,
            epsilon_svgd=epsilon_svgd,
            svgd_gamma=svgd_gamma,
            advantage_cfg=advantage_cfg,
            decay_enabled=decay_enabled,
            decay_start_ratio=decay_start_ratio,
            decay_min_factor=decay_min_factor,
            device=device,
        )


if __name__ == "__main__":
    main()
