"""Write the hyperparameter search grids run on the tuning instances.

Two modes, same number of configs and same sampler, number of agents M in 1..25:
  - ppsn   : global rank, one update per batch (no proximal step), temperature schedule;
  - evocop : per-agent rank, multi-epoch update with KL penalty (E epochs, beta).
Each grid holds the paper configuration followed by N_SAMPLED configs drawn by Latin
hypercube sampling over wide ranges (log scale for the continuous parameters), so that
the search covers the space instead of a few values per axis. Values are rounded to
3 significant digits to keep readable config names.

Output: jeanzay/grids/tuning_ppsn.json and jeanzay/grids/tuning_evocop.json, in the
"configs" format read by expe/main_expe_overall.py.

Usage:
    python source_code/utils/make_tuning_grids.py
"""

import json
import math
import os

import numpy as np
from scipy.stats import qmc

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "jeanzay", "grids")
N_SAMPLED = 49          # + 1 paper config = 50 configs per mode
SEED = 2027

AGENTS = list(range(1, 26))

LAMBDAS = [6, 8, 10, 13, 16, 20, 25]
EPOCHS = list(range(2, 13))
P_CONSTANT_GAMMA = 0.25  # fraction of configs without temperature schedule (ds=1)

# (low, high), sampled log-uniformly
RANGES = dict(
    epsilon_svgd=(0.01, 0.5),
    gamma=(0.001, 0.1),
    decay_start_ratio=(0.005, 0.5),
    decay_min_factor=(0.001, 0.5),
    kl_beta=(0.003, 3.0),
)

PPSN_PAPER = dict(
    kernel="rbf", advantage="globalrankweighted", M=7, **{"lambda": 13},
    epsilon_svgd=0.08, gamma=0.015, decay_start_ratio=0.03, decay_min_factor=0.01,
)
EVOCOP_PAPER = dict(
    kernel="rbf", advantage="peragentrankweighted", M=7, **{"lambda": 13},
    epsilon_svgd=0.06, gamma=0.01, decay_start_ratio=1.0, decay_min_factor=0.01,
    ppo_active=True, ppo_epochs=6, kl_beta=0.1,
)


def _sig3(x: float) -> float:
    return float(f"{x:.3g}")


def _log_scale(u: float, low: float, high: float) -> float:
    return _sig3(math.exp(math.log(low) + u * (math.log(high) - math.log(low))))


def _pick(u: float, values: list):
    return values[min(int(u * len(values)), len(values) - 1)]


def _sample(mode: str, n: int, seed: int) -> list:
    # Dimensions: M, lambda, eps, gamma, constant-or-schedule, ds, dm [, E, beta]
    d = 7 if mode == "ppsn" else 9
    U = qmc.LatinHypercube(d=d, seed=seed).random(n)
    configs = []
    for u in U:
        constant = u[4] < P_CONSTANT_GAMMA
        cfg = dict(
            kernel="rbf",
            advantage="globalrankweighted" if mode == "ppsn" else "peragentrankweighted",
            M=_pick(u[0], AGENTS),
            **{"lambda": _pick(u[1], LAMBDAS)},
            epsilon_svgd=_log_scale(u[2], *RANGES["epsilon_svgd"]),
            gamma=_log_scale(u[3], *RANGES["gamma"]),
            decay_start_ratio=1.0 if constant else _log_scale(u[5], *RANGES["decay_start_ratio"]),
            decay_min_factor=0.01 if constant else _log_scale(u[6], *RANGES["decay_min_factor"]),
        )
        if mode == "evocop":
            cfg.update(
                ppo_active=True,
                ppo_epochs=_pick(u[7], EPOCHS),
                kl_beta=_log_scale(u[8], *RANGES["kl_beta"]),
            )
        configs.append(cfg)
    return configs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for mode, paper, seed in (("ppsn", PPSN_PAPER, SEED), ("evocop", EVOCOP_PAPER, SEED + 1)):
        configs = [paper] + _sample(mode, N_SAMPLED, seed)
        path = os.path.join(OUT_DIR, f"tuning_{mode}.json")
        with open(path, "w") as f:
            json.dump([{"configs": configs}], f, indent=1)
        print(f"{path}: {len(configs)} configs")


if __name__ == "__main__":
    main()
