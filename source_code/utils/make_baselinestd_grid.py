"""Write the grid testing the (f(x) - b_i^t) / std^t advantage (peragentbaselineglobalstd).

Same sampler as the evocop mode of make_tuning_grids.py (M, lambda, epsilon,
gamma, epochs E, KL beta drawn by Latin hypercube), with M in 5..20, lambda <= 20,
M * lambda <= 250 and ranges narrowed around EvoCOP, with the new advantage and no
temperature schedule (ds=1). The first config is EvoCOP with the advantage swapped.
Run on the test instances (results/config), not the tuning ones.

Output: jeanzay/grids/baselinestd_evocop.json

Usage:
    python source_code/utils/make_baselinestd_grid.py
"""

import json
import os

from scipy.stats import qmc

from make_tuning_grids import EPOCHS, EVOCOP_PAPER, OUT_DIR, RANGES, _log_scale, _pick

ADVANTAGE = "peragentbaselineglobalstd"
N_SAMPLED = 49          # + EvoCOP = 50 configs
SEED = 2029

AGENTS = list(range(5, 21))
LAMBDAS = [8, 10, 13, 16, 20]
MAX_POP = 250           # M * lambda: above, too few iterations within the 50k evaluations budget
# Scaling the advantage by c ~ eps*c, gamma/c, beta/c; the new advantage is ~1.5x the
# per-agent rank, so ranges are centred near EvoCOP (eps 0.04, gamma 0.015, beta 0.15), x4 each side
RANGES = dict(RANGES, epsilon_svgd=(0.01, 0.15), gamma=(0.003, 0.06), kl_beta=(0.03, 1.0))


def _sample(n: int, seed: int, oversample: int = 1) -> list:
    # Dimensions: M, lambda, eps, gamma, E, beta
    configs = []
    for u in qmc.LatinHypercube(d=6, seed=seed).random(n * oversample):
        if _pick(u[0], AGENTS) * _pick(u[1], LAMBDAS) > MAX_POP:
            continue
        configs.append(dict(
            kernel="rbf", advantage=ADVANTAGE,
            M=_pick(u[0], AGENTS),
            **{"lambda": _pick(u[1], LAMBDAS)},
            epsilon_svgd=_log_scale(u[2], *RANGES["epsilon_svgd"]),
            gamma=_log_scale(u[3], *RANGES["gamma"]),
            decay_start_ratio=1.0, decay_min_factor=0.01,
            ppo_active=True,
            ppo_epochs=_pick(u[4], EPOCHS),
            kl_beta=_log_scale(u[5], *RANGES["kl_beta"]),
        ))
        if len(configs) == n:
            break
    return configs


def main():
    configs = [dict(EVOCOP_PAPER, advantage=ADVANTAGE)] + _sample(N_SAMPLED, SEED, oversample=3)
    path = os.path.join(OUT_DIR, "baselinestd_evocop.json")
    with open(path, "w") as f:
        json.dump([{"configs": configs}], f, indent=1)
    print(f"{path}: {len(configs)} configs")


if __name__ == "__main__":
    main()
