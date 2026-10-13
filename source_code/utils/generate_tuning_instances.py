"""Generate the tuning instances, disjoint from the test instances of source_code/instances.

Same 56 classes as the test benchmark (QUBO types 0-5, NK and NK3 with K in {1,2,4,8},
n in {64,128,256,512}), NB_INSTANCES instances per class, written to
source_code/instances_tuning/{QUBO,nk,nk3} with the same file names and formats, so that
the existing loaders read them unchanged.

QUBO: PUBOi generator (Tari et al., 2022) with the parameters of the test instances,
which use seeds 0-9 (i_1 to i_10); the tuning instances use seeds 1000+ (i_1 to i_3).
With seed 0 these parameters reproduce the test instances exactly.
NK / NK3: same generator as generate_nk_instances.py, with a deterministic seed per
(D, n, K, i) under a base distinct from the test instances.

Usage:
    python source_code/utils/generate_tuning_instances.py
"""

import argparse
import os
import sys

import numpy as np

SOURCE_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SOURCE_CODE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_CODE_DIR)

from utils.generate_nk_instances import generate_instance, write_instance
from utils.puboi_generator import PUBOi_generator, PortfolioBuilder

DIMS = [64, 128, 256, 512]
NK_KS = [1, 2, 4, 8]
QUBO_TYPES = range(6)
NB_INSTANCES = 3
QUBO_SEED_BASE = 1000
NK_SEED_BASE = 2027

# Number of terms m of the test instances, per n: (types 0-2, types 3-5).
QUBO_M = {64: (101, 403), 128: (406, 1626), 256: (1632, 6528), 512: (6541, 26163)}


def make_qubo(n: int, t: int, seed: int) -> PUBOi_generator:
    importance = {
        "size": [n // 4, 3 * n // 4],
        "degree": [1.0, 1.0] if t in (0, 3) else [8.0, 1.0],
    }
    factor = 1.1125 if t in (2, 5) else 1.0
    m = QUBO_M[n][0 if t < 3 else 1]
    return PUBOi_generator(m, importance, factor, PortfolioBuilder(n).make(), [1, 1, 1, 1], 0, True, seed)


def main():
    parser = argparse.ArgumentParser(description="Generate the tuning instances.")
    parser.add_argument("--out-root", default=os.path.join(SOURCE_CODE_DIR, "instances_tuning"))
    parser.add_argument("--nb-instances", type=int, default=NB_INSTANCES)
    args = parser.parse_args()

    qubo_dir = os.path.join(args.out_root, "QUBO")
    os.makedirs(qubo_dir, exist_ok=True)
    for n in DIMS:
        for t in QUBO_TYPES:
            for i in range(1, args.nb_instances + 1):
                gen = make_qubo(n, t, QUBO_SEED_BASE + i - 1)
                f = gen.make()
                f.to_json(os.path.join(qubo_dir, f"puboi_evo_n_{n}_t_{t}_i_{i}.json"), gen)
            print(f"[QUBO] n={n} t={t}: {args.nb_instances} instances")

    for problem, D in (("nk", 2), ("nk3", 3)):
        for n in DIMS:
            for K in NK_KS:
                out_dir = os.path.join(args.out_root, problem, str(n), str(K))
                os.makedirs(out_dir, exist_ok=True)
                for i in range(args.nb_instances):
                    rng = np.random.default_rng([NK_SEED_BASE, D, n, K, i])
                    locus, contrib = generate_instance(n, K, D, rng)
                    fname = f"nk_{n}_{K}_{D}_{i}.txt" if D > 2 else f"nk_{n}_{K}_{i}.txt"
                    write_instance(os.path.join(out_dir, fname), n, K, D, locus, contrib)
                print(f"[{problem}] n={n} K={K}: {args.nb_instances} instances")


if __name__ == "__main__":
    main()
