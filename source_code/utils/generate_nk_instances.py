"""Generate frozen NK / NK3 instance files for a new problem size.

Replicates exactly the on-the-fly random generation logic already present in
`environment.nk.getTensorInstances_NK` (the `path == ""` branch), but writes
the result to disk in the same text format as the existing instances under
`source_code/instances/nk/{dim}/{K}/nk_{dim}_{K}_{i}.txt` (D=2, NK) and
`source_code/instances/nk3/{dim}/{K}/nk_{dim}_{K}_3_{i}.txt` (D=3, NK3), so
they are frozen/reusable across every competing algorithm instead of being
resampled on every run.

Usage:
    python generate_nk_instances.py --dim 512 --k 1 2 4 8 --nb-instances 10 --seed 42
"""

import argparse
import os

import numpy as np


def generate_instance(N: int, K: int, D: int, rng: np.random.Generator):
    """Same logic as environment.nk.getTensorInstances_NK's path=='' branch."""
    matrix_locus = np.zeros((N, K + 1), dtype=np.int64)
    for x in range(N):
        neigh = [x]
        for _ in range(K):
            x1 = int(rng.integers(0, N))
            while x1 in neigh:
                x1 = int(rng.integers(0, N))
            neigh.append(x1)
        neigh.sort()
        matrix_locus[x, :] = neigh

    matrix_contrib = rng.random((N, D ** (K + 1))).astype(np.float32)

    return matrix_locus, matrix_contrib


def write_instance(path: str, N: int, K: int, D: int, matrix_locus, matrix_contrib):
    with open(path, "w") as f:
        if D > 2:
            f.write(f"{N} {K} {D}\n")
        else:
            f.write(f"{N} {K}\n")
        for x in range(N):
            for val in matrix_locus[x]:
                f.write(f"{int(val)}\n")
        for x in range(N):
            for val in matrix_contrib[x]:
                f.write(f"{float(val)}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate frozen NK/NK3 instances.")
    parser.add_argument("--dim", type=int, required=True, help="N (number of variables)")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8], help="K values (epistatic neighborhood sizes)")
    parser.add_argument("--nb-instances", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--problems", nargs="+", default=["nk", "nk3"], choices=["nk", "nk3"])
    parser.add_argument("--out-root", default=None, help="defaults to source_code/instances")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    instances_root = args.out_root or os.path.join(script_dir, "..", "instances")

    problem_D = {"nk": 2, "nk3": 3}

    for problem in args.problems:
        D = problem_D[problem]
        for K in args.k:
            out_dir = os.path.join(instances_root, problem, str(args.dim), str(K))
            os.makedirs(out_dir, exist_ok=True)
            for i in range(args.nb_instances):
                # deterministic, distinct seed per (problem, K, i) for reproducibility
                seed = args.seed + hash((problem, args.dim, K, i)) % (2**31 - 1)
                rng = np.random.default_rng(seed)
                matrix_locus, matrix_contrib = generate_instance(args.dim, K, D, rng)

                if D > 2:
                    fname = f"nk_{args.dim}_{K}_{D}_{i}.txt"
                else:
                    fname = f"nk_{args.dim}_{K}_{i}.txt"

                write_instance(os.path.join(out_dir, fname), args.dim, K, D, matrix_locus, matrix_contrib)
            print(f"[{problem}] dim={args.dim} K={K}: {args.nb_instances} instances written to {out_dir}")


if __name__ == "__main__":
    main()
