"""List which (algo, t, seed) tasks of the n=512 nevergrad campaign are still missing,
for QUBO, NK or NK3, and print the matching Slurm --array spec.

Same task encoding as jeanzay/expe_nevergrad_512.slurm:
    task_id = ALGO_IDX * (NB_T * 10) + T_IDX * 10 + SEED
with NB_T = 6 for QUBO (t = 0..5) and NB_T = 4 for NK/NK3 (K = 1, 2, 4, 8).

A task counts as done if a result file for it has more than its header line.
DiscreteNoisyInfSplits is ignored (excluded at n=512). For NK3, K=8 tasks are
printed separately: they need --cpus-per-task=4 (OOM with the default 1 CPU).

Usage (from repo root, after `git pull`):
    python3 jeanzay/compute_missing_tasks.py -p NK3
"""
import argparse
import os
import re
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGO_FILE = os.path.join(REPO_ROOT, "jeanzay", "nevergrad_discrete_algos.txt")
RESULTS_ROOT = os.path.join(REPO_ROOT, "results", "nevergrad")
EXCLUDED = {"DiscreteNoisyInfSplits"}
NB_SEEDS = 10


def compress(ids):
    if not ids:
        return ""
    parts = []
    start = prev = ids[0]
    for x in ids[1:]:
        if x == prev + 1:
            prev = x
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = x
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--problem", default="QUBO", choices=["QUBO", "NK", "NK3"])
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--budget", type=int, default=50000)
    parser.add_argument("--nb-instances", type=int, default=10)
    args = parser.parse_args()

    t_values = list(range(6)) if args.problem == "QUBO" else [1, 2, 4, 8]
    nb_t = len(t_values)

    with open(ALGO_FILE) as f:
        algos = [line.strip() for line in f if line.strip()]

    fname_re = re.compile(
        rf"^results_nevergrad_(?P<algo>.+)_{args.problem}_{args.dim}_(?P<t>\d+)_{args.nb_instances}"
        rf"_budget_{args.budget}_.+_(?P<seed>\d+)\.txt$"
    )

    done = set()
    for a, algo in enumerate(algos):
        for t_idx, t in enumerate(t_values):
            d = os.path.join(RESULTS_ROOT, algo, args.problem, str(args.dim), str(t))
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                m = fname_re.match(fname)
                if not m or m.group("algo") != algo or int(m.group("t")) != t:
                    continue
                try:
                    with open(os.path.join(d, fname)) as fh:
                        if sum(1 for _ in fh) <= 1:
                            continue
                except OSError:
                    continue
                done.add(a * nb_t * NB_SEEDS + t_idx * NB_SEEDS + int(m.group("seed")))

    expected = [
        a * nb_t * NB_SEEDS + t_idx * NB_SEEDS + s
        for a, algo in enumerate(algos) if algo not in EXCLUDED
        for t_idx in range(nb_t)
        for s in range(NB_SEEDS)
    ]
    missing = [i for i in expected if i not in done]

    print(f"# {args.problem} n={args.dim}: attendues={len(expected)} faites={len(expected) - len(missing)} manquantes={len(missing)}")
    by_t = Counter(t_values[(i // NB_SEEDS) % nb_t] for i in missing)
    print("# manquantes par t/K : " + ", ".join(f"{t}={by_t.get(t, 0)}" for t in t_values))
    by_algo = Counter(algos[i // (nb_t * NB_SEEDS)] for i in missing)
    for algo, n in by_algo.most_common(10):
        print(f"#   {algo}: {n}")

    if args.problem == "NK3":
        k8 = [i for i in missing if t_values[(i // NB_SEEDS) % nb_t] == 8]
        rest = [i for i in missing if t_values[(i // NB_SEEDS) % nb_t] != 8]
        print("# --- K=1,2,4 (memoire par defaut) ---")
        print(compress(rest))
        print("# --- K=8 (a lancer avec --cpus-per-task=4) ---")
        print(compress(k8))
    else:
        print(compress(missing))


if __name__ == "__main__":
    main()
