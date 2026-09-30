"""Comprehensive completeness check for the QUBO n=512 campaign across all
algorithm families: nevergrad (81 algos minus DiscreteNoisyInfSplits, excluded
and documented in paper/outline.md), PBIL/MIMIC/BOA, and PEDA/PPBIL.

Usage (from repo root, e.g. on Jean Zay after `git pull`):
    python jeanzay/check_qubo512.py
"""
import os
import re

ROOT = "results/nevergrad"
NB_T = 6
NB_SEEDS = 10


def is_complete(path):
    try:
        with open(path) as f:
            return sum(1 for _ in f) > 1
    except OSError:
        return False


def check_family(algos, fname_prefix, nb_instances_token):
    missing = []
    for algo in algos:
        for t in range(NB_T):
            d = os.path.join(ROOT, algo, "QUBO", "512", str(t))
            seeds_found = set()
            if os.path.isdir(d):
                pat = re.compile(
                    rf"^{fname_prefix}_{re.escape(algo)}_QUBO_512_{t}_{nb_instances_token}_budget_50000_.+_(\d+)\.txt$"
                )
                for f in os.listdir(d):
                    m = pat.match(f)
                    if m and is_complete(os.path.join(d, f)):
                        seeds_found.add(int(m.group(1)))
            for seed in range(NB_SEEDS):
                if seed not in seeds_found:
                    missing.append((algo, t, seed))
    return missing


def main():
    print("=== 1. Nevergrad (81 algos, DiscreteNoisyInfSplits exclu) ===")
    algos = [l.strip() for l in open("jeanzay/nevergrad_discrete_algos.txt") if l.strip()]
    algos = [a for a in algos if a != "DiscreteNoisyInfSplits"]
    missing_ng = check_family(algos, "results_nevergrad", "10")
    expected_ng = len(algos) * NB_T * NB_SEEDS
    print(f"  {expected_ng - len(missing_ng)}/{expected_ng} complets, {len(missing_ng)} manquants")
    if missing_ng[:10]:
        print("  exemples manquants:", missing_ng[:10])

    print()
    print("=== 2. PBIL / MIMIC / BOA ===")
    missing_edas = check_family(["PBIL", "MIMIC", "BOA"], "results_EDAs_final", "10")
    expected_edas = 3 * NB_T * NB_SEEDS
    print(f"  {expected_edas - len(missing_edas)}/{expected_edas} complets, {len(missing_edas)} manquants")
    if missing_edas[:10]:
        print("  exemples manquants:", missing_edas[:10])

    print()
    print("=== 3. PEDA / PPBIL ===")
    for algo in ["PEDA", "PPBIL"]:
        d_base = os.path.join(ROOT, algo, "QUBO", "512")
        total = 0
        complete = 0
        for t in range(NB_T):
            d = os.path.join(d_base, str(t))
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".txt"):
                    total += 1
                    if is_complete(os.path.join(d, f)):
                        complete += 1
        print(f"  {algo}: {complete}/600 fichiers complets (total trouve: {total})")

    print()
    print("=== RESUME ===")
    if not missing_ng and not missing_edas:
        print("QUBO n=512 : COMPLET (sauf DiscreteNoisyInfSplits, exclu et documente)")
    else:
        print(
            "QUBO n=512 : INCOMPLET -",
            len(missing_ng),
            "nevergrad +",
            len(missing_edas),
            "PBIL/MIMIC/BOA manquants",
        )


if __name__ == "__main__":
    main()
