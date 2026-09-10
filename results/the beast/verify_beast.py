"""
Verifie, pour CHAQUE instance rangee dans "the beast/", que la colonne fitness
correspond bien au genotype de la meme ligne — en recalculant la fitness avec les
evaluateurs du depot (environment/nk.py, utils/walsh_expansion.py + formule qubo.py).

Arbo attendue : the beast/<probleme>/<N>/<K|t>/<instance>/{fitness.csv, solutions.csv}

Conventions des colonnes fitness :
    NK / NK3 : sco/N            (maximisation ; ~0.7)
    QUBO     : -(s^T Q s), s=2x-1   (cf. _evaluate_population de qubo.py)

Usage :
    python verify_beast.py               # toutes les instances remplies
    python verify_beast.py nk/64/2       # filtre par prefixe de chemin
"""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "source_code")
sys.path.insert(0, SRC)

from environment.nk import problem_NKlandscape           # noqa: E402
from utils.walsh_expansion import WalshExpansion          # noqa: E402

BEAST = os.path.dirname(os.path.abspath(__file__))

# Les .txt NK stockent les contributions a 6 decimales : sur une somme de (K+1)*N
# termes arrondis puis /N, l'ecart au CSV (calcule en pleine precision) peut monter
# a ~1e-5 pour K=8. QUBO a des poids entiers -> exact.
TOL = {"nk": 1e-5, "nk3": 1e-5, "qubo": 1e-9}


def instance_path(problem, N, Kt, stem):
    if problem == "qubo":
        return os.path.join(SRC, "instances", "QUBO", stem + ".json")
    sub = "nk3" if problem == "nk3" else "nk"
    return os.path.join(SRC, "instances", sub, N, Kt, stem + ".txt")


def recompute(problem, inst, sols):
    if problem in ("nk", "nk3"):
        prob = problem_NKlandscape(inst)
        assert sols.shape[1] == prob.N, (sols.shape, prob.N)
        return np.array([-prob.eval(r.astype(int)) for r in sols]), f"N={prob.N} K={prob.K} D={prob.D}"
    we = WalshExpansion()
    we.load(inst)
    assert sols.shape[1] == we.n, (sols.shape, we.n)
    Q = we.to_symmetric_Q()
    S = sols * 2.0 - 1.0
    return -np.einsum("ij,jk,ik->i", S, Q, S), f"N={we.n}"


def _subdirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e)))


def find_instances(flt):
    for problem in ("nk", "nk3", "qubo"):
        root = os.path.join(BEAST, problem)
        for N in _subdirs(root):
            for Kt in _subdirs(os.path.join(root, N)):
                base = os.path.join(root, N, Kt)
                for stem in _subdirs(base):
                    d = os.path.join(base, stem)
                    rel = f"{problem}/{N}/{Kt}/{stem}"
                    fp, sp = os.path.join(d, "fitness.csv"), os.path.join(d, "solutions.csv")
                    if flt and not rel.startswith(flt):
                        continue
                    if os.path.exists(fp) and os.path.exists(sp):
                        yield problem, N, Kt, stem, rel, fp, sp


def main(flt):
    n_ok = n_tot = 0
    found = False
    for problem, N, Kt, stem, rel, fp, sp in find_instances(flt):
        found = True
        n_tot += 1
        try:
            fits = np.atleast_1d(np.loadtxt(fp))
            sols = np.atleast_2d(np.loadtxt(sp))
            inst = instance_path(problem, N, Kt, stem)
            if not os.path.exists(inst):
                print(f"[SKIP] {rel:40s} instance introuvable : {inst}")
                continue
            rec, info = recompute(problem, inst, sols)
            diff = np.abs(rec - fits)
            tol = TOL[problem]
            ok = int((diff < tol).sum())
            if len(diff) > 1 and np.std(fits) > 0 and np.std(rec) > 0:
                c = float(np.corrcoef(fits, rec)[0, 1])
            else:
                c = 1.0 if diff.max() < 1e-9 else 0.0     # variance nulle => juge sur l'ecart
            # coherent = tout dans la tolerance, OU alignement parfait a un bruit d'arrondi pres
            coherent = (ok == len(diff)) or (c > 1 - 1e-9 and diff.max() < 1e-4)
            tag = "OK " if ok == len(diff) else ("~  " if coherent else "!! ")
            print(f"[{tag}] {rel:40s} {ok:5d}/{len(diff):<5d} <{tol:g}  "
                  f"max|diff|={diff.max():.2e}  corr={c:+.5f}  ({info})")
            if coherent:
                n_ok += 1
                if ok != len(diff):
                    print(f"        -> {len(diff)-ok} lignes entre {tol:g} et {diff.max():.1e} : "
                          f"arrondi 6 decimales du .txt d'instance, pas un decalage genotype")
            else:
                bad = np.where(diff >= tol)[0][:3]
                for i in bad:
                    print(f"        ligne {i}: stocke={fits[i]:.6g}  recalcule={rec[i]:.6g}")
                print(f"        correlation={c:+.4f}  (~ -1 => inversion de signe ; "
                      f"~ 0 => mauvaise instance / lignes desynchronisees)")
        except Exception as e:                                # noqa: BLE001
            print(f"[ERR] {rel:40s} {type(e).__name__}: {e}")

    if not found:
        print("Aucune instance remplie" + (f" sous '{flt}'" if flt else "") + ".")
    else:
        print(f"\n{n_ok}/{n_tot} instance(s) 100% coherentes.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
