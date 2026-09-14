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
    """Renvoie (rec, info, bad_rows) — bad_rows = indices ignorés (valeurs hors alphabet)."""
    if problem in ("nk", "nk3"):
        prob = problem_NKlandscape(inst)
        assert sols.shape[1] == prob.N, (sols.shape, prob.N)
        alpha = prob.D
        in_range = (sols >= 0) & (sols < alpha) & (sols == np.floor(sols))
        bad_rows = np.where(~in_range.all(axis=1))[0]
        rec = np.full(sols.shape[0], np.nan)
        for i, r in enumerate(sols):
            if i in bad_rows:
                continue
            rec[i] = -prob.eval(r.astype(int))
        return rec, f"N={prob.N} K={prob.K} D={prob.D}", bad_rows
    we = WalshExpansion()
    we.load(inst)
    assert sols.shape[1] == we.n, (sols.shape, we.n)
    bad_rows = np.where(~np.isin(sols, (0, 1)).all(axis=1))[0]
    Q = we.to_symmetric_Q()
    S = np.where(np.isin(sols, (0, 1)), sols, 0.0) * 2.0 - 1.0
    rec = -np.einsum("ij,jk,ik->i", S, Q, S)
    rec[bad_rows] = np.nan
    return rec, f"N={we.n}", bad_rows


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
            rec, info, bad_rows = recompute(problem, inst, sols)
            valid = ~np.isnan(rec)
            valid_idx = np.where(valid)[0]
            n_valid = int(valid.sum())
            diff_full = np.abs(rec - fits)                 # NaN sur les lignes ignorées
            diff = diff_full[valid]
            tol = TOL[problem]
            ok = int((diff < tol).sum())
            fv, rv = fits[valid], rec[valid]
            if n_valid > 1 and np.std(fv) > 0 and np.std(rv) > 0:
                c = float(np.corrcoef(fv, rv)[0, 1])
            else:
                c = 1.0 if (diff.size and diff.max() < 1e-9) else 0.0
            dmax = float(diff.max()) if diff.size else 0.0
            # coherent = tout dans la tolerance, OU alignement parfait a un bruit d'arrondi pres
            coherent = (ok == n_valid) or (c > 1 - 1e-9 and dmax < 1e-4)
            tag = "OK " if (ok == n_valid and not len(bad_rows)) else ("~  " if coherent else "!! ")
            print(f"[{tag}] {rel:40s} {ok:5d}/{n_valid:<5d} <{tol:g}  "
                  f"max|diff|={dmax:.2e}  corr={c:+.5f}  ({info})")
            if len(bad_rows):
                print(f"        -> {len(bad_rows)} ligne(s) ignorée(s), valeur hors alphabet : "
                      f"{list(bad_rows[:5])}{' …' if len(bad_rows) > 5 else ''}  "
                      f"(données corrompues côté « the beast »)")
            if coherent:
                n_ok += 1
                if ok != n_valid:
                    print(f"        -> {n_valid-ok} lignes entre {tol:g} et {dmax:.1e} : "
                          f"arrondi 6 decimales du .txt d'instance, pas un decalage genotype")
            else:
                bad = valid_idx[np.where(diff >= tol)[0][:3]]
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
