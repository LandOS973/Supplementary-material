"""
SVGD-EDA Grid Search Dashboard
Run: streamlit run dashboard.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "config"
RANKING_DIR = ROOT / "additional_results" / "global_ranking"
BEAST_DIR = ROOT / "results" / "the beast"

INSTANCE_RE = re.compile(r"^(?P<problem>QUBO|NK|NK3)_dim(?P<dim>\d+)_t(?P<t>\d+)$")
# Partout dans le dashboard, chaque config SVGD est classée seule contre un pool FIXE de concurrents
# externes (82, 80 sur NK3) : aucune méthode SVGD ni variante dans le pool. Nos deux méthodes de
# référence figurent dans les fichiers de rang (tableaux du papier) : on les écarte ici.
SKIP_ALGOS = {"ppo-eda", "tabu", "svgd-eda", "se-svgd-eda"}

# Ordre important : les préfixes les plus spécifiques d'abord (le 1er match gagne).
# "kt" (seuil KL TRPO) doit passer avant "k" (kernel), sinon "kt0p01" écrase le kernel.
# "bo" doit passer avant "b" (kl_beta).
PARSERS = [
    ("adv", "advantage", str),
    ("eps", "epsilon_svgd", float),
    ("ds",  "decay_start_ratio", float),
    ("dm",  "decay_min_factor", float),
    ("bo",  "trpo_backoff", int),
    ("ks",  "ppo_epochs", int),
    ("pe",  "ppo_epochs", int),   # legacy (anciens résultats)
    ("ppo", "ppo_mode", str),     # ppoclip / ppokl / ppotrpo -> clip / kl / trpo
    ("ce",  "clip_eps", float),
    ("kt",  "trpo_kl_threshold", float),
    ("b",   "kl_beta", float),
    ("M",   "M", int),
    ("L",   "lambda", int),
    ("g",   "gamma", float),
    ("k",   "kernel", str),
]

COLORS = [
    "#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]




def _deslug(s: str) -> float:
    return float(s.replace("p", ".").replace("m", "-"))


def parse_config_name(name: str) -> dict:
    result: dict = {}
    for part in name.split("__"):
        for prefix, key, typ in PARSERS:
            if part.startswith(prefix) and len(part) > len(prefix):
                val = part[len(prefix):]
                try:
                    result[key] = val if typ is str else (int(val) if typ is int else _deslug(val))
                except Exception:
                    pass
                break
    return result


def _norm(series: pd.Series) -> pd.Series:
    """Return absolute value — NK stores best_fitness as negative."""
    return series.abs()


def _get_rank(problem: str, dim: int, t: int, score: float) -> int | None:
    fname = f"UBQP_N_{dim}_K_{t}_ranks.csv" if problem == "QUBO" else f"{problem}_N_{dim}_K_{t}_ranks.csv"
    path = RANKING_DIR / fname
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df = df[~df["name_algo"].str.lower().isin(SKIP_ALGOS)]
        return int(1 + (df["score"] > score).sum())
    except Exception:
        return None


DASHBOARD_DIR    = ROOT / "dashboard"
DASHBOARD_DIR.mkdir(exist_ok=True)
CACHE_FILE       = DASHBOARD_DIR / "cache.parquet"
INSTANCES_FILE   = DASHBOARD_DIR / "instances.parquet"
COMPARISONS_FILE = DASHBOARD_DIR / "comparisons.json"
FAVORITES_FILE   = DASHBOARD_DIR / "favorites.json"
ABLATIONS_FILE   = DASHBOARD_DIR / "ablations.json"
SENSI_FILE       = DASHBOARD_DIR / "sensi.json"   # méthodes, noms et paramètre de l'onglet Sensi
ANYTIME_FILE     = DASHBOARD_DIR / "anytime.json" # méthodes, concurrents et classes de l'onglet Anytime

# Migrate old files from previous locations
_old_cache = RESULTS_DIR / ".dashboard_cache.parquet"
_old_comps = ROOT / ".dashboard_comparisons.json"
if _old_cache.exists() and not CACHE_FILE.exists():
    _old_cache.rename(CACHE_FILE)
if _old_comps.exists() and not COMPARISONS_FILE.exists():
    _old_comps.rename(COMPARISONS_FILE)


def _rank_stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {"mean_rank": None, "top1": 0, "top3": 0, "top5": 0, "top10": 0, "mean_hamming": None}
    h = sub["hamming"].dropna() if "hamming" in sub.columns else pd.Series([], dtype=float)
    return {
        "mean_rank":    round(float(sub["rank"].mean()), 2),
        "top1":         int((sub["rank"] == 1).sum()),
        "top3":         int((sub["rank"] <= 3).sum()),
        "top5":         int((sub["rank"] <= 5).sum()),
        "top10":        int((sub["rank"] <= 10).sum()),
        "mean_hamming": round(float(h.mean()), 2) if not h.empty else None,
    }


def _compute_config_row(config_dir: Path) -> tuple[dict | None, list[dict]]:
    """Compute summary stats for one config directory. Returns (row | None, instance_rows)."""
    params = parse_config_name(config_dir.name)
    if not params:
        return None, []

    instance_rows = []
    for inst_dir in sorted(config_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        m = INSTANCE_RE.match(inst_dir.name)
        if not m:
            continue
        metrics = inst_dir / "best_metrics.csv"
        score = None
        hamming = None
        try:
            if metrics.exists():
                df = pd.read_csv(metrics)
                if not df.empty:
                    score = abs(float(df["best_fitness"].iloc[-1]))
                    hamming = float(df["avg_hamming"].iloc[-1]) if "avg_hamming" in df.columns else None
            if score is None:
                # best_metrics.csv vide (runs ppo rapatriés) → repli sur raw_scores.csv.
                # abs(best_fitness) == moyenne des raw_scores (vérifié NK/NK3/QUBO),
                # donc le score reste comparable aux configs non-ppo. Pas de hamming dispo.
                raw = inst_dir / "raw_scores.csv"
                if raw.exists():
                    s = pd.read_csv(raw)["score"].abs()
                    if not s.empty:
                        score = float(s.mean())
        except Exception:
            continue
        if score is None:
            continue
        rank = _get_rank(m.group("problem"), int(m.group("dim")), int(m.group("t")), score)
        if rank is not None:
            instance_rows.append({
                "config":  config_dir.name,
                "problem": m.group("problem"),
                "dim":     int(m.group("dim")),
                "t":       int(m.group("t")),
                "rank":    rank,
                "score":   score,
                "hamming": hamming,
            })

    if not instance_rows:
        return None, []
    idf = pd.DataFrame([{k: v for k, v in r.items() if k not in ("config", "dim", "t")}
                        for r in instance_rows])

    all_s  = _rank_stats(idf)
    nk_s   = _rank_stats(idf[idf["problem"] == "NK"])
    nk3_s  = _rank_stats(idf[idf["problem"] == "NK3"])
    qubo_s = _rank_stats(idf[idf["problem"] == "QUBO"])

    row = {
        "config": config_dir.name,
        **params,
        "mean_rank":         all_s["mean_rank"],
        "median_rank":       round(float(idf["rank"].median()), 1),
        "top1":  all_s["top1"],  "top3":  all_s["top3"],
        "top5":  all_s["top5"],  "top10": all_s["top10"],
        "top1_NK":           nk_s["top1"],
        "top1_NK3":          nk3_s["top1"],
        "top1_QUBO":         qubo_s["top1"],
        "mean_rank_NK":      nk_s["mean_rank"],
        "mean_rank_NK3":     nk3_s["mean_rank"],
        "mean_rank_QUBO":    qubo_s["mean_rank"],
        "top3_NK":           nk_s["top3"],
        "top3_NK3":          nk3_s["top3"],
        "top3_QUBO":         qubo_s["top3"],
        "mean_hamming":      all_s["mean_hamming"],
        "mean_hamming_NK":   nk_s["mean_hamming"],
        "mean_hamming_NK3":  nk3_s["mean_hamming"],
        "mean_hamming_QUBO": qubo_s["mean_hamming"],
        "n_instances":       len(idf),
    }
    return row, instance_rows


def load_summary() -> pd.DataFrame:
    all_dirs = [d for d in sorted(RESULTS_DIR.iterdir())
                if d.is_dir() and d.name.startswith("k")]

    # Load existing cache
    existing = pd.DataFrame()
    if CACHE_FILE.exists():
        try:
            existing = pd.read_parquet(CACHE_FILE)
        except Exception:
            pass

    # Detect configs on disk that are absent from the cache (name-based, not mtime)
    cached_names = set(existing["config"].tolist()) if not existing.empty else set()
    new_dirs = [d for d in all_dirs if d.name not in cached_names]

    # If instances file is missing or empty, rebuild it from all known configs (summary cache stays intact)
    _instances_ok = INSTANCES_FILE.exists() and INSTANCES_FILE.stat().st_size > 2048
    instances_only = False
    if not _instances_ok:
        # Fichier des instances absent ou vide : on recalcule toutes les configs, sinon il ne
        # contiendrait que les nouvelles (ou rien) et tous les rangs par instance disparaîtraient
        instances_only = not new_dirs
        new_dirs = all_dirs
    elif not new_dirs:
        return existing

    # Load existing instances file
    existing_inst = pd.DataFrame()
    if INSTANCES_FILE.exists():
        try:
            existing_inst = pd.read_parquet(INSTANCES_FILE)
        except Exception:
            pass

    # Recompute only new/missing configs
    rows: list[dict] = []
    new_inst_rows: list[dict] = []
    progress = st.progress(0, text="Mise à jour du cache…")
    for i, config_dir in enumerate(new_dirs):
        progress.progress(
            (i + 1) / max(len(new_dirs), 1),
            text=f"{i+1}/{len(new_dirs)} — {config_dir.name[:55]}",
        )
        row, inst_rows = _compute_config_row(config_dir)
        if row is not None:
            rows.append(row)
            new_inst_rows.extend(inst_rows)
    progress.empty()

    if not rows and existing.empty:
        return pd.DataFrame()

    disk_names = {d.name for d in all_dirs}
    updated = {d.name for d in new_dirs}

    if not instances_only:
        if not existing.empty:
            existing = existing[existing["config"].isin(disk_names) & ~existing["config"].isin(updated)]
        result = pd.concat(
            [existing, pd.DataFrame(rows)], ignore_index=True
        ).sort_values("mean_rank").reset_index(drop=True)
        result.to_parquet(CACHE_FILE)
    else:
        result = existing

    if not existing_inst.empty:
        existing_inst = existing_inst[
            existing_inst["config"].isin(disk_names) & ~existing_inst["config"].isin(updated)
        ]
    inst_result = pd.concat(
        [existing_inst, pd.DataFrame(new_inst_rows)], ignore_index=True
    )
    inst_result.to_parquet(INSTANCES_FILE)

    return result


@st.cache_data(show_spinner=False)
def load_curve(config: str, instance: str) -> pd.DataFrame | None:
    p = RESULTS_DIR / config / instance / "best_metrics.csv"
    try:
        if not p.exists():
            return None
        df = pd.read_csv(p)
        return df if not df.empty else None  # best_metrics vide (runs ppo) → pas de courbe
    except Exception:
        return None


def load_raw_scores(config: str, instance: str) -> pd.Series | None:
    p = RESULTS_DIR / config / instance / "raw_scores.csv"
    # La date de modification fait partie de la clé du cache : un fichier relancé puis rapatrié
    # (même nom de config) est relu au lieu de servir les anciens scores gardés en mémoire.
    return _load_raw_scores_cached(str(p), p.stat().st_mtime if p.exists() else None)


@st.cache_data(show_spinner=False)
def _load_raw_scores_cached(path: str, mtime: float | None) -> pd.Series | None:
    if mtime is None:
        return None
    try:
        return pd.read_csv(path)["score"].abs()
    except Exception:
        return None


def load_ranking(problem: str, dim: int, t: int) -> pd.DataFrame | None:
    fname = f"UBQP_N_{dim}_K_{t}_ranks.csv" if problem == "QUBO" else f"{problem}_N_{dim}_K_{t}_ranks.csv"
    p = RANKING_DIR / fname
    # Clé de cache avec la date de modification : fichiers de rang régénérés -> relus
    return _load_ranking_cached(str(p), p.stat().st_mtime if p.exists() else None)


@st.cache_data(show_spinner=False)
def _load_ranking_cached(path: str, mtime: float | None) -> pd.DataFrame | None:
    if mtime is None:
        return None
    try:
        df = pd.read_csv(path)
        return df[~df["name_algo"].str.lower().isin(SKIP_ALGOS)]
    except Exception:
        return None


def _beast_csv_paths(problem: str, dim: int, t: int, ti: int):
    """(fitness.csv, solutions.csv) pour l'instance de test `ti`, ou None.

    Non caché : un fichier déposé en cours de session est vu tout de suite.
    """
    if problem == "QUBO":
        stem, sub, ext = f"puboi_evo_n_{dim}_t_{t}_i_{ti + 1}", "qubo", "json"  # QUBO 1-indexé
    elif problem == "NK":
        stem, sub, ext = f"nk_{dim}_{t}_{ti}", "nk", "txt"
    elif problem == "NK3":
        stem, sub, ext = f"nk_{dim}_{t}_3_{ti}", "nk3", "txt"
    else:
        return None
    d = BEAST_DIR / sub / str(dim) / str(t) / stem                    # arbo courante
    fp, sp = d / "fitness.csv", d / "solutions.csv"
    if fp.exists() and sp.exists():
        return str(fp), str(sp)
    d = BEAST_DIR / f"{stem}_results"                                 # repli layout plat
    fp, sp = d / f"{stem}.{ext}_fitness.csv", d / f"{stem}.{ext}_solutions.csv"
    return (str(fp), str(sp)) if (fp.exists() and sp.exists()) else None


def load_beast_best(problem: str, dim: int, t: int, ti: int):
    """Meilleur génotype « the beast » pour l'instance de test `ti`.
    Renvoie (genotype (N,) int, fitness float, nb_lignes int) ou None.
    Convention fitness : NK/NK3 -> sco/N ; QUBO -> -sᵀQs (plus grand = meilleur)."""
    paths = _beast_csv_paths(problem, dim, t, ti)
    return _load_beast_best_cached(*paths) if paths else None


@st.cache_data(show_spinner=False)
def _load_beast_best_cached(fp: str, sp: str):
    try:
        fit = np.atleast_1d(np.loadtxt(fp))
        best = int(np.argmax(fit))
        geno = np.atleast_1d(np.loadtxt(sp, skiprows=best, max_rows=1))
        return geno.round().astype(int), float(fit[best]), int(fit.shape[0])
    except Exception:
        return None


def load_beast_optima(problem: str, dim: int, t: int, ti: int):
    """Les ~64000 optima locaux tabou : Hamming au best-known + fitness brute.
    Renvoie dict(ham, fit, best_known, best_geno, n, n_bad) ou None."""
    paths = _beast_csv_paths(problem, dim, t, ti)
    return _load_beast_optima_cached(problem, *paths) if paths else None


@st.cache_data(show_spinner="Chargement des optima « the beast »…")
def _load_beast_optima_cached(problem: str, fp: str, sp: str):
    try:
        fit = np.atleast_1d(np.loadtxt(fp)).astype(np.float64)
        S = pd.read_csv(sp, sep=r"\s+", header=None).dropna(axis=1, how="all").to_numpy()
        n = min(S.shape[0], fit.shape[0])
        S, fit = S[:n], fit[:n]
        amax = 2 if problem == "NK3" else 1                      # alphabet {0..amax}
        valid = (S.min(axis=1) >= 0) & (S.max(axis=1) <= amax)
        n_bad = int((~valid).sum())
        S, fit = S[valid], fit[valid]
        n = int(S.shape[0])
        bk = float(fit.max())

        # optima distincts = génotypes distincts, avec leur nb de visites (taille de bassin)
        uniq, inv, counts = np.unique(S, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        fit_of_uniq = fit[np.unique(inv, return_index=True)[1]]     # fitness par génotype distinct

        # optima globaux = génotypes distincts atteignant la meilleure fitness (peut y en avoir plusieurs)
        g_mask = np.isclose(fit_of_uniq, bk)
        g_counts = np.sort(counts[g_mask])[::-1]                   # visites par optimum global, décroissant
        n_global_opt = int(g_mask.sum())
        n_at_best_fit = int(g_counts.sum())                        # visites totales des optima globaux

        # ancre du plot = l'optimum global le plus visité
        g_idx = np.where(g_mask)[0]
        best_uidx = int(g_idx[np.argmax(counts[g_mask])])
        best_geno = uniq[best_uidx].round().astype(np.int16)
        ham = (S != best_geno).sum(axis=1).astype(np.float32)      # Hamming brut (bits)

        imode = int(counts.argmax())                               # optimum le plus visité, toutes fitness
        return dict(
            ham=ham, fit=fit.astype(np.float32), best_known=bk,
            best_geno=best_geno.astype(int), n=n, n_bad=n_bad,
            n_distinct=int(uniq.shape[0]),
            n_global_opt=n_global_opt,
            n_at_best_fit=n_at_best_fit,
            top_global_count=int(g_counts[0]),                     # visites de l'optimum global n°1
            mode_count=int(counts[imode]),
            mode_fit=float(fit_of_uniq[imode]),
        )
    except Exception:
        return None


INSTANCES_DIR = ROOT / "source_code" / "instances"


def _parse_nk_instance_file(path: Path, N: int, K: int, D: int):
    """Reparse un fichier d'instance nk_*.txt (même format que getTensorInstances_NK,
    mais en numpy pur, sans torch) -> (matrix_locus (N,K+1) int, matrix_contrib
    (N, D**(K+1)) float, vector_index (K+1,) float)."""
    lignes = path.read_text().splitlines()
    matrix_locus = np.zeros((N, K + 1), dtype=np.int64)
    matrix_contrib = np.zeros((N, D ** (K + 1)), dtype=np.float64)
    for n in range(N):
        for k in range(K + 1):
            matrix_locus[n, k] = int(lignes[1 + n * (K + 1) + k])
        for k in range(D ** (K + 1)):
            matrix_contrib[n, k] = float(lignes[1 + N * (K + 1) + n * (D ** (K + 1)) + k])
    vector_index = np.array([D ** (K - i) for i in range(K + 1)], dtype=np.float64)
    return matrix_locus, matrix_contrib, vector_index


@st.cache_data(show_spinner=False)
def _load_nk_arrays_cached(problem: str, dim: int, t: int, ti: int):
    D = 2 if problem == "NK" else 3
    subdir = "nk" if problem == "NK" else "nk3"
    fname = f"nk_{dim}_{t}_{ti}.txt" if D <= 2 else f"nk_{dim}_{t}_{D}_{ti}.txt"
    path = INSTANCES_DIR / subdir / str(dim) / str(t) / fname
    if not path.exists():
        return None
    try:
        return _parse_nk_instance_file(path, dim, t, D)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_qubo_Q_cached(dim: int, t: int, ti: int):
    qdir = INSTANCES_DIR / "QUBO"
    prefix = f"puboi_evo_n_{dim}_t_{t}_i_"
    try:
        files = sorted(
            (f for f in qdir.iterdir() if f.name.startswith(prefix) and f.name.endswith(".json")),
            key=lambda f: int(f.name[len(prefix):-5]),
        )
        if ti >= len(files):
            return None
        import sys as _sys_local
        if str(INSTANCES_DIR.parent / "utils") not in _sys_local.path:
            _sys_local.path.insert(0, str(INSTANCES_DIR.parent / "utils"))
        from walsh_expansion import WalshExpansion
        w = WalshExpansion()
        w.load(str(files[ti]))
        return w.to_symmetric_Q()
    except Exception:
        return None


def recompute_fitness(problem: str, dim: int, t: int, ti: int, genotype) -> float | None:
    """Fitness brute recalculée directement depuis le fichier d'instance pour un
    génotype donné — déterministe et rapide (formule tabulée / -sᵀQs, pas de
    ré-entraînement). Sert à vérifier qu'un génotype affiché et le score affiché à
    côté correspondent bien à la même chose (agrégation par vote majoritaire sur les
    restarts comprise : le score recalculé est celui du génotype réellement montré,
    pas la moyenne des scores des restarts individuels)."""
    genotype = np.asarray(genotype)
    if problem in ("NK", "NK3"):
        arrs = _load_nk_arrays_cached(problem, dim, t, ti)
        if arrs is None:
            return None
        matrix_locus, matrix_contrib, vector_index = arrs
        N = matrix_locus.shape[0]
        idx = (genotype[matrix_locus] * vector_index).sum(axis=1).astype(np.int64)
        return float(matrix_contrib[np.arange(N), idx].sum()) / N
    if problem == "QUBO":
        Q = _load_qubo_Q_cached(dim, t, ti)
        if Q is None:
            return None
        s = genotype.astype(np.float64) * 2 - 1
        return float(-(s @ Q @ s))
    return None


@st.cache_data(show_spinner=False)
def load_agent_debug(config: str) -> dict | None:
    """agent_debug.npz d'une config : clés '<instance>/genotype|epoch|score|meta'."""
    p = RESULTS_DIR / config / "agent_debug.npz"
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_comparisons() -> list[dict]:
    if COMPARISONS_FILE.exists():
        try:
            return json.loads(COMPARISONS_FILE.read_text())
        except Exception:
            return []
    return []


def save_comparisons(comps: list[dict]) -> None:
    COMPARISONS_FILE.write_text(json.dumps(comps, indent=2))
    load_comparisons.clear()


@st.cache_data(show_spinner=False)
def load_favorites() -> list[dict]:
    if FAVORITES_FILE.exists():
        try:
            return json.loads(FAVORITES_FILE.read_text())
        except Exception:
            return []
    return []


def save_favorites(favs: list[dict]) -> None:
    FAVORITES_FILE.write_text(json.dumps(favs, indent=2))
    load_favorites.clear()


@st.cache_data(show_spinner=False)
def load_ablations() -> list[dict]:
    if ABLATIONS_FILE.exists():
        try:
            return json.loads(ABLATIONS_FILE.read_text())
        except Exception:
            return []
    return []


def load_sensi() -> dict:
    try:
        return json.loads(SENSI_FILE.read_text()) if SENSI_FILE.exists() else {}
    except Exception:
        return {}


def load_anytime() -> dict:
    try:
        return json.loads(ANYTIME_FILE.read_text()) if ANYTIME_FILE.exists() else {}
    except Exception:
        return {}


def save_anytime(state: dict) -> None:
    ANYTIME_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def save_sensi(state: dict) -> None:
    SENSI_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def save_ablations(abls: list[dict]) -> None:
    ABLATIONS_FILE.write_text(json.dumps(abls, indent=2))
    load_ablations.clear()


from plotly.subplots import make_subplots

# ── Courbes des concurrents (réutilise curves/main_plot.py) ───────────────────
import csv as _csv
import sys as _sys

if str(ROOT / "curves") not in _sys.path:
    _sys.path.insert(0, str(ROOT / "curves"))
try:
    from main_plot import (
        find_competitor_files as _cmp_find,
        load_mean_across_runs as _cmp_mean,
        load_xy_from_csv as _cmp_xy,
        clip_series_to_budget as _cmp_clip,
        _normalize_score_sign as _cmp_signfix,
        aggregate_quantile_stats as _cmp_qstats,
        normalize_box_stats as _cmp_normbox,
        parse_float as _cmp_pf,
    )
    _COMPETITORS_OK = True
except Exception:
    _COMPETITORS_OK = False

_COMPETITOR_BUDGET = 50000


def _cmp_read_final_score(path, budget: int) -> float | None:
    """Score au budget exact sinon dernier score dispo (cf. main_curves_overall)."""
    try:
        with open(path, newline="") as fh:
            reader = _csv.reader(fh)
            header = next(reader, None)
            ri, si = 0, 1
            first_data = None
            if header:
                low = [c.strip().lower() for c in header]
                if "runtime" in low or "score" in low:
                    if "runtime" in low:
                        ri = low.index("runtime")
                    if "score" in low:
                        si = low.index("score")
                elif _cmp_pf(header[0].strip()) is not None and _cmp_pf(header[1].strip()) is not None:
                    first_data = header
            last = at_budget = None
            rows = ([first_data] if first_data is not None else []) + list(reader)
            for row in rows:
                if len(row) <= max(ri, si):
                    continue
                rt, sc = _cmp_pf(row[ri].strip()), _cmp_pf(row[si].strip())
                if rt is None or sc is None:
                    continue
                last = sc
                if abs(rt - budget) < 1e-9:
                    at_budget = sc
            return at_budget if at_budget is not None else last
    except OSError:
        return None


def _cmp_pct(sv: list[float], p: float) -> float:
    if p <= 0:
        return sv[0]
    if p >= 1:
        return sv[-1]
    k = (len(sv) - 1) * p
    f = int(k)
    c = min(f + 1, len(sv) - 1)
    return sv[f] if f == c else sv[f] * (c - k) + sv[c] * (k - f)


def _cmp_final_stats(paths, budget: int) -> dict | None:
    vals: list[float] = []
    for p in paths:
        p = Path(p)
        # mean_curve_budget_* → utiliser final_scores_budget_*.csv (1 score par run)
        if p.name.startswith("mean_curve_budget_"):
            fs = p.parent / f"final_scores_budget_{budget}.csv"
            if fs.exists():
                try:
                    with fs.open(newline="") as fh:
                        for row in _csv.DictReader(fh):
                            sc = _cmp_pf((row.get("score") or "").strip())
                            if sc is not None:
                                vals.append(sc)
                    continue
                except OSError:
                    pass
        sc = _cmp_read_final_score(p, budget)
        if sc is not None:
            vals.append(sc)
    if not vals:
        return None
    vals.sort()
    return {
        "low": _cmp_pct(vals, 0.02), "q1": _cmp_pct(vals, 0.25),
        "med": _cmp_pct(vals, 0.50), "q3": _cmp_pct(vals, 0.75),
        "high": _cmp_pct(vals, 0.98), "mean": sum(vals) / len(vals),
    }


@st.cache_data(show_spinner=False)
def load_competitor_curve(algo: str, problem: str, dim: int, t: int):
    """(x, y) courbe moyenne d'un concurrent — score normalisé, borné au budget."""
    if not _COMPETITORS_OK:
        return None
    try:
        paths, has_header = _cmp_find(algo, dim, t, problem)
        if has_header:
            if len(paths) > 1:
                x, y = _cmp_mean(paths)
            else:
                x, y = _cmp_xy(paths[0], has_header=True, x_key="runtime", y_key="mean")
        else:
            x, y = _cmp_xy(paths[0], has_header=False)
        x, y = _cmp_clip(list(x), list(y), max_budget=_COMPETITOR_BUDGET)
        if not x:
            return None
        y = _cmp_signfix(problem, list(y))
        return list(x), list(y)
    except Exception:
        return None


_CMP_I_RE = re.compile(r"_i_(\d+)_r_\d+")


@st.cache_data(show_spinner=False)
def load_competitor_instance_curve(algo: str, problem: str, dim: int, t: int, ti: int):
    """(x, y) courbe du concurrent sur la SEULE instance de test `ti` (moyenne sur ses restarts).
    None si pas de trajectoires par-run (dossier avec seulement mean_curve)."""
    if not _COMPETITORS_OK:
        return None
    try:
        paths, has_header = _cmp_find(algo, dim, t, problem)
    except Exception:
        return None
    sel = [p for p in paths
           if (_m := _CMP_I_RE.search(Path(p).name)) and int(_m.group(1)) == ti]
    if not sel:
        return None
    sums: dict[float, float] = {}
    counts: dict[float, int] = {}
    for p in sel:
        try:
            xs, ys = _cmp_xy(Path(p), has_header=has_header, x_key="runtime", y_key="score")
        except Exception:
            continue
        for x, y in zip(xs, ys):
            sums[x] = sums.get(x, 0.0) + y
            counts[x] = counts.get(x, 0) + 1
    if not sums:
        return None
    xs = sorted(sums)
    ys = [sums[x] / counts[x] for x in xs]
    xs, ys = _cmp_clip(list(xs), list(ys), max_budget=_COMPETITOR_BUDGET)
    if not xs:
        return None
    return list(xs), _cmp_signfix(problem, list(ys))


@st.cache_data(show_spinner=False)
def load_competitor_instance_final(algo: str, problem: str, dim: int, t: int, ti: int):
    """Score final moyen du concurrent sur l'instance de test `ti` (via final_scores_budget_*.csv)."""
    if not _COMPETITORS_OK:
        return None
    try:
        paths, _ = _cmp_find(algo, dim, t, problem)
    except Exception:
        return None
    for p in paths:
        p = Path(p)
        if not p.name.startswith("mean_curve_budget_"):
            continue
        fs = p.parent / f"final_scores_budget_{_COMPETITOR_BUDGET}.csv"
        if not fs.exists():
            continue
        vals = []
        try:
            with fs.open(newline="") as fh:
                for row in _csv.DictReader(fh):
                    if str(row.get("instance", "")).strip() == str(ti):
                        sc = _cmp_pf((row.get("score") or "").strip())
                        if sc is not None:
                            vals.append(sc)
        except OSError:
            continue
        if vals:
            return _cmp_signfix(problem, [sum(vals) / len(vals)])[0]
    return None


@st.cache_data(show_spinner=False)
def load_competitor_box(algo: str, problem: str, dim: int, t: int):
    """Stats quantiles finales d'un concurrent : {low,q1,med,q3,high,mean} ou None."""
    if not _COMPETITORS_OK:
        return None
    try:
        paths, has_header = _cmp_find(algo, dim, t, problem)
    except Exception:
        return None
    stats = None
    try:
        if has_header:
            stats = _cmp_qstats(paths, problem_name=problem)
    except Exception:
        stats = None
    if not stats:
        raw = _cmp_final_stats(paths, _COMPETITOR_BUDGET)
        if raw:
            try:
                stats = _cmp_normbox(problem, raw)
            except Exception:
                stats = raw
    if not stats:
        return None
    # boîte dégénérée (1 seule valeur récupérée) → pas de boxplot exploitable
    if abs(float(stats["high"]) - float(stats["low"])) < 1e-9:
        return None
    return stats


def _config_label(row: pd.Series) -> str:
    """Short human-readable label for a config row (used in selectboxes)."""
    parts = []
    for col, abbr in [
        ("kernel", "k"), ("ppo_epochs", "ks"), ("epsilon_svgd", "eps"),
        ("M", "M"), ("lambda", "L"), ("gamma", "g"),
    ]:
        if col in row.index and pd.notna(row.get(col)):
            parts.append(f"{abbr}={row[col]}")
    return "  ".join(parts) if parts else row["config"][:60]


# Ordered (param_key, display_abbrev) for diff labels — ds and dm excluded intentionally
_DIFF_LABEL_PARAMS = [
    ("kernel",          "k"),
    ("advantage",       "adv"),
    ("M",               "M"),
    ("lambda",          "L"),
    ("ppo_epochs",      "ks"),
    ("clip_eps",        "ce"),
    ("epsilon_svgd",    "eps"),
    ("gamma",           "g"),
]


def _diff_labels(cfg_a: str, cfg_b: str) -> tuple[str, str]:
    """Labels for A and B showing only the params that differ (ds/dm excluded)."""
    pa = parse_config_name(cfg_a)
    pb = parse_config_name(cfg_b)
    diff_keys = {k for k, _ in _DIFF_LABEL_PARAMS if pa.get(k) != pb.get(k)}

    def _fmt(v) -> str:
        if isinstance(v, float):
            return f"{v:g}"
        return str(v) if v is not None else "—"

    def _label(params: dict) -> str:
        parts = [
            f"{abbr}={_fmt(params[k])}"
            for k, abbr in _DIFF_LABEL_PARAMS
            if k in diff_keys and k in params
        ]
        return "  ".join(parts) if parts else cfg_a[-30:]

    return _label(pa), _label(pb)


# ── Tab fragments ─────────────────────────────────────────────────────────────
# Each tab is a @st.fragment: interacting with its widgets re-runs only that
# function, not the full script — other tabs are unaffected.

# Single benches shown as their own score/rank columns in the Classement tab
BENCH_COLS = [("NK3", 512, 8), ("NK3", 256, 8)]


@st.fragment
def tab_classement(filtered: pd.DataFrame) -> None:
    problem_sel = st.radio(
        "Problème", ["Tous", "NK", "NK3", "QUBO"],
        horizontal=True, key="class_problem",
    )

    stat_cols = {
        "Tous": ["mean_rank", "median_rank", "top1", "top3", "top5", "top10",
                 "top1_NK", "top1_NK3", "top1_QUBO"],
        "NK":   ["mean_rank_NK",   "top1_NK",   "top3_NK"],
        "NK3":  ["mean_rank_NK3",  "top1_NK3",  "top3_NK3"],
        "QUBO": ["mean_rank_QUBO", "top1_QUBO", "top3_QUBO"],
    }[problem_sel]

    rank_col = {
        "Tous": "mean_rank", "NK": "mean_rank_NK",
        "NK3": "mean_rank_NK3", "QUBO": "mean_rank_QUBO",
    }[problem_sel]

    display = filtered.dropna(subset=[rank_col]) if rank_col in filtered.columns else filtered
    display = display.sort_values(rank_col) if rank_col in display.columns else display

    if "M" in display.columns:
        m_excl = st.multiselect(
            "Exclure M", sorted(display["M"].dropna().unique().tolist()),
            default=[], key="class_m_excl",
        )
        if m_excl:
            display = display[~display["M"].isin(m_excl)]

    # Score / rank on single benches, to sort on them specifically
    if INSTANCES_FILE.exists():
        try:
            inst = pd.read_parquet(INSTANCES_FILE)
            for prob, dim, t in BENCH_COLS:
                b = inst[(inst["problem"] == prob) & (inst["dim"] == dim) & (inst["t"] == t)]
                sfx = f"{prob.lower()}_{dim}_{t}"
                b = b[["config", "score", "rank"]].rename(
                    columns={"score": f"score_{sfx}", "rank": f"rank_{sfx}"})
                display = display.merge(b.drop_duplicates("config"), on="config", how="left")
        except Exception:
            pass

    cols = [c for c in [
        *stat_cols,
        *[f"{k}_{p.lower()}_{d}_{t}" for p, d, t in BENCH_COLS for k in ("score", "rank")],
        "ppo_mode", "ppo_epochs", "clip_eps", "kl_beta", "trpo_kl_threshold",
        "epsilon_svgd", "gamma", "M", "lambda",
        "kernel", "advantage",
        "decay_start_ratio", "decay_min_factor",
        "n_instances", "config",
    ] if c in display.columns]

    display_reset = display[cols].reset_index(drop=True)
    event = st.dataframe(
        display_reset,
        use_container_width=True,
        height=620,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "config":         st.column_config.TextColumn("Config",     width="large"),
            "mean_rank":      st.column_config.NumberColumn("Rank moy", format="%.2f"),
            "mean_rank_NK":   st.column_config.NumberColumn("Rank NK",  format="%.2f"),
            "mean_rank_NK3":  st.column_config.NumberColumn("Rank NK3", format="%.2f"),
            "mean_rank_QUBO": st.column_config.NumberColumn("Rank QUBO",format="%.2f"),
            "median_rank":    st.column_config.NumberColumn("Rank med", format="%.1f"),
            **{f"score_{p.lower()}_{d}_{t}": st.column_config.NumberColumn(f"Score {p} {d} K{t}", format="%.6f")
               for p, d, t in BENCH_COLS},
            **{f"rank_{p.lower()}_{d}_{t}": st.column_config.NumberColumn(f"Rank {p} {d} K{t}", format="%d")
               for p, d, t in BENCH_COLS},
            "ppo_mode":       st.column_config.TextColumn("ppo mode", width="small"),
            "ppo_epochs":     st.column_config.NumberColumn("ks",       format="%d"),
            "clip_eps":       st.column_config.NumberColumn("ce",       format="%.2f"),
            "kl_beta":        st.column_config.NumberColumn("β",        format="%.3f"),
            "trpo_kl_threshold": st.column_config.NumberColumn("kt",    format="%.3f"),
            "epsilon_svgd":   st.column_config.NumberColumn("ε_svgd",  format="%.4f"),
            "gamma":          st.column_config.NumberColumn("γ",        format="%.4f"),
        },
    )
    sel_rows = event.selection.rows if event and hasattr(event, "selection") else []
    if sel_rows and "config" in display_reset.columns:
        st.code(display_reset["config"].iloc[sel_rows[0]], language=None)


def _comparison_markdown(name: str, cfg_a: str, cfg_b: str, label_a: str, label_b: str, rows: list[dict],
                         col_score_a: str, col_score_b: str, col_rank_a: str, col_rank_b: str,
                         col_hamming_a: str, col_hamming_b: str) -> str:
    """Rapport Markdown autonome d'une comparaison A vs B, destiné à un LLM : il doit pouvoir
    l'analyser sans connaître le dashboard (configs décodées, méthodologie, résumé, détail)."""
    def fmt(v, nd=4):
        return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    def params(cfg):
        return ", ".join(f"{k}={v}" for k, v in parse_config_name(cfg).items())

    n = len(rows)
    wins_a = sum(r["_gap"] > 0 for r in rows); wins_b = sum(r["_gap"] < 0 for r in rows)
    ra = [r[col_rank_a] for r in rows if r[col_rank_a] is not None]
    rb = [r[col_rank_b] for r in rows if r[col_rank_b] is not None]
    g = _global_wilcoxon(r["_gap"] for r in rows)
    out = [f"# Comparaison : {name}", "",
           f"- **A = {label_a}** : `{cfg_a}`", f"  - paramètres : {params(cfg_a)}",
           f"- **B = {label_b}** : `{cfg_b}`", f"  - paramètres : {params(cfg_b)}", "",
           "## Méthodologie", "",
           "- Algorithmes : SVGD-EDA multi-agents (m agents × λ échantillons par agent), budget de "
           "50 000 évaluations par run.",
           "- Benchmark : 56 classes d'instances — QUBO (PUBOi, n ∈ {64,128,256,512}, types t = 0..5), "
           "NK et NK3 (ternaire) (n ∈ {64,128,256,512}, K ∈ {1,2,4,8}). Chaque classe = 10 instances × "
           "10 restarts = 100 runs.",
           "- Score d'une classe : moyenne sur les 100 runs de la meilleure fitness finale (plus haut = "
           "meilleur ; valeur absolue pour QUBO).",
           "- Écart (Gap %) = (score A − score B) / |score B| × 100 ; > 0 : A meilleur.",
           "- Rang : 1 + nombre de concurrents externes strictement meilleurs, contre un pool fixe de 82 "
           "optimiseurs externes (80 sur NK3 : PBIL et PPBIL ne gèrent pas le ternaire) — 79 algorithmes "
           "Nevergrad discrets + PBIL + PEDA + PPBIL. Aucune méthode SVGD dans le pool.",
           "- p (classe) : Wilcoxon signé apparié bilatéral sur les 10 moyennes par instance (exact).",
           "- Test global : Wilcoxon signé bilatéral sur les écarts relatifs, une paire par classe "
           "(approximation normale pour n > 50).",
           "- Hamming : distance de Hamming moyenne entre agents en fin de run (diversité).", "",
           "## Résumé", "",
           "| indicateur | A | B |", "|---|---|---|",
           f"| victoires (meilleur score, sur {n} classes) | {wins_a} | {wins_b} |",
           f"| top 1 vs pool | {sum(v == 1 for v in ra)} | {sum(v == 1 for v in rb)} |"]
    if ra and rb:
        out += [f"| rang moyen vs pool | {np.mean(ra):.2f} | {np.mean(rb):.2f} |",
                f"| rang médian vs pool | {np.median(ra):.1f} | {np.median(rb):.1f} |"]
    out.append("")
    for fam in ("QUBO", "NK", "NK3"):
        fg = [r["_gap"] for r in rows if r["_prob"] == fam]
        if fg:
            out.append(f"- Écart moyen {fam} : {np.mean(fg):+.3f} % (médian {np.median(fg):+.3f} %, "
                       f"A meilleur sur {sum(x > 0 for x in fg)}/{len(fg)})")
    if g:
        verdict = ((f"A ({label_a}) meilleur" if g["median"] > 0 else f"B ({label_b}) meilleur")
                   if g["p"] < 0.05 else "pas de différence globale significative")
        out.append(f"- Test global ({g['n']} classes) : p = {g['p']:.3g}, écart médian {g['median']:+.3f} % "
                   f"→ {verdict}")
    out += ["", "## Détail par classe", "",
            "| classe | problème | n | K/t | score A | score B | écart % | p (classe) | rang A | rang B | "
            "Hamming A | Hamming B | meilleur concurrent externe (score) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: ({"NK": 0, "NK3": 1, "QUBO": 2}.get(r["_prob"], 9), r["_dim"], r["_t"])):
        inst = r["Instance"]
        p = _paired_wilcoxon(_instance_means(load_raw_scores(cfg_a, inst)),
                             _instance_means(load_raw_scores(cfg_b, inst)))
        rk = load_ranking(r["_prob"], r["_dim"], r["_t"])
        best = (f"{rk.iloc[rk['score'].argmax()]['name_algo']} ({rk['score'].max():.4f})"
                if rk is not None and not rk.empty else "—")
        out.append(f"| {inst} | {r['_prob']} | {r['_dim']} | {r['_t']} | {fmt(r[col_score_a])} | "
                   f"{fmt(r[col_score_b])} | {r['_gap']:+.3f} | {fmt(p, 3)} | {fmt(r[col_rank_a])} | "
                   f"{fmt(r[col_rank_b])} | {fmt(r[col_hamming_a], 2)} | {fmt(r[col_hamming_b], 2)} | {best} |")
    return "\n".join(out) + "\n"


@st.fragment
def tab_comparaison(all_df: pd.DataFrame, sorted_instances: list) -> None:
    comps = load_comparisons()
    all_configs = all_df["config"].tolist()
    sel_idx = st.session_state.get("comp_sel_idx")

    # ── Vue liste (dossiers) ──────────────────────────────────────────────────
    if sel_idx is None or sel_idx >= len(comps):
        h1, h2 = st.columns([6, 1])
        h1.markdown("#### 📁 Dossiers")
        if h2.button("+ Nouveau", key="comp_new", use_container_width=True):
            comps.append({
                "name": f"Comparaison {len(comps) + 1}",
                "config_a": "", "config_b": "", "instances": [],
            })
            save_comparisons(comps)
            st.session_state["comp_sel_idx"] = len(comps) - 1
            st.rerun()

        if not comps:
            st.info("Aucun dossier. Cliquez sur « + Nouveau » pour commencer.")
            return

        grid = st.columns(3)
        for i, grp in enumerate(comps):
            with grid[i % 3]:
                with st.container(border=True):
                    t1, t2 = st.columns([5, 1])
                    if t1.button(f"📁 {grp['name']}", key=f"comp_sel_{i}", use_container_width=True):
                        st.session_state["comp_sel_idx"] = i
                        st.rerun()
                    if t2.button("🗑", key=f"comp_del_{i}"):
                        comps.pop(i)
                        save_comparisons(comps)
                        st.rerun()
                    parts = []
                    for key, prefix in [("config_a", "A"), ("config_b", "B")]:
                        cfg = grp.get(key, "")
                        if cfg:
                            r = all_df[all_df["config"] == cfg]
                            lbl = _config_label(r.iloc[0]) if not r.empty else cfg[-28:]
                            parts.append(f"{prefix}: {lbl}")
                    if parts:
                        st.caption("  |  ".join(parts))
                    n_inst = len(grp.get("instances", []))
                    if n_inst:
                        st.caption(f"{n_inst} instance{'s' if n_inst > 1 else ''}")
        return

    # ── Vue comparaison (plein écran) ─────────────────────────────────────────
    grp = comps[sel_idx]

    hd1, hd2 = st.columns([1, 8])
    if hd1.button("← Retour", key="comp_back"):
        st.session_state.pop("comp_sel_idx", None)
        st.rerun()
    hd2.markdown(f"#### 📂 {grp['name']}")

    new_name = st.text_input("Nom", value=grp["name"], key=f"comp_name_{sel_idx}")
    if new_name and new_name != grp["name"]:
        comps[sel_idx]["name"] = new_name
        save_comparisons(comps)

    cfg_options = [""] + all_configs
    idx_a = cfg_options.index(grp["config_a"]) if grp.get("config_a") in cfg_options else 0
    idx_b = cfg_options.index(grp["config_b"]) if grp.get("config_b") in cfg_options else 0

    COLOR_A, COLOR_B = "#1f77b4", "#2ca02c"

    cc1, cc2 = st.columns(2)
    with cc1:
        cfg_a = st.selectbox("Config A", cfg_options, index=idx_a, key=f"comp_cfg_a_{sel_idx}")
    with cc2:
        cfg_b = st.selectbox("Config B", cfg_options, index=idx_b, key=f"comp_cfg_b_{sel_idx}")

    if cfg_a != grp.get("config_a") or cfg_b != grp.get("config_b"):
        comps[sel_idx]["config_a"] = cfg_a
        comps[sel_idx]["config_b"] = cfg_b
        save_comparisons(comps)

    if not cfg_a or not cfg_b:
        st.info("Sélectionnez deux configs pour afficher la comparaison.")
        return

    # Noms personnalisés
    nc1, nc2 = st.columns(2)
    label_a = nc1.text_input("Nom Config A", value=grp.get("label_a", "Config A"), key=f"comp_label_a_{sel_idx}", placeholder="Config A")
    label_b = nc2.text_input("Nom Config B", value=grp.get("label_b", "Config B"), key=f"comp_label_b_{sel_idx}", placeholder="Config B")
    label_a = label_a or "Config A"
    label_b = label_b or "Config B"
    if label_a != grp.get("label_a") or label_b != grp.get("label_b"):
        comps[sel_idx]["label_a"] = label_a
        comps[sel_idx]["label_b"] = label_b
        save_comparisons(comps)

    st.markdown(
        f'<span style="color:{COLOR_A};font-weight:bold;font-size:16px;">■ {label_a}</span>'
        f'&nbsp;&nbsp;&nbsp;&nbsp;'
        f'<span style="color:{COLOR_B};font-weight:bold;font-size:16px;">■ {label_b}</span>',
        unsafe_allow_html=True,
    )

    saved_insts = [i for i in grp.get("instances", []) if i in sorted_instances]
    inst_key = f"comp_inst_{sel_idx}"

    # Quick-select buttons — update session state BEFORE the multiselect renders
    nk_insts   = [i for i in sorted_instances if INSTANCE_RE.match(i).group("problem") == "NK"]
    nk3_insts  = [i for i in sorted_instances if INSTANCE_RE.match(i).group("problem") == "NK3"]
    qubo_insts = [i for i in sorted_instances if INSTANCE_RE.match(i).group("problem") == "QUBO"]

    b1, b2, b3 = st.columns(3)
    if b1.button("+ Tout NK",   key=f"comp_nk_{sel_idx}",   use_container_width=True):
        current = list(st.session_state.get(inst_key, saved_insts))
        st.session_state[inst_key] = sorted(set(current) | set(nk_insts),   key=sorted_instances.index)
    if b2.button("+ Tout NK3",  key=f"comp_nk3_{sel_idx}",  use_container_width=True):
        current = list(st.session_state.get(inst_key, saved_insts))
        st.session_state[inst_key] = sorted(set(current) | set(nk3_insts),  key=sorted_instances.index)
    if b3.button("+ Tout QUBO", key=f"comp_qubo_{sel_idx}", use_container_width=True):
        current = list(st.session_state.get(inst_key, saved_insts))
        st.session_state[inst_key] = sorted(set(current) | set(qubo_insts), key=sorted_instances.index)

    r1, r2 = st.columns([5, 1])
    with r1:
        instances = st.multiselect(
            "Instances", sorted_instances, default=saved_insts,
            key=inst_key,
        )
    with r2:
        show_top5 = st.checkbox("Top 5", value=True, key=f"comp_top5_{sel_idx}")

    if instances != saved_insts:
        comps[sel_idx]["instances"] = instances
        save_comparisons(comps)

    # ── Résumé A vs B ─────────────────────────────────────────────────────────
    summary_rows = []
    # column name helpers — built once, reused everywhere
    col_score_a   = f"Score {label_a}"
    col_score_b   = f"Score {label_b}"
    col_rank_a    = f"Rank {label_a}"
    col_rank_b    = f"Rank {label_b}"
    col_hamming_a = f"Hamming {label_a}"
    col_hamming_b = f"Hamming {label_b}"

    for inst in instances:
        ca = load_curve(cfg_a, inst)
        cb = load_curve(cfg_b, inst)
        if ca is None or cb is None:
            continue
        sa = abs(float(ca["best_fitness"].iloc[-1]))
        sb = abs(float(cb["best_fitness"].iloc[-1]))
        gap = (sa - sb) / abs(sb) * 100 if sb != 0 else 0.0
        ha = float(ca["avg_hamming"].iloc[-1]) if "avg_hamming" in ca.columns else None
        hb = float(cb["avg_hamming"].iloc[-1]) if "avg_hamming" in cb.columns else None
        la = float(ca["avg_l1"].iloc[-1]) if "avg_l1" in ca.columns else None
        lb = float(cb["avg_l1"].iloc[-1]) if "avg_l1" in cb.columns else None
        m_inst = INSTANCE_RE.match(inst)
        prob, dim, t = m_inst.group("problem"), int(m_inst.group("dim")), int(m_inst.group("t"))
        ra = _get_rank(prob, dim, t, sa)
        rb = _get_rank(prob, dim, t, sb)
        summary_rows.append({
            "Instance":     inst,
            "_prob": prob, "_dim": dim, "_t": t,
            col_score_a:    round(sa, 4),
            col_score_b:    round(sb, 4),
            "Gap %":        round(gap, 2),
            "_gap":         gap,
            col_hamming_a:  round(ha, 2) if ha is not None else None,
            col_hamming_b:  round(hb, 2) if hb is not None else None,
            "_l1_a":        round(la, 2) if la is not None else None,
            "_l1_b":        round(lb, 2) if lb is not None else None,
            col_rank_a:     ra,
            col_rank_b:     rb,
            "Résultat":     f"{label_a} ✓" if gap > 0 else (f"{label_b} ✓" if gap < 0 else "égalité"),
        })

    if summary_rows:
        st.download_button(
            "⬇️ Export", key=f"comp_llm_{sel_idx}", mime="text/markdown",
            file_name=f"comparaison_{re.sub(r'[^A-Za-z0-9_-]+', '_', grp['name'])}.md",
            help="Rapport autonome (configs, méthodologie, résumé, détail par classe) à donner à un LLM.",
            data=lambda: _comparison_markdown(grp["name"], cfg_a, cfg_b, label_a, label_b, summary_rows,
                                              col_score_a, col_score_b, col_rank_a, col_rank_b,
                                              col_hamming_a, col_hamming_b),
        )
        with st.expander("📊 Résumé", expanded=True):
            wins_a     = sum(1 for r in summary_rows if r["Résultat"] == f"{label_a} ✓")
            wins_b     = sum(1 for r in summary_rows if r["Résultat"] == f"{label_b} ✓")
            mean_gap   = sum(r["Gap %"] for r in summary_rows) / len(summary_rows)
            ranks_a    = [r[col_rank_a]    for r in summary_rows if r[col_rank_a]    is not None]
            ranks_b    = [r[col_rank_b]    for r in summary_rows if r[col_rank_b]    is not None]

            top1_a = sum(1 for r in summary_rows if r[col_rank_a] == 1)
            top1_b = sum(1 for r in summary_rows if r[col_rank_b] == 1)

            s_th  = "padding:8px 24px;font-size:18px;font-weight:700;border-bottom:2px solid #ccc;text-align:center;white-space:nowrap;"
            s_td0 = "padding:7px 24px;font-size:17px;font-weight:600;border-bottom:1px solid #f0f0f0;white-space:nowrap;color:#555;"
            s_td  = "padding:7px 24px;font-size:18px;font-weight:700;border-bottom:1px solid #f0f0f0;text-align:center;white-space:nowrap;"

            stat_rows = [("Victoires", wins_a, wins_b), ("Top 1", top1_a, top1_b)]
            if ranks_a and ranks_b:
                stat_rows += [
                    ("Rank moy", f"{sum(ranks_a)/len(ranks_a):.1f}", f"{sum(ranks_b)/len(ranks_b):.1f}"),
                    ("Rank méd", sorted(ranks_a)[len(ranks_a)//2],   sorted(ranks_b)[len(ranks_b)//2]),
                ]
            for dim in [64, 128, 256, 512]:
                dim_ha = [r[col_hamming_a] for r in summary_rows if r["_dim"] == dim and r[col_hamming_a] is not None]
                dim_hb = [r[col_hamming_b] for r in summary_rows if r["_dim"] == dim and r[col_hamming_b] is not None]
                if dim_ha and dim_hb:
                    stat_rows.append((f"Hamming {dim}", f"{sum(dim_ha)/len(dim_ha):.1f}", f"{sum(dim_hb)/len(dim_hb):.1f}"))
            for dim in [64, 128, 256, 512]:
                dim_la = [r["_l1_a"] for r in summary_rows if r["_dim"] == dim and r["_l1_a"] is not None]
                dim_lb = [r["_l1_b"] for r in summary_rows if r["_dim"] == dim and r["_l1_b"] is not None]
                if dim_la and dim_lb:
                    stat_rows.append((f"L1 {dim}", f"{sum(dim_la)/len(dim_la):.2f}", f"{sum(dim_lb)/len(dim_lb):.2f}"))
            for prob in ["NK", "NK3", "QUBO"]:
                prob_gaps = [r["Gap %"] for r in summary_rows if r["_prob"] == prob]
                if prob_gaps:
                    stat_rows.append((f"Gap {prob}", f"{sum(prob_gaps)/len(prob_gaps):+.2f}%", ""))
            # Test global, même méthode que l'onglet Ablation (un seul test ici : pas de Holm)
            g = _global_wilcoxon(r["_gap"] for r in summary_rows)
            if g:
                verdict = ((f"{label_a} meilleure" if g["median"] > 0 else f"{label_b} meilleure")
                           if g["p"] < 0.05 else "pas de différence globale")
                stat_rows.append((f"Wilcoxon global ({g['n']} classes)",
                                  f"p = {g['p']:.2g} · écart médian {g['median']:+.2f}%", verdict))

            html_stats = '<table style="border-collapse:collapse;margin-bottom:16px;">'
            html_stats += (f'<thead><tr>'
                           f'<th style="{s_th}color:#333;text-align:left;"></th>'
                           f'<th style="{s_th}color:{COLOR_A};">{label_a}</th>'
                           f'<th style="{s_th}color:{COLOR_B};">{label_b}</th>'
                           f'</tr></thead><tbody>')
            for stat, va, vb in stat_rows:
                html_stats += (f'<tr>'
                               f'<td style="{s_td0}">{stat}</td>'
                               f'<td style="{s_td}color:{COLOR_A};">{va}</td>'
                               f'<td style="{s_td}color:{COLOR_B};">{vb}</td>'
                               f'</tr>')
            html_stats += "</tbody></table>"

            # Comparaison Hamming par K (t pour QUBO) — à droite du résumé
            hk_rows = []
            for prob in ["NK", "NK3", "QUBO"]:
                klbl = "t" if prob == "QUBO" else "K"
                for k in sorted({r["_t"] for r in summary_rows if r["_prob"] == prob}):
                    ha = [r[col_hamming_a] for r in summary_rows
                          if r["_prob"] == prob and r["_t"] == k and r[col_hamming_a] is not None]
                    hb = [r[col_hamming_b] for r in summary_rows
                          if r["_prob"] == prob and r["_t"] == k and r[col_hamming_b] is not None]
                    if ha and hb:
                        hk_rows.append((f"{prob} · {klbl} {k}",
                                        sum(ha) / len(ha), sum(hb) / len(hb)))

            html_hk = ""
            if hk_rows:
                html_hk = '<table style="border-collapse:collapse;margin-bottom:16px;">'
                html_hk += (f'<thead><tr>'
                            f'<th style="{s_th}color:#333;text-align:left;">Hamming / K</th>'
                            f'<th style="{s_th}color:{COLOR_A};">{label_a}</th>'
                            f'<th style="{s_th}color:{COLOR_B};">{label_b}</th>'
                            f'</tr></thead><tbody>')
                for lbl, va, vb in hk_rows:
                    html_hk += (f'<tr><td style="{s_td0}">{lbl}</td>'
                                f'<td style="{s_td}color:{COLOR_A};">{va:.1f}</td>'
                                f'<td style="{s_td}color:{COLOR_B};">{vb:.1f}</td></tr>')
                html_hk += "</tbody></table>"

            if html_hk:
                st.markdown(
                    '<div style="display:flex;gap:56px;flex-wrap:wrap;align-items:flex-start;">'
                    + html_stats + html_hk + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(html_stats, unsafe_allow_html=True)

            _PROB_ORDER_C = {"NK": 0, "NK3": 1, "QUBO": 2}
            summary_rows.sort(key=lambda r: (_PROB_ORDER_C.get(r["_prob"], 99), r["_dim"], r["_t"]))

            has_hamming = any(r.get(col_hamming_a) is not None or r.get(col_hamming_b) is not None for r in summary_rows)
            data_cols_c = (
                ["Instance", col_score_a, col_score_b, "Gap %"]
                + ([col_hamming_a, col_hamming_b] if has_hamming else [])
                + [col_rank_a, col_rank_b, "Résultat"]
            )
            # couleur par colonne
            col_colors = {
                col_score_a: COLOR_A, col_rank_a: COLOR_A, col_hamming_a: COLOR_A,
                col_score_b: COLOR_B, col_rank_b: COLOR_B, col_hamming_b: COLOR_B,
            }

            th_base = "padding:8px 28px;border-bottom:2px solid #ccc;white-space:nowrap;font-size:19px;"
            td_l    = "padding:7px 28px;border-bottom:1px solid #f0f0f0;white-space:nowrap;font-size:18px;"
            td_r    = "padding:7px 28px;border-bottom:1px solid #f0f0f0;white-space:nowrap;font-size:18px;text-align:right;"
            td_sep  = "padding:18px 10px 5px;font-weight:bold;font-size:19px;color:#333;border-top:2px solid #bbb;"

            active_probs_c = [p for p in ["NK", "NK3", "QUBO"] if any(r["_prob"] == p for r in summary_rows)]

            for prob in active_probs_c:
                prob_rows = [r for r in summary_rows if r["_prob"] == prob]
                st.markdown(f"## {prob}")
                ncols = len(data_cols_c)
                html = '<table style="border-collapse:collapse;">'
                html += "<thead><tr>"
                for col in data_cols_c:
                    color = col_colors.get(col, "#333")
                    align = "left" if col == "Instance" else "right"
                    html += (f'<th style="{th_base}text-align:{align};color:{color};">'
                             f'{col}</th>')
                html += "</tr></thead><tbody>"
                current_dim = None
                for r in prob_rows:
                    if r["_dim"] != current_dim:
                        current_dim = r["_dim"]
                        html += f'<tr><td colspan="{ncols}" style="{td_sep}">DIM {current_dim}</td></tr>'
                    html += f'<tr><td style="{td_l}">{r["Instance"]}</td>'
                    for col in data_cols_c[1:]:
                        val = r.get(col)
                        if val is None:
                            cell = ""
                        elif col in (col_score_a, col_score_b):
                            cell = f"{val:.4f}"
                        elif col in (col_hamming_a, col_hamming_b):
                            cell = f"{val:.2f}"
                        elif col == "Gap %":
                            cell = f"{val:+.2f}%"
                        elif col in (col_rank_a, col_rank_b):
                            cell = str(int(val))
                        else:
                            cell = str(val)
                        html += f'<td style="{td_r}">{cell}</td>'
                    html += "</tr>"
                html += "</tbody></table>"
                st.markdown(html, unsafe_allow_html=True)

    # ── Convergence moyenne par famille ──────────────────────────────────────
    if cfg_a and cfg_b:
        # (cfg, problem) -> (mean Series, std Series), indexed by actual step values
        _conv_data: dict = {}
        for _cfg in [cfg_a, cfg_b]:
            for _prob in ["NK", "NK3", "QUBO"]:
                _series = []
                for _inst in instances:
                    _mi = INSTANCE_RE.match(_inst)
                    if not _mi or _mi.group("problem") != _prob:
                        continue
                    _df = load_curve(_cfg, _inst)
                    if _df is None or "best_fitness" not in _df.columns:
                        continue
                    _step_col = _df["step"] if "step" in _df.columns else pd.Series(range(len(_df)))
                    _series.append(
                        pd.Series(np.abs(_df["best_fitness"].values),
                                  index=_step_col.values)
                    )
                if _series:
                    _combined = pd.concat(_series, axis=1)
                    _conv_data[(_cfg, _prob)] = (
                        _combined.mean(axis=1),
                        _combined.std(axis=1).fillna(0),
                    )

        _active_probs = [p for p in ["NK", "NK3", "QUBO"]
                         if (_conv_data.get((cfg_a, p)) or _conv_data.get((cfg_b, p)))]
        if _active_probs:
            st.divider()
            st.markdown("#### 📈 Convergence moyenne par famille")
            fig_conv = make_subplots(
                rows=1, cols=len(_active_probs),
                subplot_titles=_active_probs,
                horizontal_spacing=0.08,
            )
            for _ci, _prob in enumerate(_active_probs, 1):
                for _cfg, _lbl, _color in [(cfg_a, label_a, COLOR_A),
                                           (cfg_b, label_b, COLOR_B)]:
                    if (_cfg, _prob) not in _conv_data:
                        continue
                    _mean, _std = _conv_data[(_cfg, _prob)]
                    _x = _mean.index.tolist()
                    _y = _mean.tolist()
                    _yu = (_mean + _std).tolist()
                    _yl = (_mean - _std).tolist()
                    # Bande ±std
                    fig_conv.add_trace(go.Scatter(
                        x=_x + _x[::-1],
                        y=_yu + _yl[::-1],
                        fill="toself",
                        fillcolor=_color,
                        opacity=0.12,
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                        legendgroup=_lbl,
                    ), row=1, col=_ci)
                    # Courbe moyenne
                    fig_conv.add_trace(go.Scatter(
                        x=_x, y=_y,
                        mode="lines",
                        name=_lbl,
                        line=dict(color=_color, width=2),
                        showlegend=(_ci == 1),
                        legendgroup=_lbl,
                        hovertemplate=f"{_lbl}: %{{y:.4f}}<extra>{_prob}</extra>",
                    ), row=1, col=_ci)
                fig_conv.update_yaxes(title_text="Score", row=1, col=_ci,
                                      gridcolor="#ebebeb", zeroline=False)
                fig_conv.update_xaxes(title_text="Évaluations", row=1, col=_ci,
                                      gridcolor="#ebebeb")
            fig_conv.update_layout(
                height=380,
                margin=dict(t=45, b=20, l=10, r=10),
                paper_bgcolor="white",
                plot_bgcolor="white",
                legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18),
                hovermode="x unified",
            )
            for _ann in fig_conv.layout.annotations:
                _ann.update(font=dict(size=13, color="#333"))
            st.plotly_chart(fig_conv, use_container_width=True, key="conv_fig")

    for pair_start in range(0, len(instances), 2):
        pair = instances[pair_start:pair_start + 2]
        st.divider()
        cols = st.columns(len(pair))
        for col, instance in zip(cols, pair):
            with col:
                st.markdown(f"##### {instance}")

                m = INSTANCE_RE.match(instance)
                problem, dim, t = m.group("problem"), int(m.group("dim")), int(m.group("t"))

                curve_a  = load_curve(cfg_a, instance)
                curve_b  = load_curve(cfg_b, instance)
                scores_a = load_raw_scores(cfg_a, instance)
                scores_b = load_raw_scores(cfg_b, instance)

                ranking = load_ranking(problem, dim, t) if show_top5 else None
                top5 = []
                if ranking is not None:
                    for _, row in ranking.nlargest(5, "score").iterrows():
                        top5.append((row["name_algo"], float(row["score"])))

                has_curve   = curve_a is not None or curve_b is not None
                has_hamming = (
                    (curve_a is not None and "avg_hamming" in curve_a.columns) or
                    (curve_b is not None and "avg_hamming" in curve_b.columns)
                )
                has_box     = scores_a is not None or scores_b is not None

                if not has_curve and not has_hamming and not has_box:
                    st.warning("Pas de données pour cette instance.")
                    continue

                # Assign subplot rows sequentially
                row_idx = 1
                curve_row = hamming_row = box_row = None
                if has_curve:
                    curve_row = row_idx; row_idx += 1
                if has_hamming:
                    hamming_row = row_idx; row_idx += 1
                if has_box:
                    box_row = row_idx; row_idx += 1

                n_rows = row_idx - 1
                titles = (
                    (["Convergence"]         if has_curve   else []) +
                    (["Hamming (diversité)"] if has_hamming else []) +
                    (["Distribution finale"] if has_box     else [])
                )
                row_heights = (
                    [0.40, 0.25, 0.35] if n_rows == 3 else
                    [0.60, 0.40]       if n_rows == 2 else
                    [1.0]
                )

                fig = make_subplots(
                    rows=n_rows, cols=1, subplot_titles=titles,
                    row_heights=row_heights,
                    vertical_spacing=0.10,
                )

                if has_curve:
                    ref = curve_a if curve_a is not None else curve_b
                    for curve, lbl, color in [(curve_a, label_a, COLOR_A), (curve_b, label_b, COLOR_B)]:
                        if curve is None:
                            continue
                        fig.add_trace(go.Scatter(
                            x=curve["step"], y=_norm(curve["best_fitness"]),
                            name=lbl, line=dict(color=color, width=2), mode="lines",
                        ), row=curve_row, col=1)
                    for j, (algo, score) in enumerate(top5):
                        fig.add_trace(go.Scatter(
                            x=[ref["step"].iloc[0], ref["step"].iloc[-1]], y=[score, score],
                            name=algo, mode="lines",
                            line=dict(dash="dash", color=COLORS[(j + 2) % len(COLORS)], width=1),
                        ), row=curve_row, col=1)
                    fig.update_xaxes(title_text="Évaluations", row=curve_row, col=1)
                    fig.update_yaxes(title_text="Score",        row=curve_row, col=1)

                if has_hamming:
                    for curve, lbl, color in [(curve_a, label_a, COLOR_A), (curve_b, label_b, COLOR_B)]:
                        if curve is None or "avg_hamming" not in curve.columns:
                            continue
                        fig.add_trace(go.Scatter(
                            x=curve["step"], y=curve["avg_hamming"],
                            name=lbl, line=dict(color=color, width=2, dash="dot"),
                            mode="lines", showlegend=False,
                        ), row=hamming_row, col=1)
                    fig.update_xaxes(title_text="Évaluations",  row=hamming_row, col=1)
                    fig.update_yaxes(title_text="Hamming moy.", row=hamming_row, col=1)

                if has_box:
                    for scores, lbl, color in [(scores_a, label_a, COLOR_A), (scores_b, label_b, COLOR_B)]:
                        if scores is None:
                            continue
                        fig.add_trace(go.Box(
                            y=scores, name=lbl,
                            marker_color=color, boxpoints="outliers", showlegend=False,
                        ), row=box_row, col=1)
                    for j, (algo, score) in enumerate(top5):
                        fig.add_hline(
                            y=score, row=box_row, col=1,
                            line=dict(dash="dash", color=COLORS[(j + 2) % len(COLORS)], width=1),
                            annotation_text=algo, annotation_position="bottom right",
                        )
                    fig.update_yaxes(title_text="Score final", row=box_row, col=1)

                fig.update_layout(
                    height={3: 960, 2: 720, 1: 440}.get(n_rows, 720),
                    margin=dict(r=100, t=40, b=10),
                    legend=dict(orientation="v", x=1.02, y=1),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"comp_fig_{sel_idx}_{instance}")


@st.fragment
def tab_favoris(all_df: pd.DataFrame, sorted_instances: list) -> None:
    favs        = load_favorites()
    all_configs = all_df["config"].tolist()
    sel_idx     = st.session_state.get("fav_sel_idx")

    # ── Vue liste (dossiers) ──────────────────────────────────────────────────
    if sel_idx is None or sel_idx >= len(favs):
        h1, h2 = st.columns([6, 1])
        h1.markdown("#### ⭐ Dossiers favoris")
        if h2.button("+ Nouveau", key="fav_new", use_container_width=True):
            favs.append({"name": f"Favoris {len(favs) + 1}",
                         "configs": [], "aliases": {}, "instances": []})
            save_favorites(favs)
            st.session_state["fav_sel_idx"] = len(favs) - 1
            st.rerun()

        if not favs:
            st.info("Aucun dossier. Cliquez sur « + Nouveau » pour commencer.")
            return

        grid = st.columns(3)
        for i, fav in enumerate(favs):
            with grid[i % 3]:
                with st.container(border=True):
                    t1, t2 = st.columns([5, 1])
                    if t1.button(f"⭐ {fav['name']}", key=f"fav_sel_{i}", use_container_width=True):
                        st.session_state["fav_sel_idx"] = i
                        st.rerun()
                    if t2.button("🗑", key=f"fav_del_{i}"):
                        favs.pop(i)
                        save_favorites(favs)
                        st.rerun()
                    cfgs_card = fav.get("configs", [])
                    aliases_card = fav.get("aliases", {})
                    n_cfg = len(cfgs_card)
                    if n_cfg:
                        st.caption(f"{n_cfg} config{'s' if n_cfg > 1 else ''}")
                    n_inst = len(fav.get("instances", []))
                    if n_inst:
                        st.caption(f"{n_inst} instance{'s' if n_inst > 1 else ''}")
                    if cfgs_card:
                        st.code(cfgs_card[0], language=None)
        return

    # ── Vue détail ────────────────────────────────────────────────────────────
    fav = favs[sel_idx]

    hd1, hd2 = st.columns([1, 8])
    if hd1.button("← Retour", key="fav_back"):
        st.session_state.pop("fav_sel_idx", None)
        st.rerun()
    hd2.markdown(f"#### ⭐ {fav['name']}")

    new_name = st.text_input("Nom du dossier", value=fav["name"], key=f"fav_name_{sel_idx}")
    if new_name and new_name != fav["name"]:
        favs[sel_idx]["name"] = new_name
        save_favorites(favs)

    st.divider()
    st.markdown("##### Configs sauvegardées")

    # Ajouter une config
    existing_cfgs = fav.get("configs", [])
    available_cfgs = [c for c in all_configs if c not in existing_cfgs]
    ac1, ac2 = st.columns([4, 1])
    add_cfg = ac1.selectbox("Ajouter une config", [""] + available_cfgs, key=f"fav_add_{sel_idx}")
    if ac2.button("Ajouter", key=f"fav_add_btn_{sel_idx}", use_container_width=True, disabled=not add_cfg):
        if add_cfg and add_cfg not in favs[sel_idx]["configs"]:
            favs[sel_idx]["configs"].append(add_cfg)
            save_favorites(favs)
            st.rerun()

    configs = fav.get("configs", [])
    aliases = fav.get("aliases", {})

    if not configs:
        st.info("Aucune config ajoutée.")
        return

    for cfg in list(configs):
        row_df = all_df[all_df["config"] == cfg]
        default_lbl = _config_label(row_df.iloc[0]) if not row_df.empty else cfg[-40:]
        r1, r2, r3 = st.columns([3, 3, 1])
        r1.code(cfg, language=None)
        alias = r2.text_input(
            "Alias", value=aliases.get(cfg, ""),
            key=f"fav_alias_{sel_idx}_{cfg}",
            placeholder=default_lbl,
            label_visibility="collapsed",
        )
        if alias != aliases.get(cfg, ""):
            favs[sel_idx]["aliases"][cfg] = alias
            save_favorites(favs)
        if r3.button("✕", key=f"fav_rm_{sel_idx}_{cfg}", use_container_width=True):
            favs[sel_idx]["configs"].remove(cfg)
            favs[sel_idx]["aliases"].pop(cfg, None)
            save_favorites(favs)
            st.rerun()

    def _fav_lbl(cfg: str) -> str:
        al = aliases.get(cfg, "")
        if al:
            return al
        row_df = all_df[all_df["config"] == cfg]
        return _config_label(row_df.iloc[0]) if not row_df.empty else cfg[-30:]

    st.divider()

    # ── Sélection d'instances ─────────────────────────────────────────────────
    saved_insts = [i for i in fav.get("instances", []) if i in sorted_instances]
    inst_key    = f"fav_inst_{sel_idx}"

    nk_insts   = [i for i in sorted_instances if INSTANCE_RE.match(i).group("problem") == "NK"]
    nk3_insts  = [i for i in sorted_instances if INSTANCE_RE.match(i).group("problem") == "NK3"]
    qubo_insts = [i for i in sorted_instances if INSTANCE_RE.match(i).group("problem") == "QUBO"]

    b1, b2, b3 = st.columns(3)
    if b1.button("+ Tout NK",   key=f"fav_nk_{sel_idx}",   use_container_width=True):
        cur = list(st.session_state.get(inst_key, saved_insts))
        st.session_state[inst_key] = sorted(set(cur) | set(nk_insts),   key=sorted_instances.index)
    if b2.button("+ Tout NK3",  key=f"fav_nk3_{sel_idx}",  use_container_width=True):
        cur = list(st.session_state.get(inst_key, saved_insts))
        st.session_state[inst_key] = sorted(set(cur) | set(nk3_insts),  key=sorted_instances.index)
    if b3.button("+ Tout QUBO", key=f"fav_qubo_{sel_idx}", use_container_width=True):
        cur = list(st.session_state.get(inst_key, saved_insts))
        st.session_state[inst_key] = sorted(set(cur) | set(qubo_insts), key=sorted_instances.index)

    instances = st.multiselect(
        "Instances", sorted_instances, default=saved_insts, key=inst_key,
    )

    if instances != saved_insts:
        favs[sel_idx]["instances"] = instances
        save_favorites(favs)

    if not instances:
        st.info("Sélectionnez au moins une instance.")
        return

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    _PROB_ORDER = {"NK": 0, "NK3": 1, "QUBO": 2}

    _LETTERS = "ABCDEFGHIJ"
    config_shorts = {cfg: _LETTERS[i] for i, cfg in enumerate(configs)}

    summary_rows = []
    for inst in instances:
        m_inst = INSTANCE_RE.match(inst)
        prob, dim, t = m_inst.group("problem"), int(m_inst.group("dim")), int(m_inst.group("t"))
        row: dict = {"Instance": inst, "_prob": prob, "_dim": dim, "_t": t}
        for cfg in configs:
            short = config_shorts[cfg]
            curve = load_curve(cfg, inst)
            if curve is not None:
                score = abs(float(curve["best_fitness"].iloc[-1]))
                rank  = _get_rank(prob, dim, t, score)
                row[f"Score {short}"] = round(score, 4)
                row[f"Rang {short}"]  = rank
                ham = float(curve["avg_hamming"].iloc[-1]) if "avg_hamming" in curve.columns else None
                row[f"_ham_{short}"]  = round(ham, 2) if ham is not None else None
                l1  = float(curve["avg_l1"].iloc[-1]) if "avg_l1" in curve.columns else None
                row[f"_l1_{short}"]   = round(l1, 2) if l1 is not None else None
            else:
                row[f"Score {short}"] = None
                row[f"Rang {short}"]  = None
                row[f"_ham_{short}"]  = None
                row[f"_l1_{short}"]   = None
        summary_rows.append(row)

    summary_rows.sort(key=lambda r: (_PROB_ORDER.get(r["_prob"], 99), r["_dim"], r["_t"]))

    st.markdown("#### 📊 Résumé")

    # Légende #1 = …
    for cfg in configs:
        st.caption(f"{config_shorts[cfg]} = {_fav_lbl(cfg)}")

    # Ligne 1 : Rank moy + Top 1 côte à côte, par config
    for cfg in configs:
        lbl   = _fav_lbl(cfg)
        short = config_shorts[cfg]
        ranks = [r[f"Rang {short}"] for r in summary_rows if r.get(f"Rang {short}") is not None]
        top1  = sum(1 for r in summary_rows if r.get(f"Rang {short}") == 1)
        mc1, mc2 = st.columns(2)
        mc1.metric(f"Rank moy — {lbl}", f"{sum(ranks)/len(ranks):.1f}" if ranks else "—")
        mc2.metric(f"Top 1 — {lbl}", top1)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Hamming / L1 moyens par dimension (64 / 128 / 256 / 512) ──────────────
    h_th_l = "padding:8px 20px;text-align:left;border-bottom:2px solid #ccc;white-space:nowrap;font-size:19px;"
    h_th   = "padding:8px 20px;text-align:right;border-bottom:2px solid #ccc;white-space:nowrap;font-size:19px;"
    h_td_l = "padding:7px 20px;border-bottom:1px solid #f0f0f0;white-space:nowrap;font-size:18px;"
    h_td_r = "padding:7px 20px;border-bottom:1px solid #f0f0f0;white-space:nowrap;font-size:18px;text-align:right;"

    _tbl_title = "font-weight:bold;font-size:17px;margin-bottom:4px;"

    def _table_html(title: str, first_hdr: str, row_defs: list) -> str:
        """row_defs = [(label, [cell_str_par_config])] ; renvoie titre + table, ou ''."""
        if not row_defs:
            return ""
        html = f'<div style="{_tbl_title}">{title}</div>'
        html += '<table style="border-collapse:collapse;margin-bottom:4px;"><thead><tr>'
        html += f'<th style="{h_th_l}">{first_hdr}</th>'
        for cfg in configs:
            html += f'<th style="{h_th}">{config_shorts[cfg]}</th>'
        html += "</tr></thead><tbody>"
        for label, cells in row_defs:
            html += f'<tr><td style="{h_td_l}">{label}</td>'
            for cell in cells:
                html += f'<td style="{h_td_r}">{cell}</td>'
            html += "</tr>"
        html += "</tbody></table>"
        return f'<div>{html}</div>'

    def _dim_html(title: str, prefix: str, fmt: str) -> str:
        rows = []
        for dim in [64, 128, 256, 512]:
            cells, has = [], False
            for cfg in configs:
                short = config_shorts[cfg]
                vals = [r[f"{prefix}{short}"] for r in summary_rows
                        if r["_dim"] == dim and r.get(f"{prefix}{short}") is not None]
                if vals:
                    has = True
                cells.append(format(sum(vals) / len(vals), fmt) if vals else "")
            if has:
                rows.append((f"DIM {dim}", cells))
        return _table_html(title, "Dim", rows)

    def _k_html(title: str, prefix: str, fmt: str, prob: str) -> str:
        klbl = "t" if prob == "QUBO" else "K"
        rows = []
        for k in sorted({r["_t"] for r in summary_rows if r["_prob"] == prob}):
            cells, has = [], False
            for cfg in configs:
                short = config_shorts[cfg]
                vals = [r[f"{prefix}{short}"] for r in summary_rows
                        if r["_prob"] == prob and r["_t"] == k and r.get(f"{prefix}{short}") is not None]
                if vals:
                    has = True
                cells.append(format(sum(vals) / len(vals), fmt) if vals else "")
            if has:
                rows.append((f"{klbl} {k}", cells))
        return _table_html(title, klbl, rows)

    _sep = ('<div style="align-self:stretch;border-left:3px solid #2ca02c;'
            'margin:0 4px;"></div>')

    def _flex_groups(groups: list) -> None:
        # groups = liste de listes de fragments HTML ; séparateur vert entre groupes
        clean = [[f for f in g if f] for g in groups]
        clean = [g for g in clean if g]
        if not clean:
            return
        inner = _sep.join("".join(g) for g in clean)
        st.markdown(
            '<div style="display:flex;gap:56px;flex-wrap:wrap;align-items:flex-start;'
            'margin-bottom:22px;">' + inner + "</div>",
            unsafe_allow_html=True,
        )

    # Tous les tableaux sur une ligne, séparateur vert entre chaque type
    _flex_groups([
        [_dim_html("Hamming moyen / dim", "_ham_", ".1f"),
         _dim_html("L1 moyen / dim",      "_l1_",  ".2f")],
        [_k_html(f"Hamming / K — {p}", "_ham_", ".1f", p) for p in ["NK", "NK3", "QUBO"]],
        [_k_html(f"L1 / K — {p}",      "_l1_",  ".2f", p) for p in ["NK", "NK3", "QUBO"]],
    ])

    # Un tableau HTML par problème — colonnes Score / Rang / Hamming par config
    col_specs: list[tuple[str, str, str]] = []  # (entête, clé, type)
    for cfg in configs:
        s = config_shorts[cfg]
        col_specs.append((f"Score {s}",   f"Score {s}",  "score"))
        col_specs.append((f"Rang {s}",    f"Rang {s}",   "rank"))
        col_specs.append((f"Hamming {s}", f"_ham_{s}",   "ham"))

    th = "padding:8px 28px;text-align:right;border-bottom:2px solid #ccc;white-space:nowrap;font-size:19px;"
    th_l = "padding:8px 28px;text-align:left;border-bottom:2px solid #ccc;white-space:nowrap;font-size:19px;"
    td_l = "padding:7px 28px;border-bottom:1px solid #f0f0f0;white-space:nowrap;font-size:18px;"
    td_r = "padding:7px 28px;border-bottom:1px solid #f0f0f0;white-space:nowrap;font-size:18px;text-align:right;"
    td_sep = "padding:18px 10px 5px;font-weight:bold;font-size:19px;color:#333;border-top:2px solid #bbb;"

    active_probs = [p for p in ["NK", "NK3", "QUBO"] if any(r["_prob"] == p for r in summary_rows)]
    prob_cols = st.columns(len(active_probs)) if active_probs else []

    for col_widget, prob in zip(prob_cols, active_probs):
        prob_rows = [r for r in summary_rows if r["_prob"] == prob]
        with col_widget:
            st.markdown(f"##### {prob}")

            ncols = 1 + len(col_specs)
            html = '<table style="border-collapse:collapse;">'
            html += f'<thead><tr><th style="{th_l}">Instance</th>'
            for hdr, _, _ in col_specs:
                html += f'<th style="{th}">{hdr}</th>'
            html += "</tr></thead><tbody>"

            current_dim = None
            for r in prob_rows:
                if r["_dim"] != current_dim:
                    current_dim = r["_dim"]
                    html += f'<tr><td colspan="{ncols}" style="{td_sep}">DIM {current_dim}</td></tr>'
                html += f'<tr><td style="{td_l}">{r["Instance"]}</td>'
                for _, key, kind in col_specs:
                    val = r.get(key)
                    if val is None:
                        cell = ""
                    elif kind == "score":
                        cell = f"{val:.4f}"
                    elif kind == "rank":
                        cell = str(int(val))
                    else:  # ham
                        cell = f"{val:.2f}"
                    html += f'<td style="{td_r}">{cell}</td>'
                html += "</tr>"

            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)

    # ── Courbes ──────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📈 Courbes")

    o1, o2, o3 = st.columns(3)
    fav_top5    = o1.checkbox("Concurrents top 5 (courbes + boxplot)", value=True, key=f"fav_top5_{sel_idx}")
    fav_hamming = o2.checkbox("Hamming (diversité)",     value=True,  key=f"fav_ham_{sel_idx}")
    fav_box     = o3.checkbox("Distribution finale",     value=True,  key=f"fav_box_{sel_idx}")

    # Sélecteur d'instance — colonne par problème, sous-groupe par dimension
    st.markdown("**Instance à visualiser**")
    st.markdown(
        "<style>"
        '[data-testid="stPills"] button,'
        '[data-testid="stBaseButton-pills"],'
        '[data-testid="stBaseButton-pillsActive"]{'
        'padding:0.4rem 0.95rem !important;border-radius:10px !important;'
        'min-height:2.2rem !important;}'
        '[data-testid="stPills"] button p,'
        '[data-testid="stBaseButton-pills"] p,'
        '[data-testid="stBaseButton-pillsActive"] p{'
        'font-size:1.15rem !important;font-weight:600 !important;}'
        ".fav-curve-hdr{font-size:1.5rem;font-weight:700;margin:6px 0 2px;}"
        ".fav-curve-dim{font-size:1.2rem;font-weight:600;color:#555;margin:10px 0 2px;}"
        "</style>",
        unsafe_allow_html=True,
    )
    _iprob = lambda i: INSTANCE_RE.match(i).group("problem")
    _idim  = lambda i: int(INSTANCE_RE.match(i).group("dim"))
    _it    = lambda i: int(INSTANCE_RE.match(i).group("t"))
    probs_curve = [p for p in ["NK", "NK3", "QUBO"] if any(_iprob(i) == p for i in instances)]
    sel_key = f"fav_curve_sel_{sel_idx}"

    # toutes les clés de puces, connues d'avance → utilisables dans le callback
    all_pick_keys = [
        f"fav_curve_pick_{sel_idx}_{p}_{d}"
        for p in probs_curve
        for d in sorted({_idim(i) for i in instances if _iprob(i) == p})
    ]

    def _fav_pick_cb(changed: str):
        v = st.session_state.get(changed)
        st.session_state[sel_key] = v or None
        if v:                                   # un clic sélectionne → vide les autres groupes
            for k in all_pick_keys:
                if k != changed:
                    st.session_state[k] = None

    for pcol, p in zip(st.columns(len(probs_curve), gap="large"), probs_curve):
        with pcol:
            st.markdown(f'<div class="fav-curve-hdr">{p}</div>', unsafe_allow_html=True)
            _plbl = "t" if p == "QUBO" else "K"
            p_insts = [i for i in instances if _iprob(i) == p]
            for d in sorted({_idim(i) for i in p_insts}):
                d_insts = sorted((i for i in p_insts if _idim(i) == d), key=_it)
                wkey = f"fav_curve_pick_{sel_idx}_{p}_{d}"
                st.markdown(f'<div class="fav-curve-dim">DIM {d}</div>', unsafe_allow_html=True)
                st.pills(
                    wkey, d_insts, selection_mode="single", key=wkey,
                    format_func=lambda i, _l=_plbl: f"{_l}{_it(i)}",
                    on_change=_fav_pick_cb, args=(wkey,),
                    label_visibility="collapsed",
                )

    sel_curve_inst = st.session_state.get(sel_key)
    if not sel_curve_inst:
        st.info("Sélectionnez une instance ci-dessus pour afficher ses courbes.")
        return

    # ── Placement dans le classement global (cf. build_global_ranking_lines) ──
    _ACCENT = "#2ca02c"

    def _ranking_html(our_score: float, rk_df: pd.DataFrame, label: str) -> str:
        entries = [(str(n), float(s)) for n, s in zip(rk_df["name_algo"], rk_df["score"])]
        entries.append(("__OURS__", our_score))
        entries.sort(key=lambda x: x[1], reverse=True)
        our   = next(i for i, (n, _) in enumerate(entries) if n == "__OURS__")
        total = len(entries)
        if our == 0:                                  # on est #1 → avance sur #2
            gap_txt = f"{our_score - entries[1][1]:+.4f} vs #2" if total > 1 else ""
        else:                                         # sinon → retard sur #1
            gap_txt = f"{our_score - entries[0][1]:+.4f} vs #1"

        head = list(range(min(12, total)))
        tail = [i for i in (our - 1, our, our + 1) if 0 <= i < total and i not in head]
        show, prev = [], None
        for i in head + tail:
            if prev is not None and i > prev + 1:
                show.append(None)          # séparateur "⋯"
            show.append(i)
            prev = i

        th  = ("padding:8px 20px;font-size:15px;font-weight:700;color:#888;"
               "text-transform:uppercase;letter-spacing:.04em;border-bottom:2px solid #e0e0e0;")
        td  = "padding:10px 20px;font-size:19px;border-bottom:1px solid #f0f0f0;"
        tdn = td + "white-space:nowrap;"                     # # et score : pas de retour
        tdr = tdn + "text-align:right;font-variant-numeric:tabular-nums;"
        td  = td + "word-break:break-word;line-height:1.25;"  # nom : peut passer à la ligne

        rows = ""
        for i in show:
            if i is None:
                rows += (f'<tr><td colspan="3" style="{tdn}text-align:center;color:#bbb;">⋯</td></tr>')
                continue
            name, score = entries[i]
            mine = (i == our)
            disp = "Nous" if mine else name
            rstyle = (f'background:rgba(44,160,44,.12);' if mine else "")
            nstyle = (f'color:{_ACCENT};font-weight:800;' if mine else "font-weight:500;")
            marker = f'<span style="color:{_ACCENT};font-weight:800;">▸ </span>' if mine else ""
            rows += (
                f'<tr style="{rstyle}">'
                f'<td style="{tdn}color:#999;">{marker}#{i + 1}</td>'
                f'<td style="{td}{nstyle}">{disp}</td>'
                f'<td style="{tdr}{nstyle}">{score:.4f}</td>'
                f"</tr>"
            )

        return (
            f'<div style="border:1px solid #e6e6e6;border-radius:12px;overflow:hidden;'
            f'margin-bottom:16px;">'
            f'<div style="padding:12px 20px;background:#fafafa;border-bottom:1px solid #eee;">'
            f'<div style="font-weight:700;font-size:18px;">{label}</div>'
            f'<div style="font-size:17px;margin-top:3px;">'
            f'rang <b style="color:{_ACCENT};font-size:26px;">{our + 1}</b>'
            f'<span style="color:#999;"> / {total}</span>'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;score <b>{our_score:.4f}</b>'
            + (f'<span style="color:#999;"> ({gap_txt})</span>' if gap_txt else "")
            + f"</div></div>"
            f'<table style="border-collapse:collapse;width:100%;">'
            f'<thead><tr><th style="{th}">#</th><th style="{th}">Algorithme</th>'
            f'<th style="{th}text-align:right;">Score</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>"
        )

    for instance in [sel_curve_inst]:
        _pl, _cv = st.columns([3, 7], gap="large")

        with _pl:
            _rm = INSTANCE_RE.match(instance)
            _rk_args = (_rm.group("problem"), int(_rm.group("dim")), int(_rm.group("t")))
            _rk_df = load_ranking(*_rk_args)
            if _rk_df is None or _rk_df.empty:
                st.caption("Pas de classement pour cette instance.")
            else:
                st.markdown(f"##### 🏅 Placement — {instance}")
                for _cfg in configs:
                    _cvcur = load_curve(_cfg, instance)
                    if _cvcur is None or _cvcur.empty:
                        continue
                    _our = abs(float(_cvcur["best_fitness"].iloc[-1]))
                    st.markdown(_ranking_html(_our, _rk_df, _fav_lbl(_cfg)),
                                unsafe_allow_html=True)

        with _cv:
            with st.container():

                m_i = INSTANCE_RE.match(instance)
                prob, dim, t = m_i.group("problem"), int(m_i.group("dim")), int(m_i.group("t"))

                curves  = {cfg: load_curve(cfg, instance) for cfg in configs}
                scores  = {cfg: load_raw_scores(cfg, instance) for cfg in configs} if fav_box else {}

                top5 = []
                comp_curves: dict = {}   # algo -> (x, y)
                comp_boxes:  dict = {}   # algo -> stats dict
                if fav_top5:
                    ranking = load_ranking(prob, dim, t)
                    if ranking is not None:
                        for _, rk in ranking.nlargest(5, "score").iterrows():
                            algo = rk["name_algo"]
                            top5.append((algo, float(rk["score"])))
                            cc = load_competitor_curve(algo, prob, dim, t)
                            if cc is not None:
                                comp_curves[algo] = cc
                            if fav_box:
                                cb = load_competitor_box(algo, prob, dim, t)
                                if cb is not None:
                                    comp_boxes[algo] = cb

                has_curve   = any(c is not None for c in curves.values())
                has_hamming = fav_hamming and any(
                    c is not None and "avg_hamming" in c.columns for c in curves.values()
                )
                has_box     = fav_box and (
                    any(s is not None for s in scores.values()) or bool(comp_boxes)
                )

                if not has_curve and not has_hamming and not has_box:
                    st.warning("Pas de données pour cette instance.")
                    continue

                row_idx = 1
                curve_row = hamming_row = box_row = None
                if has_curve:
                    curve_row = row_idx; row_idx += 1
                if has_hamming:
                    hamming_row = row_idx; row_idx += 1
                if has_box:
                    box_row = row_idx; row_idx += 1
                n_rows = row_idx - 1

                titles = (
                    (["Convergence"]         if has_curve   else []) +
                    (["Hamming (diversité)"] if has_hamming else []) +
                    (["Distribution finale"] if has_box     else [])
                )
                row_heights = (
                    [0.40, 0.25, 0.35] if n_rows == 3 else
                    [0.60, 0.40]       if n_rows == 2 else
                    [1.0]
                )

                fig = make_subplots(
                    rows=n_rows, cols=1, subplot_titles=titles,
                    row_heights=row_heights, vertical_spacing=0.10,
                )

                ref = next((c for c in curves.values() if c is not None), None)

                if has_curve:
                    for i, cfg in enumerate(configs):
                        curve = curves[cfg]
                        if curve is None:
                            continue
                        fig.add_trace(go.Scatter(
                            x=curve["step"], y=_norm(curve["best_fitness"]),
                            name=_fav_lbl(cfg),
                            line=dict(color=COLORS[i % len(COLORS)], width=2),
                            mode="lines",
                        ), row=curve_row, col=1)
                    for j, (algo, score) in enumerate(top5):
                        ccol = COLORS[(len(configs) + j) % len(COLORS)]
                        cc = comp_curves.get(algo)
                        if cc is not None:
                            cx, cy = cc
                            fig.add_trace(go.Scatter(
                                x=cx, y=cy, name=algo, mode="lines",
                                line=dict(dash="dash", color=ccol, width=1.5),
                                opacity=0.9,
                            ), row=curve_row, col=1)
                        elif ref is not None:
                            fig.add_trace(go.Scatter(
                                x=[ref["step"].iloc[0], ref["step"].iloc[-1]], y=[score, score],
                                name=algo, mode="lines",
                                line=dict(dash="dot", color=ccol, width=1),
                            ), row=curve_row, col=1)
                    fig.update_xaxes(title_text="Évaluations", row=curve_row, col=1)
                    fig.update_yaxes(title_text="Score",       row=curve_row, col=1)

                if has_hamming:
                    for i, cfg in enumerate(configs):
                        curve = curves[cfg]
                        if curve is None or "avg_hamming" not in curve.columns:
                            continue
                        fig.add_trace(go.Scatter(
                            x=curve["step"], y=curve["avg_hamming"],
                            name=_fav_lbl(cfg),
                            line=dict(color=COLORS[i % len(COLORS)], width=2, dash="dot"),
                            mode="lines", showlegend=False,
                        ), row=hamming_row, col=1)
                    fig.update_xaxes(title_text="Évaluations",  row=hamming_row, col=1)
                    fig.update_yaxes(title_text="Hamming moy.", row=hamming_row, col=1)

                if has_box:
                    for i, cfg in enumerate(configs):
                        s = scores.get(cfg)
                        if s is None:
                            continue
                        lbl = _fav_lbl(cfg)
                        fig.add_trace(go.Box(
                            y=s, x=[lbl] * len(s), name=lbl,
                            marker_color=COLORS[i % len(COLORS)],
                            boxpoints="outliers", boxmean=True, showlegend=False,
                        ), row=box_row, col=1)
                    for j, (algo, score) in enumerate(top5):
                        st_ = comp_boxes.get(algo)
                        ccol = COLORS[(len(configs) + j) % len(COLORS)]
                        if st_ is not None:
                            fig.add_trace(go.Box(
                                x=[algo], name=algo, marker_color=ccol, showlegend=False,
                                lowerfence=[st_["low"]], q1=[st_["q1"]], median=[st_["med"]],
                                q3=[st_["q3"]], upperfence=[st_["high"]],
                                mean=[st_["mean"]] if "mean" in st_ else None,
                            ), row=box_row, col=1)
                        else:
                            fig.add_hline(
                                y=score, row=box_row, col=1,
                                line=dict(dash="dot", color=ccol, width=1),
                                annotation_text=algo, annotation_position="bottom right",
                            )
                    fig.update_xaxes(type="category", row=box_row, col=1)
                    fig.update_yaxes(title_text="Score final", row=box_row, col=1)

                fig.update_layout(
                    height={3: 1240, 2: 880, 1: 520}.get(n_rows, 880),
                    margin=dict(r=20, t=40, b=10),
                    legend=dict(orientation="h", yanchor="top", y=-0.12),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"fav_fig_{sel_idx}_{instance}")

        st.divider()
        with st.container():
            st.markdown("#### 🧬 Agents — génotype & rencontre du meilleur")

            def _render_beast(inst, ti_beast, g=None, sco=None, bi=None, key=""):
                """Heatmap du meilleur génotype « the beast ». Si g/sco/bi fournis,
                ajoute le gap de fitness + Hamming vs nos agents."""
                bm = INSTANCE_RE.match(inst)
                beast = (load_beast_best(bm.group("problem"), int(bm.group("dim")),
                                         int(bm.group("t")), int(ti_beast)) if bm else None)
                if beast is None:
                    st.caption(f"« the beast » : aucun résultat rangé pour {inst} "
                               f"(test #{ti_beast}). Vérifie `results/the beast/…` "
                               f"puis `python reorganize.py --apply`.")
                    return
                bg, bfit, bn = beast
                title = (f"Génotype du meilleur trouvé par « the beast »  ·  "
                         f"fitness = {bfit:.4f}  (sur {bn} solutions)")
                bh = None
                if g is not None and sco is not None and bi is not None \
                        and bg.shape[0] == g.shape[1]:
                    bh = [int((bg != g[a]).sum()) for a in range(g.shape[0])]
                    bclose = int(np.argmin(bh))
                    our_fit = abs(float(sco[ti_beast][bi]))
                    gap = bfit - our_fit
                    title += (f"  ·  gap vs notre A{bi} = {gap:+.4f}  ·  "
                              f"Hamming = {bh[bi]} bits")
                fig_b = go.Figure(go.Heatmap(
                    z=bg.reshape(1, -1), x=list(range(bg.shape[0])), y=["beast"],
                    colorscale="Blues", zmin=0, zmax=max(1, int(bg.max())), showscale=False,
                ))
                fig_b.update_layout(title=title, xaxis_title="Position", height=170,
                                    margin=dict(l=90, r=10, t=40, b=10))
                st.plotly_chart(fig_b, use_container_width=True, key=f"fav_dbg_beast_{key}")
                if bh is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Gap de fitness (beast − nous)", f"{gap:+.4f}",
                              help=f"beast {bfit:.4f}  vs  notre meilleur agent A{bi} {our_fit:.4f}")
                    c2.metric(f"Hamming beast ↔ notre A{bi}", f"{bh[bi]} / {bg.shape[0]}")
                    c3.metric("Agent le plus proche du beast", f"A{bclose}",
                              f"{bh[bclose]} bits", delta_color="off")
                    st.caption("Hamming beast ↔ chaque agent : "
                               + " · ".join(f"A{a}={d}" for a, d in enumerate(bh)))
                elif g is not None:
                    st.caption(f"Comparaison agents impossible : N beast = {bg.shape[0]} "
                               f"≠ N run = {g.shape[1]}.")

            def _big_valley_plot(inst, ti_b, g=None, sco=None, bi=None, wi=None, key=""):
                """Nuage des optima tabou (Hamming au best-known × fitness brute),
                avec nos agents en surimpression quand g/sco sont fournis."""
                bm = INSTANCE_RE.match(inst)
                bo = (load_beast_optima(bm.group("problem"), int(bm.group("dim")),
                                        int(bm.group("t")), int(ti_b)) if bm else None)
                if bo is None:
                    return
                bk = bo["best_known"]
                xN = int(bo["best_geno"].shape[0])                    # = N
                fig = go.Figure()
                fig.add_trace(go.Histogram2d(
                    x=bo["ham"], y=bo["fit"], colorscale="Blues",
                    xbins=dict(start=0, end=xN, size=max(1, xN // 70)),
                    nbinsy=70, name="optima tabou",
                    colorbar=dict(title="densité<br>d'optima", thickness=12),
                    hovertemplate="Hamming=%{x:.0f} bits<br>fitness=%{y:.4g}"
                                  "<br>n=%{z}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=[0.0], y=[bk], mode="markers+text", text=["★ best-known"],
                    textposition="top right", cliponaxis=False,
                    marker=dict(symbol="star", size=16, color="#d62728"),
                    name="beast best", hoverinfo="text",
                ))
                if (g is not None and sco is not None
                        and g.shape[1] == bo["best_geno"].shape[0]):
                    bg = bo["best_geno"]
                    hx = [float((g[a] != bg).sum()) for a in range(g.shape[0])]   # Hamming brut
                    hy = [abs(float(sco[ti_b][a])) for a in range(g.shape[0])]
                    cols = ["#1f77b4"] * g.shape[0]
                    if bi is not None and 0 <= bi < len(cols):
                        cols[bi] = "#2ca02c"
                    if wi is not None and 0 <= wi < len(cols):
                        cols[wi] = "#d62728"
                    fig.add_trace(go.Scatter(
                        x=hx, y=hy, mode="markers+text",
                        text=[f"A{a}" for a in range(g.shape[0])],
                        textposition="middle right",
                        marker=dict(size=13, color=cols, line=dict(width=1.5, color="white")),
                        name="nos agents",
                        hovertemplate="%{text}<br>Hamming au best=%{x:.0f} bits"
                                      "<br>fitness=%{y:.4g}<extra></extra>",
                    ))
                fig.update_layout(
                    title=dict(
                        text=(f"Big-valley plot — {inst} (test #{ti_b})  ·  "
                              f"best-known = {bk:.4g}"
                              + (f"  ·  {bo['n_bad']} corrompus exclus" if bo["n_bad"] else "")),
                        y=0.98, yanchor="top",
                    ),
                    xaxis=dict(title="distance de Hamming au best-known (bits)",
                               range=[0, xN], constrain="domain"),
                    yaxis=dict(title="fitness (brute)", automargin=True),
                    height=540, margin=dict(l=55, r=10, t=40, b=70),
                    bargap=0, legend=dict(orientation="h", yanchor="top", y=-0.16, x=0),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"fav_bigvalley_{key}")

                _n = bo["n"]
                _ng = bo["n_global_opt"]
                _pc = lambda v: f"{100 * v / _n:.1f}"
                _lines = [f"<b>{bo['n_distinct']:,} optima distincts</b> trouvés "
                          f"sur {_n:,} restarts."]
                if _ng == 1:
                    _lines.append(
                        f"<b>Optimum global</b> (fitness {bk:.4g}) : 1 seul génotype, "
                        f"atteint par <b>{bo['n_at_best_fit']:,}</b> restarts "
                        f"({_pc(bo['n_at_best_fit'])}&nbsp;%)."
                    )
                else:
                    _lines.append(
                        f"<b>Optima globaux</b> (fitness {bk:.4g}) : "
                        f"<b>{_ng:,}</b> génotypes distincts.  "
                        f"<b>{bo['n_at_best_fit']:,}</b> restarts ({_pc(bo['n_at_best_fit'])}&nbsp;%) "
                        f"finissent sur l'un d'eux ; le plus attractif en capte à lui seul "
                        f"<b>{bo['top_global_count']:,}</b> ({_pc(bo['top_global_count'])}&nbsp;%)."
                    )
                if not np.isclose(bo["mode_fit"], bk):
                    _lines.append(
                        f"⚠️ L'optimum au plus gros bassin n'est <b>pas</b> le meilleur : "
                        f"<b>{bo['mode_count']:,}</b> restarts ({_pc(bo['mode_count'])}&nbsp;%) "
                        f"convergent vers un optimum de fitness {bo['mode_fit']:.4g} "
                        f"(&lt; {bk:.4g})."
                    )
                st.markdown(
                    "<div style='font-size:1.05rem;line-height:1.7'>"
                    + "<br>".join(_lines) + "</div>",
                    unsafe_allow_html=True,
                )

            _dcfg = st.pills(
                "Config", configs, selection_mode="single", format_func=_fav_lbl,
                default=configs[0] if configs else None, key=f"fav_dbg_cfg_{sel_idx}",
            )
            dbg = load_agent_debug(_dcfg) if _dcfg else None
            gkey = f"{instance}/genotype"
            if not _dcfg:
                pass
            elif not dbg or gkey not in dbg:
                st.info("Pas de `agent_debug.npz` pour cette config / instance — "
                        "relancer le grid-search (collecte debug dans main_expe_overall). "
                        "Le génotype « the beast » reste affiché ci-dessous s'il est rangé.")
                _tib = st.number_input("Instance de test (index)", min_value=0, max_value=99,
                                       value=0, step=1, key=f"fav_beast_ti_{sel_idx}")
                _render_beast(instance, int(_tib), key=f"nodbg_{sel_idx}")
                _big_valley_plot(instance, int(_tib), key=f"nodbg_{sel_idx}")
            else:
                geno = dbg[gkey]                              # (nb_inst, A, N)
                epo  = dbg[f"{instance}/epoch"]               # (nb_inst, A)
                sco  = dbg[f"{instance}/score"]               # (nb_inst, A)
                nb_i2, nb_r2, size_pop2, N2, A2 = (int(x) for x in dbg[f"{instance}/meta"])

                ti = 0
                if geno.shape[0] > 1:
                    ti = st.pills(
                        "Instance de test", list(range(geno.shape[0])),
                        selection_mode="single", default=0,
                        format_func=lambda x: f"#{x}",
                        key=f"fav_dbg_ti_{sel_idx}",
                        help=f"meilleur restart (score max) parmi {nb_r2}",
                    )
                    if ti is None:
                        ti = 0

                g = geno[ti]                                  # (A, N)
                agents = [f"A{a}" for a in range(g.shape[0])]

                # Le score stocké est la moyenne des scores réels des restarts, pas la
                # fitness du génotype affiché (qui peut être un agrégat par vote
                # majoritaire) : on la recalcule directement depuis le fichier
                # d'instance pour que génotype affiché et score affiché correspondent
                # vraiment à la même chose (cf. Hamming pairwise).
                _im = INSTANCE_RE.match(instance)
                _fit_recomputed = False
                if _im is not None:
                    _true_fit = [
                        recompute_fitness(_im.group("problem"), int(_im.group("dim")),
                                          int(_im.group("t")), ti, g[a])
                        for a in range(g.shape[0])
                    ]
                    if all(v is not None for v in _true_fit):
                        sco = np.array(sco, dtype=float, copy=True)
                        sco[ti] = _true_fit
                        _fit_recomputed = True
                if not _fit_recomputed:
                    st.caption("⚠️ Fitness recalculée indisponible (fichier d'instance "
                               "introuvable) — les scores affichés restent la moyenne "
                               "des restarts, pas la fitness du génotype montré.")

                _sc = np.abs(np.asarray(sco[ti], float))
                _bi = int(np.argmax(_sc)) if _sc.size else 0
                _wi = int(np.argmin(_sc)) if _sc.size and _sc.min() != _sc.max() else -1

                def _ylab(a: int) -> str:
                    s = f"A{a}  {abs(float(sco[ti][a])):.4f}"
                    if a == _bi:
                        return f'<span style="color:#2ca02c"><b>{s}</b></span>'
                    if a == _wi:
                        return f'<span style="color:#d62728"><b>{s}</b></span>'
                    return s

                fig_g = go.Figure(go.Heatmap(
                    z=g, x=list(range(g.shape[1])), y=agents,
                    colorscale="Blues", zmin=0, zmax=max(1, int(g.max())), showscale=False,
                ))
                fig_g.update_layout(
                    title="Génotype final par agent", xaxis_title="Position",
                    height=600, margin=dict(l=90, r=10, t=40, b=10),
                    yaxis=dict(autorange="reversed", tickmode="array",
                               tickvals=agents, ticktext=[_ylab(a) for a in range(g.shape[0])]),
                )
                st.plotly_chart(fig_g, use_container_width=True, key=f"fav_dbg_g_{sel_idx}")

                # ── Génotype du meilleur trouvé par « the beast » (comparaison) ──
                _render_beast(instance, ti, g=g, sco=sco, bi=_bi, key=str(sel_idx))
                _big_valley_plot(instance, ti, g=g, sco=sco, bi=_bi, wi=_wi, key=str(sel_idx))

                _hcol, _ecol = st.columns(2, gap="large")
                with _hcol:
                    H = (g[:, None, :] != g[None, :, :]).sum(-1)
                    fig_h = go.Figure(go.Heatmap(
                        z=H, x=agents, y=agents, colorscale="Reds",
                        text=H, texttemplate="%{text}", showscale=False,
                    ))
                    fig_h.update_layout(
                        title="Hamming pairwise (solutions finales)",
                        height=560, margin=dict(l=10, r=10, t=40, b=10),
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig_h, use_container_width=True, key=f"fav_dbg_h_{sel_idx}")
                with _ecol:
                    ev = epo[ti] * max(1, size_pop2)
                    x_all   = list(agents)
                    y_all   = [float(v) for v in ev]
                    txt_all = [f"{abs(s):.4f}" for s in sco[ti]]
                    colors  = [COLORS[1]] * len(agents)
                    # couleur par SCORE de l'agent : vert = meilleur, rouge = pire
                    _asc = np.abs(np.asarray(sco[ti], float))
                    _ibest = int(np.argmax(_asc)) if _asc.size else -1
                    _iworst = (int(np.argmin(_asc))
                               if _asc.size and _asc.min() != _asc.max() else -1)
                    if _ibest >= 0:
                        colors[_ibest] = "#2ca02c"
                    if _iworst >= 0:
                        colors[_iworst] = "#d62728"

                    # bar orange à gauche : moment où le concurrent #1 (ou #2 si on est #1) plafonne
                    _am = INSTANCE_RE.match(instance)
                    _rk = load_ranking(_am.group("problem"), int(_am.group("dim")), int(_am.group("t")))
                    if _rk is not None and not _rk.empty:
                        _comps = _rk.sort_values("score", ascending=False).reset_index(drop=True)
                        _ourcv = load_curve(_dcfg, instance)
                        _our_s = (abs(float(_ourcv["best_fitness"].iloc[-1]))
                                  if _ourcv is not None and not _ourcv.empty else None)
                        _tgt = None
                        if (_our_s is not None and len(_comps) > 1
                                and _our_s >= float(_comps.iloc[0]["score"])):
                            _tgt = str(_comps.iloc[1]["name_algo"])       # on est #1 → #2
                        elif len(_comps):
                            _tgt = str(_comps.iloc[0]["name_algo"])       # sinon → #1
                        if _tgt:
                            _pb = _am.group("problem")
                            _pd, _pt = int(_am.group("dim")), int(_am.group("t"))

                            def _moment_98(cx, cy):
                                cx = np.asarray(cx, float); cy = np.asarray(cy, float)
                                lo, hi = cy.min(), cy.max()
                                thr = lo + 0.98 * (hi - lo) if hi > lo else hi
                                return float(cx[int(np.argmax(cy >= thr))])

                            # courbe du concurrent sur CETTE instance de test (moy. sur ses 10 restarts)
                            _cc = load_competitor_instance_curve(_tgt, _pb, _pd, _pt, int(ti))
                            _moment = _cfinal = None
                            if _cc is not None:
                                _moment = _moment_98(_cc[0], _cc[1])
                                _cfinal = float(np.asarray(_cc[1], float)[-1])
                            else:                                   # dossier sans trajectoires par-run
                                _cfinal = load_competitor_instance_final(_tgt, _pb, _pd, _pt, int(ti))
                                _g = load_competitor_curve(_tgt, _pb, _pd, _pt)
                                if _g is not None:
                                    _moment = _moment_98(_g[0], _g[1])   # moment approx. (courbe globale)

                            if _moment is not None:
                                _lbl = _tgt if len(_tgt) <= 16 else _tgt[:15] + "…"
                                x_all   = [_lbl] + x_all
                                y_all   = [_moment] + y_all
                                txt_all = [f"{abs(_cfinal):.4f}" if _cfinal is not None else ""] + txt_all
                                colors  = ["#ff7f0e"] + colors

                    # étiquettes d'axe colorées : meilleur agent en vert, pire en rouge
                    _off = len(x_all) - len(agents)          # 1 si barre concurrent en tête, sinon 0
                    _ticktext = list(x_all)
                    if _ibest >= 0:
                        _ticktext[_off + _ibest] = f'<span style="color:#2ca02c"><b>{agents[_ibest]}</b></span>'
                    if _iworst >= 0:
                        _ticktext[_off + _iworst] = f'<span style="color:#d62728"><b>{agents[_iworst]}</b></span>'

                    fig_e = go.Figure(go.Bar(
                        x=x_all, y=y_all, marker_color=colors,
                        text=txt_all, textposition="outside",
                    ))
                    fig_e.update_layout(
                        title="Moment où chaque agent atteint son meilleur (≈ évaluations)  ·  "
                              "vert = meilleur score · rouge = pire · orange = concurrent #1",
                        yaxis=dict(title="Évaluations", range=[0, _COMPETITOR_BUDGET]),
                        xaxis=dict(tickmode="array", tickvals=x_all, ticktext=_ticktext),
                        height=560, margin=dict(l=10, r=10, t=40, b=30),
                    )
                    st.plotly_chart(fig_e, use_container_width=True, key=f"fav_dbg_e_{sel_idx}")


_INSTANCES_PER_CLASS = 10   # nb_instances_test de main_expe_overall


def _instance_means(scores: pd.Series | None) -> list[float]:
    """Moyenne par instance de test. raw_scores est rangé instance par instance : 10 blocs
    consécutifs de nb_restarts runs. nb_restarts vaut 10 en général, moins après un repli OOM
    (ex. 80 scores = 8 restarts) : la taille des blocs se déduit donc du nombre de scores."""
    if scores is None:
        return []
    v = scores.to_numpy(dtype=float)
    if len(v) == 0 or len(v) % _INSTANCES_PER_CLASS:
        return []   # fichier incomplet ou corrompu : pas d'appariement fiable
    return v.reshape(_INSTANCES_PER_CLASS, -1).mean(axis=1).tolist()


def _global_wilcoxon(gaps) -> dict | None:
    """Test global : Wilcoxon signé sur les écarts relatifs (%) par classe, une paire par classe
    (les 10 instances d'une classe ne sont pas indépendantes : pas de test sur 560 paires).
    -> {n, median, p} ou None. Même méthode dans les onglets Ablation et Comparaison."""
    from scipy.stats import wilcoxon
    gaps = np.asarray(list(gaps), dtype=float)
    if len(gaps) < 2:
        return None
    # Bilatéral : on teste « différent de la référence », le sens est donné par l'écart médian
    p = 1.0 if np.allclose(gaps, 0) else float(wilcoxon(gaps, alternative="two-sided").pvalue)
    return {"n": len(gaps), "median": float(np.median(gaps)), "p": p}


def _ablation_global_tests(rows: list[dict], configs: list[str], ref: str) -> dict:
    """Test global par variante : Wilcoxon signé sur les écarts relatifs (%) à la référence, un par
    classe (les 10 instances d'une classe ne sont pas indépendantes : pas de test sur 560 paires),
    puis correction de Holm sur l'ensemble des variantes. -> {cfg: {n, median, p, p_holm}}."""
    out = {}
    for cfg in configs:
        if cfg == ref:
            continue
        g = _global_wilcoxon(r["gap"][cfg] for r in rows if cfg in r.get("gap", {}))
        if g:
            out[cfg] = g
    # Holm : p triées croissantes, p_(i) * (m - i), puis maximum cumulé, borné à 1
    order = sorted(out, key=lambda c: out[c]["p"])
    m, running = len(order), 0.0
    for i, cfg in enumerate(order):
        running = max(running, min(1.0, (m - i) * out[cfg]["p"]))
        out[cfg]["p_holm"] = running
    return out


def _paired_wilcoxon(a: list[float], b: list[float]) -> float | None:
    """p-valeur du Wilcoxon apparié sur les moyennes par instance (comme main_table.py)."""
    n = min(len(a), len(b))
    if n < 2:
        return None
    d = np.asarray(a[:n]) - np.asarray(b[:n])
    if np.allclose(d, 0):
        return 1.0
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(a[:n], b[:n], alternative="two-sided").pvalue)
    except Exception:
        return None


# Paramètres balayables dans l'onglet Sensi (ordre d'affichage, clés de parse_config_name)
SENSI_PARAMS = list(dict.fromkeys(key for _, key, _ in PARSERS))
SENSI_DEFAULTS = (   # (config, nom) proposés par défaut pour les méthodes 1, 2, 3
    ("krbf__advglobalrankweighted__M7__L13__eps0p08__g0p015__ds0p03__dm0p01", "PPSN"),
    ("krbf__advperagentrankweighted__M7__L13__eps0p06__g0p01__ds1__dm0p01__ppokl__pe6__b0p1", "EvoCOP"),
    ("krbf__advperagentrankweighted__M7__L13__eps0p08__g0p015__ds0p03__dm0p01", "PPSN + rang/agent"),
)
SENSI_MAX_METHODS = 6


def _sensi_variants(ref_cfg: str, param: str, all_configs: list[str]) -> dict:
    """Configs identiques à ref_cfg sur tous les paramètres sauf `param` -> {valeur: config}."""
    ref = parse_config_name(ref_cfg)
    if param not in ref:
        return {}
    ref_rest = {k: v for k, v in ref.items() if k != param}
    out = {}
    for cfg in all_configs:
        p = parse_config_name(cfg)
        if param in p and {k: v for k, v in p.items() if k != param} == ref_rest:
            out[p[param]] = cfg
    return dict(sorted(out.items()))


# Slots 1-2 de la palette catégorielle de référence (validée CVD)
SENSI_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
# Forme de marqueur par méthode : l'identité ne repose pas que sur la couleur (impression N&B, daltonisme)
SENSI_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "star"]
# Export PDF : même ratio que dans le dashboard (≈ 1060×640 px en demi-largeur, ratio 1,66) et
# mêmes tailles en px. 880×530 px -> page 660×398 pt ; avec \includegraphics[width=12cm] l'échelle
# est 0,52 : texte export 26 px -> ≈ 10 pt. Marges nulles (automargin garde graduations et titres)
SENSI_PDF_W, SENSI_PDF_H = 880, 530
# Place des graduations + titres d'axe (automargin mesure avant le chargement de Latin Modern et
# sous-estime : titre y sur les graduations). Pas d'espace blanc au-delà.
SENSI_PDF_MB = 78


def _pdf_left_margin(tick_labels: list[str]) -> int:
    """Marge gauche (px) du PDF : titre de l'axe y (≈ 40 px) + graduations les plus larges
    (≈ 0,55 em par caractère à 25 px) + 12 px d'écart. ≈ 107 px pour « 0.75 », 79 pour « 50 »."""
    chars = max((len(t) for t in tick_labels), default=2)
    return int(40 + chars * 0.55 * 25 + 12)
SENSI_PDF_FONT = "Latin Modern Roman"   # police LaTeX/LNCS ; un seul nom : une liste fait retomber Chromium sur la serif par défaut


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _default_label_y(n_methods: int) -> float:
    """Hauteur (repère papier) de l'étiquette « default » : tout en haut avec 2 méthodes ; sous la
    légende au-delà, quand la légende horizontale s'étend jusqu'au-dessus de la bande grise."""
    return 0.995 if n_methods <= 2 else 0.88


def _sensi_caption(inst: str) -> str:
    """Texte de légende LaTeX, ex. « NK3, n=512, K=8 »."""
    m = INSTANCE_RE.match(inst)
    pb, dim, t = m.group("problem"), m.group("dim"), m.group("t")
    return f"{pb}, n={dim}, " + (f"t={t}" if pb == "QUBO" else f"K={t}")


def _sensi_figure(stats, param: str, cats: list[str], ref_vals: set[str],
                  style: str, export: bool, dark: bool = False, legend: bool = True) -> go.Figure:
    """stats : [(label, color, {valeur: scores})]. Sans titre (l'instance va dans la légende LaTeX)."""
    fig = go.Figure()
    for j, (label, color, per_val) in enumerate(stats):
        symbol = SENSI_SYMBOLS[j % len(SENSI_SYMBOLS)]
        xs = [v for v in cats if v in per_val]
        if style == "Boîtes":
            fig.add_trace(go.Box(
                x=[v for v in xs for _ in per_val[v]],
                y=[y for v in xs for y in per_val[v].tolist()],
                name=label, legendgroup=label, offsetgroup=label,
                fillcolor=_hex_rgba(color, 0.22), line=dict(color=color, width=1.2),
                boxpoints=False, whiskerwidth=0.5,
            ))
            continue
        q = {v: np.percentile(per_val[v], [25, 50, 75]) for v in xs}
        # Bande interquartile (q75 puis q25 en remplissage)
        fig.add_trace(go.Scatter(x=xs, y=[q[v][2] for v in xs], mode="lines", line=dict(width=0),
                                 legendgroup=label, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=[q[v][0] for v in xs], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=_hex_rgba(color, 0.2),
                                 legendgroup=label, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=xs, y=[q[v][1] for v in xs], name=label, legendgroup=label, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(color=color, size=9, symbol=symbol),
            customdata=[[q[v][0], q[v][2]] for v in xs],
            hovertemplate=(f"<b>{label}</b><br>{param}=%{{x}}<br>médiane=%{{y:.4g}}"
                           "<br>IQR=[%{customdata[0]:.4g}, %{customdata[1]:.4g}]<extra></extra>"),
        ))
    for rv in ref_vals & set(cats):
        k = cats.index(rv)
        fig.add_vrect(x0=k - 0.5, x1=k + 0.5, fillcolor="rgba(128,128,128,0.13)", line_width=0,
                      layer="below")
        # En haut de la bande, dans la marge réservée en haut de l'axe y (au-dessus des courbes)
        fig.add_annotation(x=rv, y=_default_label_y(len(stats) if legend or not export else 0), yref="paper", yanchor="top", text="default", showarrow=False,
                           font=dict(size=22 if export else 19, color="rgba(110,110,110,1)"))

    fs = 26 if export else 22   # export : ≈ 10 pt une fois inclus à 12 cm
    ink = "#262730" if export or not dark else "#e6e6e6"   # texte : thème clair / sombre
    y_max = max(float(np.max(sc)) for _, _, pv in stats for sc in pv.values())
    # Étendue tracée (bande IQR ou boîtes), + marge en haut pour loger la légende sans recouvrement
    lo_q, hi_q = (0, 100) if style == "Boîtes" else (25, 75)
    y_lo = min(float(np.percentile(sc, lo_q)) for _, _, pv in stats for sc in pv.values())
    y_hi = max(float(np.percentile(sc, hi_q)) for _, _, pv in stats for sc in pv.values())
    pad = (y_hi - y_lo) or 1.0
    x_title = "<i>m</i>" if param == "M" else param
    step = 2 if len(cats) > 15 else 1
    fig.update_layout(
        boxmode="group", boxgap=0.25, boxgroupgap=0.08,
        height=640, margin=dict(t=44, l=80, r=10, b=70),  # t : place pour la barre d'outils Plotly
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=fs, color=ink,
                  family=SENSI_PDF_FONT if export else None),
        xaxis=dict(title=dict(text=x_title, standoff=8), automargin=True, type="category",
                   categoryorder="array", categoryarray=cats,
                   tickvals=cats[::step] if step > 1 else None,
                   showgrid=False, showline=True, linecolor="rgba(128,128,128,0.6)", ticks="",
                   tickfont=dict(size=fs - 1)),
        yaxis=dict(title=dict(text="Final score", standoff=8), automargin=True, showgrid=True,
                   gridcolor="rgba(128,128,128,0.2)", zeroline=False, showline=False,
                   tickformat=".2f" if y_max < 10 else "~s", tickfont=dict(size=fs - 1),
                   # Marge en haut pour la légende ; réduite quand le PDF est exporté sans légende
                   range=[y_lo - 0.04 * pad,
                          y_hi + ((0.2 if len(stats) <= 2 else 0.32) if legend or not export else 0.1) * pad]),
        # Légende dans la zone de tracé, en haut à droite (les courbes y sont basses : grands m)
        legend=dict(orientation="h", y=0.99, yanchor="top", x=0.99, xanchor="right",
                    bgcolor="rgba(255,255,255,0.75)" if not dark or export else "rgba(14,17,23,0.75)",
                    bordercolor="rgba(128,128,128,0.35)", borderwidth=1),
        hoverlabel=dict(font_size=17), hovermode="x unified" if style != "Boîtes" else "closest",
    )
    if export:
        widest = f"{y_hi:.2f}" if y_max < 10 else (f"{y_hi / 1000:.1f}k" if y_hi >= 1000 else f"{y_hi:.0f}")
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", showlegend=legend,
                          margin=dict(t=0, l=_pdf_left_margin([widest]), r=4, b=SENSI_PDF_MB))
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.12)")
        fig.update_xaxes(linecolor="rgba(0,0,0,0.5)")
    return fig


# Légende seule, en bandeau, à placer une fois au-dessus des figures dans LaTeX. Pensée pour
# \includegraphics[width=\textwidth] : 1000 px -> 750 pt, échelle ≈ 0,46 sur 12,2 cm -> texte ≈ 9 pt
SENSI_LEGEND_H, SENSI_LEGEND_FS = 96, 26


def _sensi_legend_width(labels: list[str]) -> int:
    """Largeur (px) du bandeau : texte (~0,55 em par caractère) + marqueur et espacement par entrée,
    avec une marge, pour qu'aucun nom ne soit coupé quel que soit le nombre de méthodes."""
    chars = sum(len(l) for l in labels)
    return int(chars * 0.55 * SENSI_LEGEND_FS + len(labels) * 110 + 80)


def _crop_pdf(pdf: bytes, margin_pt: float = 2.0) -> bytes:
    """Recadre un PDF au plus près de son contenu. Chromium (kaleido) peint toujours un fond blanc
    sur toute la page, donc pdfcrop ne voit pas de marge : on cherche le contenu sur un rendu image
    (pdftoppm), puis Ghostscript fixe la CropBox. Le PDF reste vectoriel. Inchangé en cas d'échec."""
    import subprocess, tempfile
    import numpy as np
    from PIL import Image
    dpi = 300
    with tempfile.TemporaryDirectory() as d:
        src, png, dst = Path(d) / "in.pdf", Path(d) / "page", Path(d) / "out.pdf"
        src.write_bytes(pdf)
        try:
            subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(src), str(png)],
                           check=True, capture_output=True, timeout=60)
            img = np.asarray(Image.open(f"{png}.png").convert("L"))
            ys, xs = np.nonzero(img < 250)
            if not len(xs):
                return pdf
            h = img.shape[0]
            k = 72.0 / dpi                                  # px -> pt ; origine PDF en bas à gauche
            box = [xs.min() * k - margin_pt, (h - 1 - ys.max()) * k - margin_pt,
                   (xs.max() + 1) * k + margin_pt, (h - ys.min()) * k + margin_pt]
            subprocess.run(["gs", "-q", "-o", str(dst), "-sDEVICE=pdfwrite", "-dUseCropBox",
                            "-c", f"[/CropBox [{' '.join(f'{v:.2f}' for v in box)}] /PAGES pdfmark",
                            "-f", str(src)], check=True, capture_output=True, timeout=60)
            return dst.read_bytes()
        except Exception:
            return pdf


def _sensi_pdf(fig: go.Figure) -> bytes:
    """PDF d'une figure Sensi exportée, recadré au plus près du contenu (marge 2 pt)."""
    return _crop_pdf(fig.to_image(format="pdf", width=fig.layout.width or SENSI_PDF_W,
                                  height=fig.layout.height or SENSI_PDF_H))


def _sensi_legend_pdf(labels: list[str]) -> bytes:
    fig = _sensi_legend_figure(labels)
    return _crop_pdf(fig.to_image(format="pdf", width=_sensi_legend_width(labels), height=SENSI_LEGEND_H))


def _sensi_legend_figure(labels: list[str]) -> go.Figure:
    fig = go.Figure()
    for j, label in enumerate(labels):
        color = SENSI_COLORS[j % len(SENSI_COLORS)]
        fig.add_trace(go.Scatter(x=[None], y=[None], name=label, mode="lines+markers",
                                 line=dict(color=color, width=2),
                                 marker=dict(color=color, size=11, symbol=SENSI_SYMBOLS[j % len(SENSI_SYMBOLS)])))
    fig.update_layout(
        width=_sensi_legend_width(labels), height=SENSI_LEGEND_H, margin=dict(t=0, b=0, l=0, r=0),
        # Fond transparent : sinon pdfcrop voit le fond blanc comme du contenu et ne recadre rien
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        font=dict(family=SENSI_PDF_FONT, size=SENSI_LEGEND_FS, color="#262730"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=0.5, yanchor="middle",
                    itemwidth=60, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def _sensi_rank_figure(df: pd.DataFrame, labels: list[str], param: str, ref_vals: set[str],
                       export: bool, dark: bool = False, legend: bool = True) -> go.Figure:
    """Rang moyen (vs concurrents) en fonction du paramètre, axe y inversé (plus bas = meilleur)."""
    cats = [str(v) for v in df[param]]
    fs = 26 if export else 22   # export : ≈ 10 pt une fois inclus à 12 cm
    ink = "#262730" if export or not dark else "#e6e6e6"
    fig = go.Figure()
    for j, label in enumerate(labels):
        col = f"{label} · rang moy"
        if col not in df.columns:
            continue
        color = SENSI_COLORS[j % len(SENSI_COLORS)]
        fig.add_trace(go.Scatter(
            x=cats, y=df[col], name=label, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(color=color, size=9, symbol=SENSI_SYMBOLS[j % len(SENSI_SYMBOLS)]),
            hovertemplate=f"<b>{label}</b><br>{param}=%{{x}}<br>rang moyen=%{{y:.1f}}<extra></extra>",
        ))
    for rv in ref_vals & set(cats):
        k = cats.index(rv)
        fig.add_vrect(x0=k - 0.5, x1=k + 0.5, fillcolor="rgba(128,128,128,0.13)", line_width=0, layer="below")
        fig.add_annotation(x=rv, y=_default_label_y(len(labels) if legend or not export else 0), yref="paper", yanchor="top", text="default", showarrow=False,
                           font=dict(size=22 if export else 19, color="rgba(110,110,110,1)"))
    y_hi = float(np.nanmax(df[[f"{l} · rang moy" for l in labels if f"{l} · rang moy" in df]].to_numpy()))
    step = 2 if len(cats) > 15 else 1
    fig.update_layout(
        height=640, margin=dict(t=44, l=80, r=10, b=70),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=fs, color=ink,
                  family=SENSI_PDF_FONT if export else None),
        xaxis=dict(title=dict(text="<i>m</i>" if param == "M" else param, standoff=8), automargin=True,
                   type="category", categoryorder="array", categoryarray=cats,
                   tickvals=cats[::step] if step > 1 else None, showgrid=False, showline=True,
                   linecolor="rgba(128,128,128,0.6)", ticks="", tickfont=dict(size=fs - 1)),
        # Inversé : rang 1 en haut. Bande vide au-dessus du rang 1 (≈ 25 % de la hauteur) pour la
        # légende et « default », que les courbes ne peuvent pas atteindre ; graduations ≥ 1 seulement
        yaxis=dict(title=dict(text="Mean rank", standoff=8), automargin=True,
                   range=[y_hi * 1.04, 1 - (0.33 if legend or not export else 0.12) * (y_hi - 1)],
                   tickvals=[1] + list(range(10, int(y_hi) + 1, 10)),
                   showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=False,
                   tickfont=dict(size=fs - 1)),
        legend=dict(orientation="h", y=0.99, yanchor="top", x=0.99, xanchor="right",
                    bgcolor="rgba(255,255,255,0.75)" if not dark or export else "rgba(14,17,23,0.75)",
                    bordercolor="rgba(128,128,128,0.35)", borderwidth=1),
        hoverlabel=dict(font_size=17), hovermode="x unified",
    )
    if export:
        ticks = [str(v) for v in [1] + list(range(10, int(y_hi) + 1, 10))]
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", showlegend=legend,
                          margin=dict(t=0, l=_pdf_left_margin(ticks), r=4, b=SENSI_PDF_MB))
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.12)")
        fig.update_xaxes(linecolor="rgba(0,0,0,0.5)")
    return fig


def _sensi_summary_table(series, param: str, ref_vals: set[str], pdf_legend: bool = True):
    """Renvoie la figure d'export du rang moyen (pour « Tout exporter »), ou None."""
    """Récap par valeur du paramètre : rang moyen / top-k vs concurrents, et duel entre les deux méthodes."""
    try:
        inst = pd.read_parquet(INSTANCES_FILE)
    except Exception:
        st.info("Récap indisponible (instances.parquet illisible).")
        return None
    scope = st.radio("Périmètre du récap", ["Tout", "QUBO", "NK", "NK3"], horizontal=True,
                     key="sensi_scope")
    key = ["problem", "dim", "t"]
    # Rang du cache : chaque variante seule contre le pool fixe de concurrents externes
    inst = inst[inst["config"].isin({c for _, variants in series for c in variants.values()})]
    if scope != "Tout":
        inst = inst[inst["problem"] == scope]
    vals = sorted({v for _, variants in series for v in variants})
    rows = []
    for v in vals:
        row = {param: v}
        per = []
        for label, variants in series:
            sub = inst[inst["config"] == variants[v]] if v in variants else inst.iloc[0:0]
            per.append(sub.set_index(key)["score"])
            row[f"{label} · rang moy"] = round(float(sub["rank"].mean()), 1) if len(sub) else None
            row[f"{label} · top 1"] = int((sub["rank"] == 1).sum()) if len(sub) else None
            row[f"{label} · top 3"] = int((sub["rank"] <= 3).sum()) if len(sub) else None
        # Duel et écart par famille de chaque méthode vs la première (référence ; > 0 : meilleure)
        for j in range(1, len(series)):
            if not (len(per[0]) and len(per[j])):
                continue
            a, b = per[0].align(per[j], join="inner")
            sfx = "" if len(series) == 2 else f" ({series[j][0]})"
            row[f"{series[j][0]} > {series[0][0]}"] = f"{int((b > a).sum())}/{len(a)}"
            gap = (b / a - 1) * 100
            fams = gap.index.get_level_values("problem")
            for fam in ("QUBO", "NK", "NK3"):
                if (fams == fam).any():
                    row[f"Δ {fam} %{sfx}"] = float(gap[fams == fam].mean())
        rows.append(row)
    df = pd.DataFrame(rows)

    rank_cols = [c for c in df.columns if c.endswith("rang moy")]
    top_cols = [c for c in df.columns if "· top" in c]
    gap_cols = [c for c in df.columns if c.startswith("Δ ")]

    def _style(d: pd.DataFrame):
        out = pd.DataFrame("", index=d.index, columns=d.columns)
        ref = d[param].astype(str).isin(ref_vals)
        out.loc[ref, :] = "background-color: rgba(128,128,128,0.15)"
        for c in rank_cols:
            out.loc[d[c] == d[c].min(), c] += "; font-weight: 700"
        for c in top_cols + gap_cols:
            out.loc[d[c] == d[c].max(), c] += "; font-weight: 700"
        return out

    n_inst = inst[key].drop_duplicates().shape[0]
    scope_txt = scope if scope != "Tout" else "toutes les instances"
    left, right = st.columns(2)
    with left:
        st.markdown(f"##### Rang moyen — {scope_txt} ({n_inst} instances)")
        dark = getattr(st.context.theme, "type", None) == "dark"
        labels = [label for label, _ in series]
        fig = _sensi_rank_figure(df, labels, param, ref_vals, export=False, dark=dark)
        st.plotly_chart(fig, use_container_width=True, theme=None, key="sensi_rank_fig")
        st.download_button(
            "⬇️ PDF", key="sensi_rank_dl", mime="application/pdf",
            file_name=f"sensi_{param}_mean_rank{'' if scope == 'Tout' else '_' + scope}.pdf",
            data=lambda: _sensi_pdf(_sensi_rank_figure(df, labels, param, ref_vals, export=True,
                                                       legend=pdf_legend)),
        )
    right.markdown(f"##### Récap — {scope_txt} ({n_inst} instances)")
    right.dataframe(
        df.style.apply(_style, axis=None).format(
            {c: "{:.1f}" for c in rank_cols} | {c: "{:.0f}" for c in top_cols}
            | {c: "{:+.2f}" for c in gap_cols}, na_rep="—"),
        hide_index=True, use_container_width=True,
        height=640,
    )
    gap_txt = (f" Δ famille % : écart de score moyen de chaque méthode par rapport à {series[0][0]} "
               f"(> 0 : meilleure que {series[0][0]}).")
    right.caption("Rang contre le pool fixe de concurrents externes (82, 80 sur NK3 ; aucune méthode SVGD ; "
                  "1 = meilleur). En gras : meilleure valeur de chaque colonne ; "
               "ligne grisée : valeur par défaut." + gap_txt)
    rank_name = f"sensi_{param}_mean_rank{'' if scope == 'Tout' else '_' + scope}.pdf"
    return rank_name, lambda: _sensi_rank_figure(df, labels, param, ref_vals, export=True, legend=pdf_legend)


@st.fragment
def tab_sensi(all_df: pd.DataFrame, sorted_instances: list) -> None:
    """Sensibilité à un paramètre : N configs de référence (méthodes) et toutes leurs variantes.
    La première méthode sert de référence pour les duels et les écarts par famille."""
    all_configs = all_df["config"].tolist()
    if not all_configs:
        st.info("Aucune config disponible.")
        return

    # Choix sauvegardés dans dashboard/sensi.json : ils survivent au rechargement de la page
    saved = load_sensi()
    saved_methods = saved.get("methods", [])
    h1, h2 = st.columns([2, 2])
    n_methods = int(h1.number_input("Nombre de méthodes", 2, SENSI_MAX_METHODS,
                                    min(max(int(saved.get("n", 2)), 2), SENSI_MAX_METHODS), key="sensi_n"))
    saved_param = saved.get("param", "M")
    param = h2.selectbox("Paramètre", SENSI_PARAMS,
                         index=SENSI_PARAMS.index(saved_param if saved_param in SENSI_PARAMS else "M"),
                         key="sensi_param")
    sel = []
    cols = st.columns(min(n_methods, 3))
    for i in range(n_methods):
        letter = "abcdefgh"[i]
        if i < len(saved_methods):
            default, name = saved_methods[i]["config"], saved_methods[i]["name"]
        elif i < len(SENSI_DEFAULTS):
            default, name = SENSI_DEFAULTS[i]
        else:
            default, name = all_configs[0], f"Méthode {i + 1}"
        with cols[i % len(cols)]:
            idx = all_configs.index(default) if default in all_configs else 0
            cfg = st.selectbox(f"Config {letter.upper()}" + (" (réf.)" if i == 0 else ""), all_configs,
                               index=idx, key=f"sensi_{letter}")
            label = st.text_input("Nom", name, key=f"sensi_{letter}_label")
            sel.append((cfg, label))
    # Les méthodes au-delà de n_methods restent mémorisées (repasser de 2 à 3 les retrouve)
    methods = [{"config": c, "name": l} for c, l in sel] + saved_methods[n_methods:]
    state = {"n": n_methods, "param": param, "methods": methods}
    if state != saved:
        save_sensi(state)

    series = []
    for cfg, label in sel:
        variants = _sensi_variants(cfg, param, all_configs)
        series.append((label, variants))
        vals = ", ".join(str(v) for v in variants) or "—"
        st.caption(f"**{label}** : {len(variants)} variantes ({param} = {vals})")
    if not any(v for _, v in series):
        st.warning(f"Aucune variante trouvée en faisant varier `{param}`.")
        return

    # Pour LaTeX : figures sans légende + une légende seule, placée une fois au-dessus
    l1, l2 = st.columns([1, 1])
    pdf_legend = l1.checkbox("Légende dans les PDF", value=True, key="sensi_pdf_legend",
                             help="Décocher pour exporter les figures sans légende, et placer la légende "
                                  "seule une fois au-dessus des figures dans LaTeX.")
    legend_labels = [label for label, _ in series]
    l2.download_button("⬇️ Légende seule (PDF)", key="sensi_legend_dl", mime="application/pdf",
                       file_name=f"sensi_{param}_legend.pdf", use_container_width=True,
                       data=lambda: _sensi_legend_pdf(legend_labels))
    rank_export = _sensi_summary_table(series, param, {str(parse_config_name(cfg).get(param)) for cfg, _ in sel},
                                       pdf_legend)

    groups = sorted({(m.group("problem"), int(m.group("dim")))
                     for m in map(INSTANCE_RE.match, sorted_instances) if m},
                    key=lambda g: (["QUBO", "NK", "NK3"].index(g[0]), g[1]))
    g1, g2 = st.columns(2)
    problem = g1.selectbox("Problème", sorted({g[0] for g in groups}, key=["QUBO", "NK", "NK3"].index),
                           key="sensi_pb")
    dim = g2.selectbox("Dimension", [d for p, d in groups if p == problem], key="sensi_dim")
    instances = [i for i in sorted_instances
                 if (m := INSTANCE_RE.match(i)) and m.group("problem") == problem
                 and int(m.group("dim")) == dim]
    instances.sort(key=lambda i: int(INSTANCE_RE.match(i).group("t")))

    all_vals = sorted({v for _, variants in series for v in variants})
    cats = [str(v) for v in all_vals]
    ref_vals = {str(parse_config_name(cfg).get(param)) for cfg, _ in sel}

    s1, s2 = st.columns([3, 2])
    style = s1.radio("Style", ["Médiane + IQR", "Boîtes"], horizontal=True, key="sensi_style")
    export = s2.button("📄 Tout exporter en PDF (12 cm)", key="sensi_export", use_container_width=True,
                       help="Écrit un PDF par figure affichée dans figures/sensi/<paramètre>/")

    figs = []
    for inst in instances:
        stats = []  # (label, color, {val: scores})
        for i, (label, variants) in enumerate(series):
            per_val = {str(v): sc for v, cfg in variants.items()
                       if (sc := load_raw_scores(cfg, inst)) is not None}
            if per_val:
                stats.append((label, SENSI_COLORS[i % len(SENSI_COLORS)], per_val))
        if stats:
            figs.append((inst, stats))

    dark = getattr(st.context.theme, "type", None) == "dark"
    cols = st.columns(2)
    for n, (inst, stats) in enumerate(figs):
        fig = _sensi_figure(stats, param, cats, ref_vals, style, export=False, dark=dark)
        with cols[n % 2]:
            st.caption(_sensi_caption(inst))
            # theme=None : sinon le thème Streamlit écrase tailles de police et couleurs
            st.plotly_chart(fig, use_container_width=True, theme=None, key=f"sensi_fig_{inst}")
            st.download_button(
                "⬇️ PDF", key=f"sensi_dl_{inst}", mime="application/pdf",
                file_name=f"sensi_{param}_{inst}{'_box' if style == 'Boîtes' else ''}.pdf",
                # Généré au clic seulement (kaleido ~1 s par figure)
                data=lambda st_=stats: _sensi_pdf(_sensi_figure(st_, param, cats, ref_vals, style,
                                                                export=True, legend=pdf_legend)),
            )

    if export and figs:
        out_dir = ROOT / "figures" / "sensi" / param
        out_dir.mkdir(parents=True, exist_ok=True)
        with st.spinner(f"Export de {len(figs) + 1} PDF…"):
            if rank_export:
                rank_name, make_rank = rank_export
                (out_dir / rank_name).write_bytes(_sensi_pdf(make_rank()))
            for inst, stats in figs:
                fig = _sensi_figure(stats, param, cats, ref_vals, style, export=True, legend=pdf_legend)
                (out_dir / f"sensi_{param}_{inst}{'_box' if style == 'Boîtes' else ''}.pdf").write_bytes(
                    _sensi_pdf(fig))
        st.success(f"{len(figs) + bool(rank_export)} PDF écrits dans `{out_dir.relative_to(ROOT)}/`")


# ── Anytime : meilleure fitness en fonction du nombre d'évaluations ────────────
ANYTIME_BUDGET = 50000
ANYTIME_TOP_COMPETITORS = 2      # concurrents externes affichés : les meilleurs de chaque classe
ANYTIME_DEFAULT_CLASSES = ["NK_dim64_t1", "NK3_dim512_t8", "QUBO_dim512_t3"]
# Concurrents : couleurs prises hors de celles des méthodes (jamais de collision), traits pointillés
ANYTIME_COMP_DASH = ["dash", "dot", "dashdot", "longdash"]
ANYTIME_COMP_EXTRA_COLORS = ["#7f7f7f", "#8c564b", "#17becf", "#bcbd22", "#000000", "#9467bd"]


@st.cache_data(show_spinner=False)
def _anytime_method_curve(config: str, instance: str, mtime: float | None):
    """(évaluations, moyenne, q25, q75) de la meilleure fitness sur les 100 runs (best_metrics.csv).
    Courbe = MOYENNE, comme le score final, les rangs et les courbes des concurrents (la médiane sur
    10 instances d'optimums différents n'est pas comparable). QUBO stocké en négatif : valeur absolue."""
    if mtime is None:
        return None
    try:
        d = pd.read_csv(RESULTS_DIR / config / instance / "best_metrics.csv")
        if d.empty or not {"step", "mean", "25%", "75%"} <= set(d.columns):
            return None
        q1, mean, q3 = d["25%"].abs().to_numpy(), d["mean"].abs().to_numpy(), d["75%"].abs().to_numpy()
        x, m, lo, hi = d["step"].tolist(), mean.tolist(), np.minimum(q1, q3).tolist(), np.maximum(q1, q3).tolist()
        # Dernière génération complète < budget (ex. 549 × 91 = 49 959) : les évaluations restantes
        # servent aux solutions finales des agents ; la meilleure fitness ne peut que rester égale
        if x and x[-1] < ANYTIME_BUDGET:
            x, m, lo, hi = x + [ANYTIME_BUDGET], m + [m[-1]], lo + [lo[-1]], hi + [hi[-1]]
        return x, m, lo, hi
    except Exception:
        return None


def _anytime_curve(config: str, instance: str):
    p = RESULTS_DIR / config / instance / "best_metrics.csv"
    return _anytime_method_curve(config, instance, p.stat().st_mtime if p.exists() else None)


def _short_algo(name: str) -> str:
    """Nom court d'un concurrent Nevergrad pour les légendes : sans « Discrete », « OnePlusOne » -> « (1+1) »,
    « Recombining » -> « Recomb ». Ex. HugeLognormalDiscreteOnePlusOne -> HugeLognormal (1+1)."""
    n = name.strip()
    if "OnePlusOne" not in n:          # DiscreteDE, PEDA, PPBIL… : nom inchangé (« DE » serait ambigu)
        return n
    n = n.replace("Discrete", "").replace("Recombining", "Recomb")
    if n.endswith("OnePlusOne"):
        n = n[: -len("OnePlusOne")] + " (1+1)"
    elif "OnePlusOne" in n:
        n = n.replace("OnePlusOne", " (1+1) ")
    return n.strip() or name


@st.cache_data(show_spinner=False)
def _anytime_comp_style(all_instances: tuple[str, ...], n_methods: int) -> dict:
    """Couleur et trait FIXES de chaque concurrent sur tout le benchmark : on parcourt les meilleurs
    concurrents de toutes les classes ; un même concurrent a la même apparence dans chaque figure."""
    from collections import Counter
    cnt = Counter()
    for inst in all_instances:
        m = INSTANCE_RE.match(inst)
        rk = load_ranking(m.group("problem"), int(m.group("dim")), int(m.group("t")))
        if rk is not None and not rk.empty:
            cnt.update(rk.nlargest(ANYTIME_TOP_COMPETITORS, "score")["name_algo"].tolist())
    method_colors = {SENSI_COLORS[j % len(SENSI_COLORS)] for j in range(n_methods)}
    colors = [c for c in SENSI_COLORS + ANYTIME_COMP_EXTRA_COLORS if c not in method_colors]
    names = [a for a, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]   # les plus fréquents d'abord
    return {a: (colors[k % len(colors)], ANYTIME_COMP_DASH[(k // len(colors)) % len(ANYTIME_COMP_DASH)])
            for k, a in enumerate(names)}


ANYTIME_ZOOM_FROM = 40000   # encart : zoom sur la fin du run, de 40 000 à 50 000 évaluations


def _add_end_zoom(fig: go.Figure, methods, comps, x0: float, fs: int, dark: bool, logx: bool = False) -> None:
    """Encart en bas à droite avec les mêmes courbes sur [x0, budget] et un axe y resserré ;
    rectangle pointillé sur la figure principale."""
    series = ([(c[0], c[1], SENSI_COLORS[j % len(SENSI_COLORS)], "solid", 2.5) for j, (_, c) in enumerate(methods)]
              + [(c[0], c[1], color, dash, 2) for _, c, color, dash in comps])
    window = []
    for x, y, color, dash, width in series:
        x, y = np.asarray(x, float), np.asarray(y, float)
        keep = x >= x0
        if keep.any():
            i0 = max(int(np.argmax(keep)) - 1, 0)           # un point avant x0 : courbe continue au bord
            window.append((x[i0:], y[i0:], color, dash, width))
    if not window:
        return
    ys = np.concatenate([w[1][w[0] >= x0] for w in window])
    lo, hi = float(ys.min()), float(ys.max())
    pad = (hi - lo) * 0.12 or abs(hi) * 0.002 or 1e-3
    lo, hi = lo - pad, hi + pad
    for x, y, color, dash, width in window:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", xaxis="x2", yaxis="y2", showlegend=False,
                                 line=dict(color=color, width=width, dash=dash), hoverinfo="skip"))
    dom_x, dom_y = [0.56, 0.98], [0.07, 0.47]
    y_fmt = ".3f" if hi < 10 else "~s"
    fig.update_layout(
        xaxis2=dict(domain=dom_x, anchor="y2", range=[x0, ANYTIME_BUDGET * 1.01], showgrid=False,
                    tickvals=[x0, (x0 + ANYTIME_BUDGET) / 2, ANYTIME_BUDGET],
                    ticktext=[f"{x0 / 1000:g}k", f"{(x0 + ANYTIME_BUDGET) / 2000:g}k", f"{ANYTIME_BUDGET / 1000:g}k"],
                    tickfont=dict(size=fs - 7), showline=True, mirror=True, linecolor="rgba(128,128,128,0.7)",
                    ticks=""),
        yaxis2=dict(domain=dom_y, anchor="x2", range=[lo, hi], tickformat=y_fmt, nticks=4,
                    tickfont=dict(size=fs - 7), showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                    showline=True, mirror=True, linecolor="rgba(128,128,128,0.7)", zeroline=False),
    )
    bg = "white" if not dark else "#0e1117"
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=dom_x[0] - 0.07, x1=dom_x[1] + 0.005,
                  y0=dom_y[0] - 0.06, y1=dom_y[1] + 0.01, fillcolor=bg, line_width=0, layer="below")
    fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=ANYTIME_BUDGET, y0=lo, y1=hi,
                  line=dict(color="rgba(110,110,110,0.9)", width=1.5, dash="dot"), fillcolor="rgba(0,0,0,0)")


def _panel_title(inst: str) -> str:
    """Titre de panneau : « QUBO, n=256, type 5 » ou « NK3, n=128, K=8 »."""
    m = INSTANCE_RE.match(inst)
    pb, dim, t = m.group("problem"), m.group("dim"), m.group("t")
    return f"{pb}, n={dim}, " + (f"type {t}" if pb == "QUBO" else f"K={t}")


def _anytime_figure(methods, comps, logx: bool, export: bool, dark: bool = False,
                    legend: bool = True, band: bool = False, zoom: bool = False,
                    title: str | None = None) -> go.Figure:
    """methods : [(nom, (x, moyenne, q1, q3))] ; comps : [(nom, (x, moyenne), couleur, trait)]. Toutes
    les courbes sont des moyennes sur les runs. band : intervalle interquartile de nos méthodes."""
    fs = 26 if export else 22
    ink = "#262730" if export or not dark else "#e6e6e6"
    fig = go.Figure()
    for j, (label, (x, med, q1, q3)) in enumerate(methods):
        color = SENSI_COLORS[j % len(SENSI_COLORS)]
        if band:
            fig.add_trace(go.Scatter(x=x, y=q3, mode="lines", line=dict(width=0), showlegend=False,
                                     hoverinfo="skip", legendgroup=label))
            fig.add_trace(go.Scatter(x=x, y=q1, mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=_hex_rgba(color, 0.15), showlegend=False, hoverinfo="skip",
                                     legendgroup=label))
        fig.add_trace(go.Scatter(x=x, y=med, name=label, legendgroup=label, mode="lines",
                                 line=dict(color=color, width=2.5),
                                 hovertemplate=f"<b>{label}</b><br>%{{x}} évals<br>moyenne %{{y:.4g}}<extra></extra>"))
    for label, (x, y), color, dash in comps:
        fig.add_trace(go.Scatter(x=x, y=y, name=_short_algo(label), mode="lines",
                                 line=dict(color=color, width=2, dash=dash),
                                 hovertemplate=f"<b>{label}</b><br>%{{x}} évals<br>moyenne %{{y:.4g}}<extra></extra>"))
    ys = [v for _, c in methods for v in c[1]] + [v for _, c, _, _ in comps for v in c[1]]
    y_max = max(ys) if ys else 1.0
    fig.update_layout(
        height=700, margin=dict(t=44, l=80, r=36, b=150),   # r : place pour la graduation « 50k »
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=fs, color=ink, family=SENSI_PDF_FONT if export else None),
        xaxis=dict(title=dict(text="Evaluations", standoff=8), type="log" if logx else "linear",
                   tickvals=[100, 1000, 10000, 50000] if logx else [0, 10000, 20000, 30000, 40000, 50000],
                   ticktext=["100", "1k", "10k", "50k"] if logx else ["0", "10k", "20k", "30k", "40k", "50k"],
                   # Axe exactement jusqu'au budget (plotly : bornes en log10 sur un axe log)
                   # (+1 % au-delà : sinon plotly masque la graduation « 50k » posée sur le bord)
                   range=[2, np.log10(ANYTIME_BUDGET * 1.01)] if logx else [0, ANYTIME_BUDGET * 1.01],
                   automargin=True, showgrid=False, showline=True, linecolor="rgba(128,128,128,0.6)",
                   ticks="", tickfont=dict(size=fs - 1)),
        yaxis=dict(title=dict(text="Best fitness", standoff=8), automargin=True, showgrid=True,
                   gridcolor="rgba(128,128,128,0.2)", zeroline=False, tickfont=dict(size=fs - 1),
                   tickformat=".2f" if y_max < 10 else "~s"),
        # Sous la figure : selon la classe, les courbes occupent tous les coins (montée tôt ou tard)
        legend=dict(orientation="h", y=-0.2, yanchor="top", x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=17), hovermode="x unified",
    )
    if zoom:
        _add_end_zoom(fig, methods, comps, ANYTIME_ZOOM_FROM, fs, dark and not export, logx)
    if export:
        widest = f"{y_max:.2f}" if y_max < 10 else (f"{y_max / 1000:.1f}k" if y_max >= 1000 else f"{y_max:.0f}")
        # Légende (si incluse) sous le titre de l'axe x : la figure grandit vers le bas d'autant,
        # la zone du graphique garde la taille qu'elle a sans légende
        # Légende compacte (police 22 px, symboles serrés) ; lignes estimées d'après la longueur des noms
        lfs = fs - 4
        names = [l for l, _ in methods] + [_short_algo(l) for l, *_ in comps]
        line_w = SENSI_PDF_W - _pdf_left_margin([widest]) - 36
        n_rows, cur = 1, 0.0
        for nm in names:
            w = len(nm) * 0.5 * lfs + 52                       # texte + trait + espacement
            if cur and cur + w > line_w:
                n_rows, cur = n_rows + 1, 0.0
            cur += w
        extra = 34 * n_rows + 16 if legend else 0
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", showlegend=legend,
                          width=SENSI_PDF_W, height=SENSI_PDF_H + extra,
                          margin=dict(t=6, l=_pdf_left_margin([widest]), r=36, b=SENSI_PDF_MB + extra),
                          legend=dict(y=-(SENSI_PDF_MB + 8) / (SENSI_PDF_H - SENSI_PDF_MB - 6), yanchor="top",
                                      font=dict(size=lfs), itemwidth=30, tracegroupgap=0))
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.12)")
        fig.update_xaxes(linecolor="rgba(0,0,0,0.5)")
    if title:
        # Titre centré au-dessus du graphique, même police ; la figure grandit d'autant vers le haut
        t_px = 44
        fig.update_layout(title=dict(text=title, x=0.5, xanchor="center", xref="paper",
                                     y=1.0, yanchor="bottom", yref="paper", pad=dict(b=8),
                                     font=dict(size=fs, color="#262730" if export else None,
                                               family=SENSI_PDF_FONT if export else None)),
                          margin=dict(t=(fig.layout.margin.t or 0) + t_px),
                          height=(fig.layout.height or SENSI_PDF_H) + (t_px if export else 0))
    return fig


ANYTIME_CHECKPOINTS = [1000, 5000, 10000, 20000, 30000, 40000, 50000]


def _anytime_markdown(panels: list, sel: list) -> str:
    """Rapport Markdown autonome (pour un LLM) des courbes anytime : panels = [(classe, méthodes, concurrents)]."""
    def at(c, x):
        xs, ys = c[0], c[1]
        v = [y for xx, y in zip(xs, ys) if xx <= x]
        return v[-1] if v else None

    def reach(c, target):
        return next((int(xx) for xx, y in zip(c[0], c[1]) if y >= target), None)

    fmt = lambda v: "—" if v is None else (f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}")
    out = ["# Courbes anytime (meilleure fitness en fonction des évaluations)", "",
           "## Méthodes SVGD", ""]
    out += [f"- **{l}** : `{c}` — " + ", ".join(f"{k}={v}" for k, v in parse_config_name(c).items()) for c, l in sel]
    out += ["", "## Méthodologie", "",
            "- Courbe = moyenne, sur les 100 runs (10 instances × 10 restarts), de la meilleure fitness trouvée "
            "jusqu'ici ; plus haut = meilleur (valeur absolue pour QUBO). Budget : 50 000 évaluations.",
            "- Concurrents : les 2 meilleurs du pool externe (82 optimiseurs, 80 sur NK3 ; Nevergrad + PBIL/PEDA/PPBIL) "
            "sur la classe, au score final ; courbe = moyenne de leurs runs.",
            "- Rang final : 1 + nombre de concurrents externes strictement meilleurs.",
            "- Hamming (méthodes SVGD seulement, colonne `avg_hamming` de best_metrics.csv) : distance de Hamming "
            "*espérée* entre deux agents, calculée à chaque génération à partir de leurs distributions (pas sur des "
            "solutions tirées). Pour deux agents i, j : somme sur les N variables de la probabilité que leurs "
            "tirages diffèrent, soit p_i + p_j − 2 p_i p_j par variable binaire (QUBO, NK) et 1 − Σ_d p_i,d p_j,d "
            "par variable ternaire (NK3). Moyenne sur toutes les paires d'agents i ≠ j, puis sur les 100 runs. "
            "Va de 0 (agents identiques) à N ; vaut 0 pour m = 1. Mesure la diversité entre agents.", ""]
    for inst, methods, comps in panels:
        m = INSTANCE_RE.match(inst)
        pb, dim, t = m.group("problem"), int(m.group("dim")), int(m.group("t"))
        ref = next((c for l, c in methods if l == sel[0][1]), None)
        out += [f"## {_panel_title(inst)}", "",
                "| méthode | " + " | ".join(f"{x // 1000}k" for x in ANYTIME_CHECKPOINTS) + " | rang final | "
                f"évals pour atteindre le score final de {sel[0][1]} |",
                "|---" * (len(ANYTIME_CHECKPOINTS) + 3) + "|"]
        rows = [(l, c) for l, c in methods] + [(f"{a} ({_short_algo(a)})", c) for a, c, *_ in comps]
        for label, c in rows:
            r = _get_rank(pb, dim, t, c[1][-1])
            e = reach(c, ref[1][-1]) if ref else None
            out.append(f"| {label} | " + " | ".join(fmt(at(c, x)) for x in ANYTIME_CHECKPOINTS)
                       + f" | {r if r else '—'} | {e if e is not None else 'jamais'} |")
        out.append("")
        # Diversité entre agents (Hamming espéré), méthodes SVGD seulement
        ham_rows, ham_bad = [], []
        for cfg, label in sel:
            pth = RESULTS_DIR / cfg / inst / "best_metrics.csv"
            try:
                d = pd.read_csv(pth)
            except Exception:
                continue
            if "avg_hamming" not in d.columns or d.empty:
                continue
            h = [d.loc[d["step"] <= x, "avg_hamming"].iloc[-1] if (d["step"] <= x).any() else None
                 for x in ANYTIME_CHECKPOINTS]
            # Contrôle : à la 1re génération les agents sont quasi uniformes -> ≈ N·(1 − 1/D). Un écart > 15 %
            # signale un fichier calculé avec une ancienne formule (ex. SVGD-EDA sur NK3 n ≤ 256, runs de juin)
            D_ = 3 if pb == "NK3" else 2
            bad = abs(d["avg_hamming"].iloc[0] - dim * (1 - 1 / D_)) > 0.15 * dim * (1 - 1 / D_)
            ham_rows.append(f"| {label}{' ⚠' if bad else ''} | "
                            + " | ".join("—" if v is None else f"{v:.2f}" for v in h) + " |")
            if bad:
                ham_bad.append(label)
        if ham_rows:
            out += [f"Distance de Hamming moyenne entre agents (N = {dim}) :", "",
                    "| méthode | " + " | ".join(f"{x // 1000}k" for x in ANYTIME_CHECKPOINTS) + " |",
                    "|---" * (len(ANYTIME_CHECKPOINTS) + 1) + "|"] + ham_rows + [""]
            if ham_bad:
                out += ["⚠ **Ne pas s'appuyer sur la distance de Hamming de " + ", ".join(ham_bad) + " sur cette "
                        "classe** : ces runs ont été produits avec une ancienne formule du Hamming pour les variables "
                        "ternaires (valeur initiale incompatible avec la définition, jusqu'à dépasser N). Leurs scores "
                        "et rangs ne sont pas concernés.", ""]
    return "\n".join(out)


@st.fragment
def tab_anytime(all_df: pd.DataFrame, sorted_instances: list) -> None:
    """Courbes de convergence : meilleure fitness en fonction des évaluations (médiane + IQR sur les
    100 runs pour nos méthodes, moyenne pour les concurrents), et évaluations pour atteindre une cible."""
    all_configs = all_df["config"].tolist()
    saved = load_anytime()
    # Par défaut, mêmes méthodes et mêmes noms que l'onglet Sensi (légendes identiques dans le papier)
    saved_methods = (saved.get("methods") or load_sensi().get("methods")
                     or [{"config": c, "name": n} for c, n in SENSI_DEFAULTS])

    h1, h2, h3 = st.columns([2, 2, 2])
    n_methods = int(h1.number_input("Nombre de méthodes", 1, SENSI_MAX_METHODS,
                                    min(max(int(saved.get("n", 3)), 1), SENSI_MAX_METHODS), key="any_n"))
    logx = h2.checkbox("Évaluations en échelle log", value=bool(saved.get("logx", False)), key="any_logx",
                       help="Linéaire : axe de 0 à 50 000. Log : de 100 à 50 000 (log 0 n'existe pas).")
    zoom = h3.checkbox(f"Zoom sur la fin ({ANYTIME_ZOOM_FROM // 1000}k → {ANYTIME_BUDGET // 1000}k)",
                       value=bool(saved.get("zoom", True)), key="any_zoom")
    band = h2.checkbox("Bande interquartile (nos méthodes)", value=bool(saved.get("band", False)), key="any_band",
                       help="Quartiles des 100 runs : mélange les 10 instances de la classe, d'où une bande large.")

    sel = []
    cols = st.columns(min(n_methods, 3))
    for i in range(n_methods):
        letter = "abcdefgh"[i]
        default, name = ((saved_methods[i]["config"], saved_methods[i]["name"]) if i < len(saved_methods)
                         else (all_configs[0], f"Méthode {i + 1}"))
        with cols[i % len(cols)]:
            idx = all_configs.index(default) if default in all_configs else 0
            cfg = st.selectbox(f"Config {letter.upper()}", all_configs, index=idx, key=f"any_{letter}")
            label = st.text_input("Nom", name, key=f"any_{letter}_label")
            sel.append((cfg, label))

    # Sélection d'une seule classe, par puces groupées par problème et dimension (comme l'onglet Favoris)
    st.markdown("**Classe à visualiser**")
    st.markdown(
        "<style>"
        '[data-testid="stPills"] button,'
        '[data-testid="stBaseButton-pills"],'
        '[data-testid="stBaseButton-pillsActive"]{'
        'padding:0.4rem 0.95rem !important;border-radius:10px !important;'
        'min-height:2.2rem !important;}'
        '[data-testid="stPills"] button p,'
        '[data-testid="stBaseButton-pills"] p,'
        '[data-testid="stBaseButton-pillsActive"] p{'
        'font-size:1.15rem !important;font-weight:600 !important;}'
        ".fav-curve-hdr{font-size:1.5rem;font-weight:700;margin:6px 0 2px;}"
        ".fav-curve-dim{font-size:1.2rem;font-weight:600;color:#555;margin:10px 0 2px;}"
        "</style>",
        unsafe_allow_html=True,
    )
    _iprob = lambda i: INSTANCE_RE.match(i).group("problem")
    _idim = lambda i: int(INSTANCE_RE.match(i).group("dim"))
    _it = lambda i: int(INSTANCE_RE.match(i).group("t"))
    probs = [p for p in ["NK", "NK3", "QUBO"] if any(_iprob(i) == p for i in sorted_instances)]
    pick_keys = {(p, d): f"any_pick_{p}_{d}" for p in probs
                 for d in sorted({_idim(i) for i in sorted_instances if _iprob(i) == p})}
    group_of = lambda c: pick_keys[(_iprob(c), _idim(c))]
    # Jusqu'à 2 classes : 1 -> figure seule ; 2 -> figure double (gauche = 1er clic, droite = 2e clic)
    if "any_sel" not in st.session_state:
        init = saved.get("pair") or ([saved["class"]] if saved.get("class") else ANYTIME_DEFAULT_CLASSES[:1])
        st.session_state["any_sel"] = [c for c in dict.fromkeys(init) if c in sorted_instances][:2]
        for k in pick_keys.values():
            st.session_state[k] = [c for c in st.session_state["any_sel"] if group_of(c) == k]

    def _any_pick_cb(changed: str):
        now = set(st.session_state.get(changed) or [])
        order = [c for c in st.session_state["any_sel"] if group_of(c) != changed or c in now]
        order += [c for c in sorted(now) if c not in order]           # clic ajouté en dernier
        order = order[-2:]                                              # 3e clic : remplace le plus ancien
        st.session_state["any_sel"] = order
        for k in pick_keys.values():
            st.session_state[k] = [c for c in order if group_of(c) == k]

    for pcol, p in zip(st.columns(len(probs), gap="large"), probs):
        with pcol:
            st.markdown(f'<div class="fav-curve-hdr">{p}</div>', unsafe_allow_html=True)
            lbl = "t" if p == "QUBO" else "K"
            p_insts = [i for i in sorted_instances if _iprob(i) == p]
            for d in sorted({_idim(i) for i in p_insts}):
                wkey = pick_keys[(p, d)]
                st.markdown(f'<div class="fav-curve-dim">DIM {d}</div>', unsafe_allow_html=True)
                st.pills(wkey, sorted((i for i in p_insts if _idim(i) == d), key=_it), selection_mode="multi",
                         key=wkey, format_func=lambda i, _l=lbl: f"{_l}{_it(i)}",
                         on_change=_any_pick_cb, args=(wkey,), label_visibility="collapsed")
    classes = list(st.session_state.get("any_sel", []))
    comp_style = _anytime_comp_style(tuple(sorted_instances), n_methods)

    state = {"n": n_methods, "logx": logx, "band": band, "zoom": zoom,
             "class": classes[0] if classes else saved.get("class"), "pair": classes,
             "methods": [{"config": c, "name": l} for c, l in sel] + saved_methods[n_methods:]}
    if state != saved:
        save_anytime(state)
    if not classes:
        st.info("Sélectionnez une ou deux classes ci-dessus (deux : figure double avec légende commune).")
        return

    def _panel(inst):
        """(méthodes, concurrents) d'une classe, prêts pour _anytime_figure."""
        m = INSTANCE_RE.match(inst)
        pb, dim, t = m.group("problem"), int(m.group("dim")), int(m.group("t"))
        rk = load_ranking(pb, dim, t)
        top = [] if rk is None or rk.empty else rk.nlargest(ANYTIME_TOP_COMPETITORS, "score")["name_algo"].tolist()
        methods = [(label, c) for cfg, label in sel if (c := _anytime_curve(cfg, inst)) is not None]
        comps = [(algo, c, *comp_style[algo]) for algo in top
                 if algo in comp_style and (c := load_competitor_curve(algo, pb, dim, t)) is not None]
        return methods, comps

    comp_names = list(dict.fromkeys(a for inst in classes for a, *_ in _panel(inst)[1]))
    st.caption("Concurrents externes : les " + str(ANYTIME_TOP_COMPETITORS) + " meilleurs du pool sur chaque "
               "classe (score final) — " + ", ".join(_short_algo(a) for a in comp_names))

    if len(classes) == 1:
        inst = classes[0]
        pdf_legend = st.checkbox("Légende dans le PDF", value=True, key="any_pdf_legend")
        methods, comps = _panel(inst)
        # Aperçu = la figure exportée, aux dimensions du PDF : mêmes proportions, police, légende et zoom
        fig = _anytime_figure(methods, comps, logx, export=True, band=band, zoom=zoom, legend=pdf_legend)
        st.caption(_sensi_caption(inst) + " — aperçu à l'échelle du PDF")
        st.plotly_chart(fig, use_container_width=False, theme=None, key=f"any_fig_{inst}",
                        config={"displayModeBar": False})
        st.download_button("⬇️ Export (Markdown)", key=f"any_md_{inst}", mime="text/markdown",
                           file_name=f"anytime_{inst}.md",
                           data=lambda: _anytime_markdown([(inst, methods, comps)], sel))
        st.download_button(
            "⬇️ PDF", key=f"any_dl_{inst}", mime="application/pdf", file_name=f"anytime_{inst}.pdf",
            data=lambda: _sensi_pdf(_anytime_figure(methods, comps, logx, export=True,
                                                    legend=pdf_legend, band=band, zoom=zoom)))
        return

    # Deux classes : figure double (gauche, droite) + légende commune dessous, aperçu = le PDF lui-même
    left, right = classes
    key = (left, right, logx, band, zoom, tuple(sel),
           *(p.stat().st_mtime for c, _ in sel for inst in (left, right)
             if (p := RESULTS_DIR / c / inst / "best_metrics.csv").exists()))
    cache = st.session_state.setdefault("_any_double_cache", {})
    if key not in cache:
        with st.spinner("Composition de la figure double…"):
            (ml, cl), (mr, cr) = _panel(left), _panel(right)
            union = list(dict.fromkeys([a for a, *_ in cl] + [a for a, *_ in cr]))
            # Légende sur deux lignes indépendantes (méthodes, puis concurrents), chacune à largeur naturelle :
            # une seule légende repliée aligne les entrées sur la plus longue et coupe les concurrents en deux
            method_entries = [(l, SENSI_COLORS[j % len(SENSI_COLORS)], "solid") for j, (_, l) in enumerate(sel)]
            comp_entries = [(_short_algo(a), *comp_style[a]) for a in union]
            legends = [_anytime_legend_pdf(e, max_width=100000) for e in (method_entries, comp_entries) if e]
            pdf = _compose_double_pdf(
                _sensi_pdf(_anytime_figure(ml, cl, logx, export=True, legend=False, band=band, zoom=zoom,
                                           title=_panel_title(left))),
                _sensi_pdf(_anytime_figure(mr, cr, logx, export=True, legend=False, band=band, zoom=zoom,
                                           title=_panel_title(right))),
                legends)
            cache.clear()
            cache[key] = (pdf, _pdf_to_png(pdf))
    pdf, png = cache[key]
    st.caption(f"{_sensi_caption(left)}  |  {_sensi_caption(right)} — aperçu du PDF")
    st.image(png, width=1300)
    # Correspondance des noms abrégés, à coller dans la légende LaTeX de la figure
    shown = list(dict.fromkeys(a for inst in (left, right) for a, *_ in _panel(inst)[1]))
    abbrev = [(_short_algo(a), a) for a in shown if _short_algo(a) != a]
    if abbrev:
        mapping = "; ".join(f"{short} = \\texttt{{{full}}}" for short, full in abbrev)
        st.markdown("**Correspondance des noms abrégés** (à coller dans ta `\\caption{…}`) :")
        st.code(f"Abbreviated Nevergrad names: {mapping}.", language="latex")
    st.download_button("⬇️ Export (Markdown)", key="any_double_md", mime="text/markdown",
                       file_name=f"anytime_{left}__{right}.md",
                       data=lambda: _anytime_markdown([(c, *_panel(c)) for c in (left, right)], sel))
    st.download_button("⬇️ Figure double + légende (PDF)", key="any_double_dl", mime="application/pdf",
                       file_name=f"anytime_{left}__{right}.pdf", data=pdf,
                       help="Un seul PDF, tout à la même échelle. Dans LaTeX : \\includegraphics[width=\\textwidth]{…}")


def _pdf_to_png(pdf: bytes, dpi: int = 110) -> bytes | None:
    """Rendu PNG d'un PDF (pdftoppm), pour l'aperçu exact dans le dashboard."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "f.pdf").write_bytes(pdf)
        try:
            subprocess.run(["pdftoppm", "-cropbox", "-png", "-r", str(dpi), "-singlefile",
                            str(Path(d) / "f.pdf"), str(Path(d) / "f")], check=True, capture_output=True, timeout=60)
            return (Path(d) / "f.png").read_bytes()
        except Exception:
            return None


def _compose_double_pdf(left: bytes, right: bytes, legend: bytes | list[bytes], gap_pt: int = 18) -> bytes:
    """Assemble deux panneaux côte à côte et la légende commune centrée dessous, avec pdflatex
    (classe standalone). Tout est inclus à sa taille naturelle : panneaux et légende gardent la même
    échelle, donc des polices identiques ; on redimensionne ensuite la figure entière dans le papier."""
    import subprocess, tempfile
    legends = legend if isinstance(legend, list) else [legend]   # une ligne de légende par PDF, empilées
    rows = r"\\[2pt]".join(r"\includegraphics{legend%d.pdf}" % k for k in range(len(legends)))
    tex = (r"\documentclass[border=0pt]{standalone}\usepackage{graphicx}\begin{document}"
           r"\begin{tabular}{@{}c@{}}"
           r"\includegraphics{left.pdf}\hspace{" + str(gap_pt) + r"pt}\includegraphics{right.pdf}\\[6pt]"
           + rows +
           r"\end{tabular}\end{document}")
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        (dd / "left.pdf").write_bytes(left); (dd / "right.pdf").write_bytes(right)
        for k, lg in enumerate(legends):
            (dd / f"legend{k}.pdf").write_bytes(lg)
        (dd / "fig.tex").write_text(tex)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"],
                       cwd=d, check=True, capture_output=True, timeout=120)
        return (dd / "fig.pdf").read_bytes()


ANYTIME_LEGEND_MAX_W = 1150   # légende commune : au-delà, les entrées passent à la ligne


def _anytime_legend_pdf(entries: list[tuple[str, str, str]], max_width: int = ANYTIME_LEGEND_MAX_W) -> bytes:
    """Légende seule : (nom, couleur, trait) — traits pleins pour nos méthodes, pointillés pour les concurrents.
    Largeur bornée : plusieurs lignes si besoin (sinon un bandeau trop large pour la page)."""
    labels = [e[0] for e in entries]
    width = min(_sensi_legend_width(labels), max_width)
    # Légende repliée : plotly aligne les entrées en colonnes de largeur égale (celle du nom le plus long)
    entry_w = max(len(l) for l in labels) * 0.55 * SENSI_LEGEND_FS + 110
    per_row = max(1, int((width - 40) // entry_w)) if width < _sensi_legend_width(labels) else len(labels)
    n_rows = -(-len(labels) // per_row)
    height = 90 * n_rows + 80          # large : le recadrage (_crop_pdf) retire ensuite le blanc
    fig = go.Figure()
    for label, color, dash in entries:
        fig.add_trace(go.Scatter(x=[None], y=[None], name=label, mode="lines",
                                 line=dict(color=color, width=3, dash=dash)))
    fig.update_layout(width=width, height=height, margin=dict(t=0, b=0, l=0, r=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      font=dict(family=SENSI_PDF_FONT, size=SENSI_LEGEND_FS, color="#262730"),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=0.5, yanchor="middle",
                                  itemwidth=60, bgcolor="rgba(0,0,0,0)"))
    return _crop_pdf(fig.to_image(format="pdf", width=width, height=height))


def tab_ablation(all_df: pd.DataFrame, sorted_instances: list) -> None:
    """Comme Favoris, mais les configs ne sont comparées qu'entre elles (pas aux concurrents)."""
    abls        = load_ablations()
    all_configs = all_df["config"].tolist()
    sel_idx     = st.session_state.get("abl_sel_idx")

    # ── Vue liste ─────────────────────────────────────────────────────────────
    if sel_idx is None or sel_idx >= len(abls):
        h1, h2 = st.columns([6, 1])
        h1.markdown("#### 🧪 Études d'ablation")
        if h2.button("+ Nouvelle", key="abl_new", use_container_width=True):
            abls.append({"name": f"Ablation {len(abls) + 1}", "configs": [],
                         "aliases": {}, "instances": [], "ref": None})
            save_ablations(abls)
            st.session_state["abl_sel_idx"] = len(abls) - 1
            st.rerun()
        if not abls:
            st.info("Aucune étude. Cliquez sur « + Nouvelle » pour commencer.")
            return
        grid = st.columns(3)
        for i, abl in enumerate(abls):
            with grid[i % 3]:
                with st.container(border=True):
                    t1, t2 = st.columns([5, 1])
                    if t1.button(f"🧪 {abl['name']}", key=f"abl_sel_{i}", use_container_width=True):
                        st.session_state["abl_sel_idx"] = i
                        st.rerun()
                    if t2.button("🗑", key=f"abl_del_{i}"):
                        abls.pop(i)
                        save_ablations(abls)
                        st.rerun()
                    n_cfg = len(abl.get("configs", []))
                    st.caption(f"{n_cfg} config{'s' if n_cfg > 1 else ''}")
        return

    # ── Vue détail ────────────────────────────────────────────────────────────
    abl = abls[sel_idx]
    hd1, hd2 = st.columns([1, 8])
    if hd1.button("← Retour", key="abl_back"):
        st.session_state.pop("abl_sel_idx", None)
        st.rerun()
    hd2.markdown(f"#### 🧪 {abl['name']}")

    new_name = st.text_input("Nom de l'étude", value=abl["name"], key=f"abl_name_{sel_idx}")
    if new_name and new_name != abl["name"]:
        abls[sel_idx]["name"] = new_name
        save_ablations(abls)

    st.divider()
    st.markdown("##### Variantes")
    configs = abl.get("configs", [])
    aliases = abl.setdefault("aliases", {})
    ac1, ac2 = st.columns([4, 1])
    add_cfg = ac1.selectbox("Ajouter une config", [""] + [c for c in all_configs if c not in configs],
                            key=f"abl_add_{sel_idx}")
    if ac2.button("Ajouter", key=f"abl_add_btn_{sel_idx}", use_container_width=True, disabled=not add_cfg):
        abls[sel_idx]["configs"].append(add_cfg)
        save_ablations(abls)
        st.rerun()

    if not configs:
        st.info("Aucune config ajoutée.")
        return

    for cfg in list(configs):
        r1, r2, r3 = st.columns([3, 3, 1])
        r1.code(cfg, language=None)
        alias = r2.text_input("Alias", value=aliases.get(cfg, ""), key=f"abl_alias_{sel_idx}_{cfg}",
                              placeholder="nom de la variante", label_visibility="collapsed")
        if alias != aliases.get(cfg, ""):
            abls[sel_idx]["aliases"][cfg] = alias
            save_ablations(abls)
        if r3.button("✕", key=f"abl_rm_{sel_idx}_{cfg}", use_container_width=True):
            abls[sel_idx]["configs"].remove(cfg)
            abls[sel_idx]["aliases"].pop(cfg, None)
            if abls[sel_idx].get("ref") == cfg:
                abls[sel_idx]["ref"] = None
            save_ablations(abls)
            st.rerun()

    def _lbl(cfg: str) -> str:
        if aliases.get(cfg):
            return aliases[cfg]
        row_df = all_df[all_df["config"] == cfg]
        return _config_label(row_df.iloc[0]) if not row_df.empty else cfg[-30:]

    ref = abl.get("ref") if abl.get("ref") in configs else configs[0]
    new_ref = st.selectbox("Référence (écarts et Wilcoxon calculés par rapport à elle)", configs,
                           index=configs.index(ref), format_func=_lbl, key=f"abl_ref_{sel_idx}")
    if new_ref != abl.get("ref"):
        abls[sel_idx]["ref"] = new_ref
        save_ablations(abls)
    ref = new_ref

    # ── Instances ─────────────────────────────────────────────────────────────
    saved_insts = [i for i in abl.get("instances", []) if i in sorted_instances] or list(sorted_instances)
    instances = st.multiselect("Instances", sorted_instances, default=saved_insts, key=f"abl_inst_{sel_idx}")
    if instances != abl.get("instances"):
        abls[sel_idx]["instances"] = instances
        save_ablations(abls)
    if not instances:
        st.info("Sélectionnez au moins une instance.")
        return

    # ── Données ───────────────────────────────────────────────────────────────
    _PROB_ORDER = {"NK": 0, "NK3": 1, "QUBO": 2}
    short = {cfg: "ABCDEFGHIJ"[i] for i, cfg in enumerate(configs)}
    rows = []
    for inst in instances:
        m = INSTANCE_RE.match(inst)
        prob, dim, t = m.group("problem"), int(m.group("dim")), int(m.group("t"))
        r = {"inst": inst, "prob": prob, "dim": dim, "t": t, "score": {}, "ham": {}, "means": {}}
        for cfg in configs:
            curve = load_curve(cfg, inst)
            if curve is not None:
                r["score"][cfg] = abs(float(curve["best_fitness"].iloc[-1]))
                if "avg_hamming" in curve.columns:
                    r["ham"][cfg] = float(curve["avg_hamming"].iloc[-1])
            r["means"][cfg] = _instance_means(load_raw_scores(cfg, inst))
        rows.append(r)
    rows.sort(key=lambda r: (_PROB_ORDER.get(r["prob"], 99), r["dim"], r["t"]))

    for r in rows:
        present = [c for c in configs if c in r["score"]]
        order = sorted(present, key=lambda c: -r["score"][c])
        r["rank"] = {c: order.index(c) + 1 for c in present}
        r["gap"] = ({c: (r["score"][c] - r["score"][ref]) / abs(r["score"][ref]) * 100 for c in present}
                    if ref in r["score"] else {})
        r["p"] = {c: _paired_wilcoxon(r["means"][c], r["means"][ref]) for c in configs if c != ref}

    # ── Résumé ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📊 Résumé")
    n_ok = len(rows)
    glob = _ablation_global_tests(rows, configs, ref)
    mcols = st.columns(len(configs))
    for col, cfg in zip(mcols, configs):
        wins  = sum(1 for r in rows if r["rank"].get(cfg) == 1)
        rks   = [r["rank"][cfg] for r in rows if cfg in r["rank"]]
        grk   = [_get_rank(r["prob"], r["dim"], r["t"], r["score"][cfg])
                 for r in rows if cfg in r["score"]]
        grk   = [g for g in grk if g is not None]
        with col.container(border=True):
            st.markdown(f"**{short[cfg]} — {_lbl(cfg)}**" + (" · réf." if cfg == ref else ""))
            st.metric("Meilleure des variantes", f"{wins} / {n_ok}")
            st.metric("Rang moyen entre variantes", f"{np.mean(rks):.2f}" if rks else "—")
            st.metric("Rang moyen vs concurrents", f"{np.mean(grk):.1f}" if grk else "—")
            if cfg != ref:
                better = sum(1 for r in rows if r["p"].get(cfg) is not None and r["p"][cfg] < 0.05
                             and r["gap"].get(cfg, 0) > 0)
                worse  = sum(1 for r in rows if r["p"].get(cfg) is not None and r["p"][cfg] < 0.05
                             and r["gap"].get(cfg, 0) < 0)
                st.caption(f"vs réf. (Wilcoxon p<0.05) : ▲ {better} · ▼ {worse}")
                g = glob.get(cfg)
                if g:
                    arrow = ("▲" if g["median"] > 0 else "▼") if g["p_holm"] < 0.05 else "="
                    st.caption(f"Global ({g['n']} classes, Holm) : {arrow} p = {g['p_holm']:.2g} · "
                               f"écart médian {g['median']:+.2f} %")

    # Écart moyen à la référence par famille × dimension
    st.markdown(f"##### Écart moyen à la référence ({_lbl(ref)}) par famille et dimension")
    gap_rows = []
    for prob in ["NK", "NK3", "QUBO"]:
        for dim in [64, 128, 256, 512]:
            sub = [r for r in rows if r["prob"] == prob and r["dim"] == dim]
            if not sub:
                continue
            line = {"Famille": prob, "n": dim}
            for cfg in configs:
                if cfg == ref:
                    continue
                g = [r["gap"][cfg] for r in sub if cfg in r["gap"]]
                line[f"{short[cfg]} — {_lbl(cfg)}"] = f"{np.mean(g):+.2f} %" if g else ""
            gap_rows.append(line)
    if gap_rows:
        st.dataframe(pd.DataFrame(gap_rows), hide_index=True, use_container_width=True)

    # ── Tableau par instance : un tableau par problème, comme Favoris ─────────
    st.markdown("##### Par instance")
    for cfg in configs:
        st.caption(f"{short[cfg]} = {_lbl(cfg)}" + ("  (référence)" if cfg == ref else ""))
    st.caption("Vert = meilleur score · rouge = pire · Écart = vs référence, ▲/▼ = significatif "
               "(Wilcoxon apparié sur les moyennes par instance, p<0.05)")

    col_specs: list[tuple[str, str, str]] = []   # (entête, config, type)
    for cfg in configs:
        s_ = _lbl(cfg)
        col_specs.append((f"Score {s_}", cfg, "score"))
        if cfg != ref:
            col_specs.append((f"Écart {s_}", cfg, "gap"))

    th   = "padding:8px 28px;text-align:right;border-bottom:2px solid #ccc;white-space:nowrap;font-size:19px;"
    th_l = "padding:8px 28px;text-align:left;border-bottom:2px solid #ccc;white-space:nowrap;font-size:19px;"
    td_l = "padding:7px 28px;border-bottom:1px solid #f0f0f0;white-space:nowrap;font-size:18px;"
    td_r = "padding:7px 28px;border-bottom:1px solid #f0f0f0;white-space:nowrap;font-size:18px;text-align:right;"
    td_sep = "padding:18px 10px 5px;font-weight:bold;font-size:19px;color:#333;border-top:2px solid #bbb;"

    active_probs = [p for p in ["NK", "NK3", "QUBO"] if any(r["prob"] == p for r in rows)]
    # Empilés (pas côte à côte) : avec 4 variantes, ~11 colonnes par problème ne tiennent pas en 3 colonnes
    for prob in active_probs:
        prob_rows = [r for r in rows if r["prob"] == prob]
        fmt = ".2f" if prob == "QUBO" else ".4f"
        with st.container():
            st.markdown(f"##### {prob}")
            ncols = 1 + len(col_specs)
            html = '<table style="border-collapse:collapse;">'
            html += f'<thead><tr><th style="{th_l}">Instance</th>'
            for hdr, _, _ in col_specs:
                html += f'<th style="{th}">{hdr}</th>'
            html += "</tr></thead><tbody>"
            current_dim = None
            for r in prob_rows:
                if r["dim"] != current_dim:
                    current_dim = r["dim"]
                    html += f'<tr><td colspan="{ncols}" style="{td_sep}">DIM {current_dim}</td></tr>'
                html += f'<tr><td style="{td_l}">{r["inst"]}</td>'
                for _, cfg, kind in col_specs:
                    if kind == "score":
                        v = r["score"].get(cfg)
                        cell = "" if v is None else format(v, fmt)
                        n_present = len(r["rank"])
                        if v is not None and n_present > 1 and r["rank"].get(cfg) == 1:
                            cell = f'<b style="color:#2ca02c">{cell}</b>'
                        elif v is not None and n_present > 1 and r["rank"].get(cfg) == n_present:
                            cell = f'<b style="color:#d62728">{cell}</b>'
                    elif kind == "gap":
                        g, pv = r["gap"].get(cfg), r["p"].get(cfg)
                        cell = "" if g is None else f"{g:+.2f}%"
                        if g is not None and pv is not None and pv < 0.05:
                            cell += (' <span style="color:#2ca02c">▲</span>' if g > 0
                                     else ' <span style="color:#d62728">▼</span>')
                    else:
                        h = r["ham"].get(cfg)
                        cell = "" if h is None else f"{h:.2f}"
                    html += f'<td style="{td_r}">{cell}</td>'
                html += "</tr>"
            html += "</tbody></table>"
            st.markdown(f'<div style="overflow-x:auto;margin-bottom:18px;">{html}</div>',
                        unsafe_allow_html=True)

    # ── Courbes : même sélecteur que Favoris (puces par problème / dimension) ─
    st.divider()
    st.markdown("#### 📈 Courbes")
    st.markdown("**Instance à visualiser**")
    st.markdown(
        "<style>"
        '[data-testid="stPills"] button,'
        '[data-testid="stBaseButton-pills"],'
        '[data-testid="stBaseButton-pillsActive"]{'
        'padding:0.4rem 0.95rem !important;border-radius:10px !important;'
        'min-height:2.2rem !important;}'
        '[data-testid="stPills"] button p,'
        '[data-testid="stBaseButton-pills"] p,'
        '[data-testid="stBaseButton-pillsActive"] p{'
        'font-size:1.15rem !important;font-weight:600 !important;}'
        ".fav-curve-hdr{font-size:1.5rem;font-weight:700;margin:6px 0 2px;}"
        ".fav-curve-dim{font-size:1.2rem;font-weight:600;color:#555;margin:10px 0 2px;}"
        "</style>",
        unsafe_allow_html=True,
    )
    _iprob = lambda i: INSTANCE_RE.match(i).group("problem")
    _idim  = lambda i: int(INSTANCE_RE.match(i).group("dim"))
    _it    = lambda i: int(INSTANCE_RE.match(i).group("t"))
    probs_curve = [p for p in ["NK", "NK3", "QUBO"] if any(_iprob(i) == p for i in instances)]
    sel_key = f"abl_curve_sel_{sel_idx}"
    all_pick_keys = [
        f"abl_curve_pick_{sel_idx}_{p}_{d}"
        for p in probs_curve
        for d in sorted({_idim(i) for i in instances if _iprob(i) == p})
    ]

    def _abl_pick_cb(changed: str):
        v = st.session_state.get(changed)
        st.session_state[sel_key] = v or None
        if v:                                   # un clic sélectionne → vide les autres groupes
            for k in all_pick_keys:
                if k != changed:
                    st.session_state[k] = None

    for pcol, p in zip(st.columns(len(probs_curve), gap="large"), probs_curve):
        with pcol:
            st.markdown(f'<div class="fav-curve-hdr">{p}</div>', unsafe_allow_html=True)
            _plbl = "t" if p == "QUBO" else "K"
            p_insts = [i for i in instances if _iprob(i) == p]
            for d in sorted({_idim(i) for i in p_insts}):
                d_insts = sorted((i for i in p_insts if _idim(i) == d), key=_it)
                wkey = f"abl_curve_pick_{sel_idx}_{p}_{d}"
                st.markdown(f'<div class="fav-curve-dim">DIM {d}</div>', unsafe_allow_html=True)
                st.pills(
                    wkey, d_insts, selection_mode="single", key=wkey,
                    format_func=lambda i, _l=_plbl: f"{_l}{_it(i)}",
                    on_change=_abl_pick_cb, args=(wkey,),
                    label_visibility="collapsed",
                )

    inst = st.session_state.get(sel_key)
    if not inst:
        st.info("Sélectionnez une instance ci-dessus pour afficher ses courbes.")
        return
    curves = {cfg: load_curve(cfg, inst) for cfg in configs}
    raws   = {cfg: load_raw_scores(cfg, inst) for cfg in configs}
    has_ham = any(c is not None and "avg_hamming" in c.columns for c in curves.values())
    titles = ["Convergence"] + (["Hamming (diversité)"] if has_ham else []) + ["Distribution finale"]
    n_rows = len(titles)
    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=titles, vertical_spacing=0.09,
                        row_heights=[0.40, 0.25, 0.35] if n_rows == 3 else [0.6, 0.4])
    for i, cfg in enumerate(configs):
        color = COLORS[i % len(COLORS)]
        c = curves[cfg]
        if c is not None:
            fig.add_trace(go.Scatter(x=c["step"], y=_norm(c["best_fitness"]), name=_lbl(cfg),
                                     mode="lines", line=dict(color=color, width=2)), row=1, col=1)
            if has_ham and "avg_hamming" in c.columns:
                fig.add_trace(go.Scatter(x=c["step"], y=c["avg_hamming"], name=_lbl(cfg), mode="lines",
                                         line=dict(color=color, width=2, dash="dot"), showlegend=False),
                              row=2, col=1)
        s = raws[cfg]
        if s is not None:
            fig.add_trace(go.Box(y=s, x=[_lbl(cfg)] * len(s), name=_lbl(cfg), marker_color=color,
                                 boxpoints="outliers", boxmean=True, showlegend=False), row=n_rows, col=1)
    fig.update_xaxes(title_text="Évaluations", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    if has_ham:
        fig.update_xaxes(title_text="Évaluations", row=2, col=1)
        fig.update_yaxes(title_text="Hamming moy.", row=2, col=1)
    fig.update_xaxes(type="category", row=n_rows, col=1)
    fig.update_yaxes(title_text="Score final", row=n_rows, col=1)
    fig.update_layout(height=1150 if n_rows == 3 else 850, margin=dict(r=20, t=40, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.08))
    st.plotly_chart(fig, use_container_width=True, key=f"abl_fig_{sel_idx}_{inst}")


# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="SVGD-EDA Dashboard", layout="wide", page_icon="📊")
st.title("SVGD-EDA — Grid Search Dashboard")

df = load_summary()
if df.empty:
    st.error(f"Aucun résultat trouvé dans `{RESULTS_DIR}`")
    st.stop()

@st.cache_data(show_spinner=False)
def _has_raw(config_name: str) -> bool:
    d = RESULTS_DIR / config_name
    if not d.exists():
        return False
    return any((sub / "raw_scores.csv").exists() for sub in d.iterdir() if sub.is_dir())

df = df[df["config"].apply(_has_raw)].reset_index(drop=True)
if df.empty:
    st.warning("Aucune config avec `raw_scores.csv` trouvée.")
    st.stop()

# Configs incompletes (grid interrompue, timeout...) : leurs top1/rang moyen ne sont pas
# comparables a ceux des configs completes, on les ecarte.
EXPECTED_INSTANCES = 56
df = df[df["n_instances"] == EXPECTED_INSTANCES].reset_index(drop=True)
if df.empty:
    st.warning(f"Aucune config avec les {EXPECTED_INSTANCES} instances.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filtres")
    if st.button("🔄 Recalculer le cache", help="Supprime le cache disque et recalcule tout"):
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        st.rerun()

    def multisel(label, col):
        if col not in df.columns:
            return None
        vals = sorted(df[col].dropna().unique().tolist())
        return st.multiselect(label, vals, default=vals)

    pe_sel  = multisel("ppo_epochs",   "ppo_epochs")
    ce_sel  = multisel("clip_eps",     "clip_eps")
    eps_sel = multisel("epsilon_svgd", "epsilon_svgd")
    k_sel   = multisel("Kernel",       "kernel")
    m_sel   = multisel("M",            "M")
    l_sel   = multisel("lambda",       "lambda")

    max_rank = float(df["mean_rank"].max())
    rank_lim = st.slider("Rank moyen ≤", 1.0, max_rank, max_rank, step=0.5)
    show_n   = st.number_input("Configs max", 5, 2000, 200)

# Apply filters
mask = pd.Series(True, index=df.index)
for sel, col in [
    (pe_sel,  "ppo_epochs"),
    (ce_sel,  "clip_eps"),
    (eps_sel, "epsilon_svgd"),
    (k_sel,   "kernel"),
    (m_sel,   "M"),
    (l_sel,   "lambda"),
]:
    if sel is not None and col in df.columns:
        mask &= df[col].isin(sel) | df[col].isna()
mask &= df["mean_rank"] <= rank_lim
filtered = df[mask].head(int(show_n))

# ── Sidebar stats ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("##### Résumé sélection")
    st.metric("Configs", len(filtered))

    if not filtered.empty:
        c1, c2 = st.columns(2)
        c1.metric("Rank moy",  f"{filtered['mean_rank'].mean():.1f}")
        c2.metric("Top 1",     int(filtered["top1"].sum()))
        c1.metric("Top 3",     int(filtered["top3"].sum()))
        c2.metric("Top 5",     int(filtered["top5"].sum()))

        if "ppo_epochs" in filtered.columns and filtered["ppo_epochs"].nunique() > 1:
            st.markdown("**Rank moy par ppo_epochs**")
            breakdown = (
                filtered.groupby("ppo_epochs")[["mean_rank", "top1", "top3"]]
                .mean()
                .round(2)
                .rename(columns={"mean_rank": "rank_moy", "top1": "top1_moy", "top3": "top3_moy"})
            )
            st.dataframe(breakdown, use_container_width=True)

        if "epsilon_svgd" in filtered.columns and filtered["epsilon_svgd"].nunique() > 1:
            st.markdown("**Rank moy par epsilon_svgd**")
            breakdown_eps = (
                filtered.groupby("epsilon_svgd")[["mean_rank", "top1"]]
                .mean()
                .round(2)
                .rename(columns={"mean_rank": "rank_moy", "top1": "top1_moy"})
            )
            st.dataframe(breakdown_eps, use_container_width=True)

st.caption(f"{len(filtered)} configs affichées sur {len(df)} disponibles")

@st.cache_data(show_spinner=False)
def _collect_sorted_instances(configs: tuple[str, ...]) -> list[str]:
    """Union of instance dirs across all config dirs — configs no longer all
    share the same instances now that some, e.g. the PPSN baseline, have been
    extended to n=512 while others haven't been re-run there yet."""
    instance_set: set[str] = set()
    for cfg in configs:
        d = RESULTS_DIR / cfg
        if d.exists():
            instance_set.update(
                sub.name for sub in d.iterdir()
                if sub.is_dir() and INSTANCE_RE.match(sub.name)
            )
    return sorted(instance_set)


sorted_instances = _collect_sorted_instances(tuple(df["config"]))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    ["📋 Classement", "⚖️ Comparaison", "⭐ Favoris", "🧪 Ablation", "🎚️ Sensi", "⏱️ Anytime"]
)

with tab1:
    tab_classement(filtered)

with tab5:
    tab_comparaison(df, sorted_instances)

with tab6:
    tab_favoris(df, sorted_instances)

with tab7:
    tab_ablation(df, sorted_instances)

with tab8:
    tab_sensi(df, sorted_instances)

with tab9:
    tab_anytime(df, sorted_instances)
