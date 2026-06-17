"""
PPO-EDA Grid Search Dashboard
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

INSTANCE_RE = re.compile(r"^(?P<problem>QUBO|NK|NK3)_dim(?P<dim>\d+)_t(?P<t>\d+)$")
SKIP_ALGOS = {"ppo-eda", "tabu", "svgd-eda"}

PARSERS = [
    ("adv", "advantage", str),
    ("eps", "epsilon_svgd", float),
    ("ds",  "decay_start_ratio", float),
    ("dm",  "decay_min_factor", float),
    ("bw",  "bandwith_kernel", float),
    ("ks",  "ppo_epochs", int),
    ("pe",  "ppo_epochs", int),   # legacy (anciens résultats)
    ("ce",  "clip_eps", float),
    ("M",   "M", int),
    ("L",   "lambda", int),
    ("g",   "gamma", float),
    ("k",   "kernel", str),
]

COLORS = [
    "#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

CURVE_PARAMS = [
    ("ppo_epochs",        "ppo_epochs"),
    ("clip_eps",          "clip_eps"),
    ("epsilon_svgd",      "epsilon_svgd"),
    ("gamma",             "gamma"),
    ("M",                 "M"),
    ("lambda",            "lambda"),
    ("kernel",            "kernel"),
    ("advantage",         "advantage"),
    ("decay_start_ratio", "decay_start_ratio"),
    ("decay_min_factor",  "decay_min_factor"),
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
        if not metrics.exists():
            continue
        try:
            df = pd.read_csv(metrics)
            score = abs(float(df["best_fitness"].iloc[-1]))
            hamming = float(df["avg_hamming"].iloc[-1]) if "avg_hamming" in df.columns else None
        except Exception:
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
    if not new_dirs:
        if _instances_ok:
            return existing
        new_dirs = all_dirs
        instances_only = True

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


def load_curve(config: str, instance: str) -> pd.DataFrame | None:
    p = RESULTS_DIR / config / instance / "best_metrics.csv"
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:
        return None


def load_raw_scores(config: str, instance: str) -> pd.Series | None:
    p = RESULTS_DIR / config / instance / "raw_scores.csv"
    try:
        return pd.read_csv(p)["score"].abs() if p.exists() else None
    except Exception:
        return None


def load_ranking(problem: str, dim: int, t: int) -> pd.DataFrame | None:
    fname = f"UBQP_N_{dim}_K_{t}_ranks.csv" if problem == "QUBO" else f"{problem}_N_{dim}_K_{t}_ranks.csv"
    p = RANKING_DIR / fname
    try:
        df = pd.read_csv(p) if p.exists() else None
        if df is None:
            return None
        return df[~df["name_algo"].str.lower().isin(SKIP_ALGOS)]
    except Exception:
        return None


def load_comparisons() -> list[dict]:
    if COMPARISONS_FILE.exists():
        try:
            return json.loads(COMPARISONS_FILE.read_text())
        except Exception:
            return []
    return []


def save_comparisons(comps: list[dict]) -> None:
    COMPARISONS_FILE.write_text(json.dumps(comps, indent=2))


from plotly.subplots import make_subplots


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
    ("bandwith_kernel", "bw"),
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


def _vs_top5_figure(sel_cfg: str, instance: str, label: str,
                    problem: str, dim: int, t: int) -> go.Figure | None:
    """Combined curve (top) + boxplot (bottom) figure vs global top 5."""
    curve  = load_curve(sel_cfg, instance)
    scores = load_raw_scores(sel_cfg, instance)
    if curve is None and scores is None:
        return None

    ranking = load_ranking(problem, dim, t)
    top5 = []
    if ranking is not None:
        for _, row in ranking.nlargest(5, "score").iterrows():
            top5.append((row["name_algo"], float(row["score"])))

    has_box = scores is not None
    n_rows  = 2 if (curve is not None and has_box) else 1
    titles  = []
    if curve is not None:
        titles.append("Convergence")
    if has_box:
        titles.append("Distribution finale")

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=titles,
        row_heights=[0.6, 0.4] if n_rows == 2 else [1.0],
        vertical_spacing=0.12,
    )
    curve_row = 1
    box_row   = 2 if (curve is not None and has_box) else 1

    # ── Curve panel ──────────────────────────────────────────────────────────
    if curve is not None:
        fig.add_trace(go.Scatter(
            x=curve["step"], y=_norm(curve["best_fitness"]),
            name=label, line=dict(color=COLORS[0], width=3), mode="lines",
        ), row=curve_row, col=1)
        for j, (algo, score) in enumerate(top5):
            color = COLORS[(j + 1) % len(COLORS)]
            fig.add_trace(go.Scatter(
                x=[curve["step"].iloc[0], curve["step"].iloc[-1]],
                y=[score, score],
                name=algo, mode="lines",
                line=dict(dash="dash", color=color, width=1.5),
            ), row=curve_row, col=1)
        fig.update_xaxes(title_text="Évaluations", row=curve_row, col=1)
        fig.update_yaxes(title_text="Score",        row=curve_row, col=1)

    # ── Boxplot panel ─────────────────────────────────────────────────────────
    if has_box:
        fig.add_trace(go.Box(
            y=scores, name=label,
            marker_color=COLORS[0], boxpoints="outliers", showlegend=False,
        ), row=box_row, col=1)
        for j, (algo, score) in enumerate(top5):
            color = COLORS[(j + 1) % len(COLORS)]
            fig.add_hline(
                y=score, row=box_row, col=1,
                line=dict(dash="dash", color=color, width=1.5),
                annotation_text=algo, annotation_position="bottom right",
            )
        fig.update_yaxes(title_text="Score final", row=box_row, col=1)

    fig.update_layout(
        height=720 if n_rows == 2 else 440,
        margin=dict(r=140, t=40, b=10),
        legend=dict(orientation="v", x=1.02, y=1),
    )
    return fig


# ── Tab fragments ─────────────────────────────────────────────────────────────
# Each tab is a @st.fragment: interacting with its widgets re-runs only that
# function, not the full script — other tabs are unaffected.

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

    cols = [c for c in [
        *stat_cols,
        "ppo_epochs", "clip_eps",
        "epsilon_svgd", "gamma", "M", "lambda",
        "kernel", "advantage",
        "decay_start_ratio", "decay_min_factor",
        "n_instances", "config",
    ] if c in display.columns]

    st.dataframe(
        display[cols].reset_index(drop=True),
        use_container_width=True,
        height=620,
        column_config={
            "config":         st.column_config.TextColumn("Config",     width="large"),
            "mean_rank":      st.column_config.NumberColumn("Rank moy", format="%.2f"),
            "mean_rank_NK":   st.column_config.NumberColumn("Rank NK",  format="%.2f"),
            "mean_rank_NK3":  st.column_config.NumberColumn("Rank NK3", format="%.2f"),
            "mean_rank_QUBO": st.column_config.NumberColumn("Rank QUBO",format="%.2f"),
            "median_rank":    st.column_config.NumberColumn("Rank med", format="%.1f"),
            "ppo_epochs":     st.column_config.NumberColumn("ks",       format="%d"),
            "clip_eps":       st.column_config.NumberColumn("ce",       format="%.2f"),
            "epsilon_svgd":   st.column_config.NumberColumn("ε_svgd",  format="%.4f"),
            "gamma":          st.column_config.NumberColumn("γ",        format="%.4f"),
        },
    )


@st.fragment
def tab_courbes(filtered: pd.DataFrame, sorted_instances: list) -> None:
    # Reset in-tab widgets when the parent filtered set changes
    sig = tuple(filtered["config"].tolist())
    if st.session_state.get("_curve_sig") != sig:
        st.session_state["_curve_sig"] = sig
        for _, col in CURVE_PARAMS:
            st.session_state.pop(f"curve_{col}", None)
        st.session_state.pop("curve_inst", None)
        st.session_state.pop("curve_vs_inst", None)

    mode = st.radio("Mode", ["Multi-config", "vs Top 5"], horizontal=True, key="curve_mode")

    # ── Mode vs Top 5 ──────────────────────────────────────────────────────────
    if mode == "vs Top 5":
        cfg_list = filtered["config"].tolist()
        if not cfg_list:
            st.info("Aucune config disponible.")
            return

        c1, c2 = st.columns([3, 2])
        with c1:
            labels = [_config_label(filtered[filtered["config"] == c].iloc[0]) for c in cfg_list]
            sel_idx = st.selectbox(
                "Config", range(len(cfg_list)),
                format_func=lambda i: labels[i],
                key="curve_vs_idx",
            )
            sel_cfg = cfg_list[sel_idx]
        with c2:
            vs_inst = st.selectbox("Instance", sorted_instances, key="curve_vs_inst")

        if not vs_inst:
            return
        m = INSTANCE_RE.match(vs_inst)
        problem, dim, t = m.group("problem"), int(m.group("dim")), int(m.group("t"))

        fig = _vs_top5_figure(sel_cfg, vs_inst, labels[sel_idx], problem, dim, t)
        if fig is None:
            st.warning("Pas de données pour cette config / cette instance.")
        else:
            st.plotly_chart(fig, use_container_width=True)
        return

    # ── Mode Multi-config ──────────────────────────────────────────────────────
    st.markdown("##### Filtres configs")
    filter_cols = st.columns(5)
    curve_mask = pd.Series(True, index=filtered.index)
    active_filters: dict[str, list] = {}

    for idx, (label, col) in enumerate(CURVE_PARAMS):
        if col not in filtered.columns:
            continue
        vals = sorted(filtered[col].dropna().unique().tolist())
        if len(vals) <= 1:
            continue
        with filter_cols[idx % 5]:
            sel = st.multiselect(label, vals, default=vals, key=f"curve_{col}")
        if not sel:
            sel = vals
        curve_mask &= filtered[col].isin(sel) | filtered[col].isna()
        active_filters[col] = sel

    curve_filtered = filtered[curve_mask]
    sel_configs = curve_filtered["config"].tolist()
    st.caption(f"{len(sel_configs)} configs sélectionnées")

    varying = [col for col, sel in active_filters.items() if len(sel) > 1 and col in filtered.columns]

    def curve_label(cfg_row: pd.Series) -> str:
        if not varying:
            return cfg_row["config"][-50:]
        return "  ".join(f"{p}={cfg_row[p]}" for p in varying if pd.notna(cfg_row.get(p)))

    c1, c2 = st.columns([3, 1])
    with c1:
        instance = st.selectbox("Instance", sorted_instances, key="curve_inst")
    with c2:
        show_top5 = st.checkbox("Top 5 classement global", value=True)

    if sel_configs and instance:
        m = INSTANCE_RE.match(instance)
        problem, dim, t = m.group("problem"), int(m.group("dim")), int(m.group("t"))

        fig = go.Figure()
        for i, cfg in enumerate(sel_configs):
            curve = load_curve(cfg, instance)
            if curve is None:
                continue
            row = curve_filtered[curve_filtered["config"] == cfg].iloc[0]
            label = curve_label(row)
            fig.add_trace(go.Scatter(
                x=curve["step"],
                y=_norm(curve["best_fitness"]),
                name=label,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                mode="lines",
            ))

        if show_top5:
            ranking = load_ranking(problem, dim, t)
            if ranking is not None:
                for _, row in ranking.nlargest(5, "score").iterrows():
                    fig.add_hline(
                        y=row["score"],
                        line=dict(dash="dash", color="gray", width=1),
                        annotation_text=row["name_algo"],
                        annotation_position="bottom right",
                    )

        fig.update_layout(
            xaxis_title="Évaluations",
            yaxis_title="Score",
            legend=dict(orientation="h", yanchor="top", y=-0.15),
            height=520,
            margin=dict(r=20, t=20),
        )
        st.plotly_chart(fig, use_container_width=True)


@st.fragment
def tab_boxplots(filtered: pd.DataFrame, sorted_instances: list) -> None:
    # Reset in-tab widgets when the parent filtered set changes
    sig = tuple(filtered["config"].tolist())
    if st.session_state.get("_box_sig") != sig:
        st.session_state["_box_sig"] = sig
        for _, col in CURVE_PARAMS:
            st.session_state.pop(f"box_{col}", None)
        st.session_state.pop("box_inst", None)
        st.session_state.pop("box_vs_inst", None)

    mode = st.radio("Mode", ["Multi-config", "vs Top 5"], horizontal=True, key="box_mode")

    # ── Mode vs Top 5 ──────────────────────────────────────────────────────────
    if mode == "vs Top 5":
        cfg_list = filtered["config"].tolist()
        if not cfg_list:
            st.info("Aucune config disponible.")
            return

        c1, c2 = st.columns([3, 2])
        with c1:
            labels = [_config_label(filtered[filtered["config"] == c].iloc[0]) for c in cfg_list]
            sel_idx = st.selectbox(
                "Config", range(len(cfg_list)),
                format_func=lambda i: labels[i],
                key="box_vs_idx",
            )
            sel_cfg = cfg_list[sel_idx]
        with c2:
            vs_inst = st.selectbox("Instance", sorted_instances, key="box_vs_inst")

        if not vs_inst:
            return
        m = INSTANCE_RE.match(vs_inst)
        problem, dim, t = m.group("problem"), int(m.group("dim")), int(m.group("t"))

        fig = _vs_top5_figure(sel_cfg, vs_inst, labels[sel_idx], problem, dim, t)
        if fig is None:
            st.warning("Pas de données pour cette config / cette instance.")
        else:
            st.plotly_chart(fig, use_container_width=True)
        return

    # ── Mode Multi-config ──────────────────────────────────────────────────────
    st.markdown("##### Filtres configs")
    box_filter_cols = st.columns(5)
    box_mask = pd.Series(True, index=filtered.index)
    box_active: dict[str, list] = {}

    for idx, (label, col) in enumerate(CURVE_PARAMS):
        if col not in filtered.columns:
            continue
        vals = sorted(filtered[col].dropna().unique().tolist())
        if len(vals) <= 1:
            continue
        with box_filter_cols[idx % 5]:
            sel = st.multiselect(label, vals, default=vals, key=f"box_{col}")
        if not sel:
            sel = vals
        box_mask &= filtered[col].isin(sel) | filtered[col].isna()
        box_active[col] = sel

    box_filtered = filtered[box_mask]
    box_configs = box_filtered["config"].tolist()
    box_varying = [col for col, sel in box_active.items() if len(sel) > 1 and col in filtered.columns]

    def box_label(cfg_row: pd.Series) -> str:
        if not box_varying:
            return cfg_row["config"][-40:]
        return "  ".join(f"{p}={cfg_row[p]}" for p in box_varying if pd.notna(cfg_row.get(p)))

    box_instance = st.selectbox("Instance", sorted_instances, key="box_inst")

    if box_configs and box_instance:
        # Only show configs that have raw_scores.csv for this instance
        available = [cfg for cfg in box_configs
                     if (RESULTS_DIR / cfg / box_instance / "raw_scores.csv").exists()]
        st.caption(f"{len(available)}/{len(box_configs)} configs ont des scores bruts pour cette instance")

        if not available:
            st.warning("Aucun fichier `raw_scores.csv` pour ces configs / cette instance.")
        else:
            fig = go.Figure()
            for i, cfg in enumerate(available):
                scores = load_raw_scores(cfg, box_instance)
                if scores is None:
                    continue
                row = box_filtered[box_filtered["config"] == cfg].iloc[0]
                label = box_label(row)
                fig.add_trace(go.Box(
                    y=scores,
                    name=label,
                    marker_color=COLORS[i % len(COLORS)],
                    boxpoints="outliers",
                ))
            fig.update_layout(
                yaxis_title="Score final",
                height=520,
                showlegend=False,
                margin=dict(t=20),
            )
            st.plotly_chart(fig, use_container_width=True)


@st.fragment
def tab_analyse(filtered: pd.DataFrame) -> None:
    # ── Selectors ─────────────────────────────────────────────────────────────
    sc1, sc2, sc3 = st.columns([2, 1, 3])
    with sc1:
        problem_sel = st.radio(
            "Problème", ["Tous", "NK", "NK3", "QUBO"],
            horizontal=True, key="analyse_problem",
        )

    # Load per-instance data for size filtering
    inst_all = pd.DataFrame()
    if INSTANCES_FILE.exists():
        try:
            inst_all = pd.read_parquet(INSTANCES_FILE)
            inst_all = inst_all[inst_all["config"].isin(filtered["config"])]
        except Exception:
            pass

    if not inst_all.empty:
        if problem_sel != "Tous":
            inst_prob = inst_all[inst_all["problem"] == problem_sel]
        else:
            inst_prob = inst_all
        avail_dims = sorted(inst_prob["dim"].unique().tolist())
    else:
        inst_prob = pd.DataFrame()
        avail_dims = []

    with sc2:
        dim_sel = st.selectbox(
            "Taille",
            ["Toutes"] + [str(d) for d in avail_dims],
            key="analyse_dim",
        )

    rank_col    = {"Tous": "mean_rank",      "NK": "mean_rank_NK",      "NK3": "mean_rank_NK3",      "QUBO": "mean_rank_QUBO"}[problem_sel]
    top1_col    = {"Tous": "top1",           "NK": "top1_NK",           "NK3": "top1_NK3",           "QUBO": "top1_QUBO"}[problem_sel]
    hamming_col = {"Tous": "mean_hamming",   "NK": "mean_hamming_NK",   "NK3": "mean_hamming_NK3",   "QUBO": "mean_hamming_QUBO"}[problem_sel]

    analyse_df = filtered.dropna(subset=[rank_col])
    available = [(lbl, col) for lbl, col in CURVE_PARAMS
                 if col in analyse_df.columns and analyse_df[col].nunique() > 1]

    if not available:
        st.info("Pas assez de variabilité dans les configs filtrées pour une analyse.")
        return

    with sc3:
        param_sel = st.selectbox(
            "Paramètre à analyser",
            [lbl for lbl, _ in available],
            key="analyse_param",
        )

    param_col = next(col for lbl, col in available if lbl == param_sel)

    # ── Aggregation ───────────────────────────────────────────────────────────
    use_inst = dim_sel != "Toutes" and not inst_prob.empty

    if use_inst:
        # Per-instance aggregation filtered by size
        inst_filtered = inst_prob[inst_prob["dim"] == int(dim_sel)]
        inst_joined = inst_filtered.merge(
            filtered[["config", param_col]].drop_duplicates(), on="config"
        )
        agg_dict: dict = {
            "rank_moy": ("rank", "mean"),
            "top1_moy": ("rank", lambda x: int((x == 1).sum())),
            "n":        ("config", "nunique"),
        }
        has_hamming = "hamming" in inst_joined.columns and inst_joined["hamming"].notna().any()
        if has_hamming:
            agg_dict["hamming_moy"] = ("hamming", "mean")
        grp = (
            inst_joined.groupby(param_col)
            .agg(**agg_dict)
            .round(2)
            .reset_index()
            .sort_values("rank_moy")
        )
    else:
        agg_dict = {
            "rank_moy": (rank_col, "mean"),
            "top1_moy": (top1_col, "mean"),
            "n":        ("config", "count"),
        }
        has_hamming = hamming_col in analyse_df.columns and analyse_df[hamming_col].notna().any()
        if has_hamming:
            agg_dict["hamming_moy"] = (hamming_col, "mean")
        grp = (
            analyse_df.groupby(param_col)
            .agg(**agg_dict)
            .round(2)
            .reset_index()
            .sort_values("rank_moy")
        )

    x = grp[param_col].astype(str)
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(grp))]

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_specs = [
        ("rank_moy",  "Rank moyen"),
        ("top1_moy",  "Top 1 moyen"),
    ]
    if has_hamming:
        chart_specs.append(("hamming_moy", "Hamming moyen"))

    n_charts = len(chart_specs)
    tick_angle = -35 if len(grp) > 4 else 0

    fig = make_subplots(
        rows=1, cols=n_charts,
        subplot_titles=[t for _, t in chart_specs],
        horizontal_spacing=0.10,
    )
    for ci, (y_col, y_title) in enumerate(chart_specs, 1):
        vals = grp[y_col].tolist()
        fig.add_trace(go.Bar(
            x=x,
            y=vals,
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
            textfont=dict(size=11, color="#444"),
            marker=dict(color=bar_colors, opacity=0.85, line=dict(width=0)),
            hovertemplate="%{x}<br><b>" + y_title + "</b>: %{y:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=ci)
        fig.update_yaxes(
            title_text=y_title, row=1, col=ci,
            gridcolor="#ebebeb", showgrid=True, zeroline=False,
            title_font=dict(size=12, color="#555"),
        )
        fig.update_xaxes(
            tickangle=tick_angle, row=1, col=ci,
            showgrid=False, tickfont=dict(size=11),
        )

    fig.update_layout(
        height=480,
        margin=dict(t=60, b=10, l=10, r=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="sans-serif", size=12),
        bargap=0.32,
    )
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=14, color="#333", family="sans-serif"), y=ann.y + 0.02)

    st.plotly_chart(fig, use_container_width=True, key="analyse_fig")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.divider()
    rename_map = {param_col: param_sel, "rank_moy": "Rank moy", "top1_moy": "Top 1 moy", "n": "N configs"}
    if has_hamming:
        rename_map["hamming_moy"] = "Hamming moy"
    col_cfg: dict = {
        "Rank moy":    st.column_config.NumberColumn("Rank moy",    format="%.2f"),
        "Top 1 moy":   st.column_config.NumberColumn("Top 1 moy",   format="%.2f"),
        "Hamming moy": st.column_config.NumberColumn("Hamming moy", format="%.2f"),
    }
    st.dataframe(
        grp.rename(columns=rename_map),
        use_container_width=True,
        hide_index=True,
        column_config=col_cfg,
    )


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

    if not instances:
        st.info("Sélectionnez au moins une instance.")
        return

    label_a, label_b = _diff_labels(cfg_a, cfg_b)

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
        m_inst = INSTANCE_RE.match(inst)
        prob, dim, t = m_inst.group("problem"), int(m_inst.group("dim")), int(m_inst.group("t"))
        ra = _get_rank(prob, dim, t, sa)
        rb = _get_rank(prob, dim, t, sb)
        summary_rows.append({
            "Instance":     inst,
            col_score_a:    round(sa, 4),
            col_score_b:    round(sb, 4),
            "Gap %":        round(gap, 2),
            col_hamming_a:  round(ha, 2) if ha is not None else None,
            col_hamming_b:  round(hb, 2) if hb is not None else None,
            col_rank_a:     ra,
            col_rank_b:     rb,
            "Résultat":     f"{label_a} ✓" if gap > 0 else (f"{label_b} ✓" if gap < 0 else "égalité"),
        })

    if summary_rows:
        with st.expander("📊 Résumé", expanded=True):
            wins_a     = sum(1 for r in summary_rows if r["Résultat"] == f"{label_a} ✓")
            wins_b     = sum(1 for r in summary_rows if r["Résultat"] == f"{label_b} ✓")
            mean_gap   = sum(r["Gap %"] for r in summary_rows) / len(summary_rows)
            ranks_a    = [r[col_rank_a]    for r in summary_rows if r[col_rank_a]    is not None]
            ranks_b    = [r[col_rank_b]    for r in summary_rows if r[col_rank_b]    is not None]
            hammings_a = [r[col_hamming_a] for r in summary_rows if r[col_hamming_a] is not None]
            hammings_b = [r[col_hamming_b] for r in summary_rows if r[col_hamming_b] is not None]

            top1_a = sum(1 for r in summary_rows if r[col_rank_a] == 1)
            top1_b = sum(1 for r in summary_rows if r[col_rank_b] == 1)

            mc = st.columns(11)
            mc[0].metric(f"Victoires {label_a}", wins_a)
            mc[1].metric(f"Top 1 {label_a}", top1_a)
            mc[2].metric(f"Victoires {label_b}", wins_b)
            mc[3].metric(f"Top 1 {label_b}", top1_b)
            mc[4].metric("Gap moyen", f"{mean_gap:+.2f}%",
                         help=f"+ si {label_a} meilleur, − si {label_b} meilleur")
            if ranks_a:
                mc[5].metric(f"Rank moy {label_a}", f"{sum(ranks_a)/len(ranks_a):.1f}")
                mc[6].metric(f"Rank méd {label_a}", f"{sorted(ranks_a)[len(ranks_a)//2]}")
            if ranks_b:
                mc[7].metric(f"Rank moy {label_b}", f"{sum(ranks_b)/len(ranks_b):.1f}")
                mc[8].metric(f"Rank méd {label_b}", f"{sorted(ranks_b)[len(ranks_b)//2]}")
            if hammings_a:
                mc[9].metric(f"Hamming {label_a}", f"{sum(hammings_a)/len(hammings_a):.1f}")
            if hammings_b:
                mc[10].metric(f"Hamming {label_b}", f"{sum(hammings_b)/len(hammings_b):.1f}")

            sdf = pd.DataFrame(summary_rows)
            st.dataframe(
                sdf,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Gap %":      st.column_config.NumberColumn("Gap %",      format="%+.2f"),
                    col_score_a:  st.column_config.NumberColumn(col_score_a,  format="%.4f"),
                    col_score_b:  st.column_config.NumberColumn(col_score_b,  format="%.4f"),
                    col_hamming_a:st.column_config.NumberColumn(col_hamming_a,format="%.2f"),
                    col_hamming_b:st.column_config.NumberColumn(col_hamming_b,format="%.2f"),
                    col_rank_a:   st.column_config.NumberColumn(col_rank_a,   format="%d"),
                    col_rank_b:   st.column_config.NumberColumn(col_rank_b,   format="%d"),
                },
            )

    # ── Convergence moyenne par famille ──────────────────────────────────────
    if cfg_a and cfg_b:
        # (cfg, problem) -> (mean Series, std Series), indexed by actual step values
        _conv_data: dict = {}
        for _cfg in [cfg_a, cfg_b]:
            for _prob in ["NK", "NK3", "QUBO"]:
                _series = []
                for _inst in sorted_instances:
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
                for _cfg, _lbl, _color in [(cfg_a, label_a, COLORS[0]),
                                           (cfg_b, label_b, COLORS[1])]:
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
                    for curve, lbl, color in [(curve_a, label_a, COLORS[0]), (curve_b, label_b, COLORS[1])]:
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
                    for curve, lbl, color in [(curve_a, label_a, COLORS[0]), (curve_b, label_b, COLORS[1])]:
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
                    for scores, lbl, color in [(scores_a, label_a, COLORS[0]), (scores_b, label_b, COLORS[1])]:
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


# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="PPO-EDA Dashboard", layout="wide", page_icon="📊")
st.title("PPO-EDA — Grid Search Dashboard")

df = load_summary()
if df.empty:
    st.error(f"Aucun résultat trouvé dans `{RESULTS_DIR}`")
    st.stop()

# Garder uniquement les configs avec au moins un raw_scores.csv
def _has_raw(config_name: str) -> bool:
    d = RESULTS_DIR / config_name
    if not d.exists():
        return False
    return any((sub / "raw_scores.csv").exists() for sub in d.iterdir() if sub.is_dir())

df = df[df["config"].apply(_has_raw)].reset_index(drop=True)
if df.empty:
    st.warning("Aucune config avec `raw_scores.csv` trouvée.")
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
    show_n   = st.number_input("Configs max", 5, 500, 100)

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

# ── Sorted instances (computed once, passed to tabs that need it) ──────────────
all_instances: set[str] = set()
for cfg in filtered["config"]:
    d = RESULTS_DIR / cfg
    if d.exists():
        for sub in d.iterdir():
            if sub.is_dir() and INSTANCE_RE.match(sub.name):
                all_instances.add(sub.name)
sorted_instances = sorted(all_instances)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 Classement", "📈 Courbes", "📦 Boxplots", "🔍 Analyse", "⚖️ Comparaison"]
)

with tab1:
    tab_classement(filtered)

with tab2:
    tab_courbes(filtered, sorted_instances)

with tab3:
    tab_boxplots(filtered, sorted_instances)

with tab4:
    tab_analyse(filtered)

with tab5:
    tab_comparaison(df, sorted_instances)
