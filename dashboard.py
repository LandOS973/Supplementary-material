"""
PPO-EDA Grid Search Dashboard
Run: streamlit run dashboard.py
"""
from __future__ import annotations

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


CACHE_FILE = RESULTS_DIR / ".dashboard_cache.parquet"


def _cache_is_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    cache_mtime = CACHE_FILE.stat().st_mtime
    for config_dir in RESULTS_DIR.iterdir():
        if config_dir.is_dir() and config_dir.name.startswith("k"):
            if config_dir.stat().st_mtime > cache_mtime:
                return False
    return True


def _rank_stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {"mean_rank": None, "top1": 0, "top3": 0, "top5": 0, "top10": 0}
    return {
        "mean_rank": round(float(sub["rank"].mean()), 2),
        "top1":  int((sub["rank"] == 1).sum()),
        "top3":  int((sub["rank"] <= 3).sum()),
        "top5":  int((sub["rank"] <= 5).sum()),
        "top10": int((sub["rank"] <= 10).sum()),
    }


@st.cache_data(show_spinner=False)
def load_summary() -> pd.DataFrame:
    if _cache_is_fresh():
        return pd.read_parquet(CACHE_FILE)

    rows = []
    config_dirs = [d for d in sorted(RESULTS_DIR.iterdir())
                   if d.is_dir() and d.name.startswith("k")]
    progress = st.progress(0, text="Calcul du résumé…")

    for i, config_dir in enumerate(config_dirs):
        progress.progress((i + 1) / max(len(config_dirs), 1),
                          text=f"Config {i+1}/{len(config_dirs)} — {config_dir.name[:50]}")
        params = parse_config_name(config_dir.name)
        if not params:
            continue

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
            except Exception:
                continue
            rank = _get_rank(m.group("problem"), int(m.group("dim")), int(m.group("t")), score)
            if rank is not None:
                instance_rows.append({
                    "problem": m.group("problem"),
                    "rank": rank,
                    "score": score,
                })

        if not instance_rows:
            continue
        idf = pd.DataFrame(instance_rows)

        all_s  = _rank_stats(idf)
        nk_s   = _rank_stats(idf[idf["problem"] == "NK"])
        nk3_s  = _rank_stats(idf[idf["problem"] == "NK3"])
        qubo_s = _rank_stats(idf[idf["problem"] == "QUBO"])

        rows.append({
            "config": config_dir.name,
            **params,
            "mean_rank":    all_s["mean_rank"],
            "median_rank":  round(float(idf["rank"].median()), 1),
            "top1":  all_s["top1"],   "top3":  all_s["top3"],
            "top5":  all_s["top5"],   "top10": all_s["top10"],
            "top1_NK":   nk_s["top1"],
            "top1_NK3":  nk3_s["top1"],
            "top1_QUBO": qubo_s["top1"],
            "mean_rank_NK":   nk_s["mean_rank"],
            "mean_rank_NK3":  nk3_s["mean_rank"],
            "mean_rank_QUBO": qubo_s["mean_rank"],
            "top3_NK":   nk_s["top3"],
            "top3_NK3":  nk3_s["top3"],
            "top3_QUBO": qubo_s["top3"],
            "n_instances": len(idf),
        })

    progress.empty()
    result = pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True) if rows else pd.DataFrame()
    if not result.empty:
        result.to_parquet(CACHE_FILE)
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
    from plotly.subplots import make_subplots

    problem_sel = st.radio(
        "Type de problème",
        ["Tous", "NK", "NK3", "QUBO"],
        horizontal=True,
        key="analyse_problem",
    )
    rank_col = {
        "Tous": "mean_rank",
        "NK":   "mean_rank_NK",
        "NK3":  "mean_rank_NK3",
        "QUBO": "mean_rank_QUBO",
    }[problem_sel]
    top1_col = {
        "Tous": "top1",
        "NK":   "top1_NK",
        "NK3":  "top1_NK3",
        "QUBO": "top1_QUBO",
    }[problem_sel]

    analyse_df = filtered.dropna(subset=[rank_col])
    available = [(lbl, col) for lbl, col in CURVE_PARAMS
                 if col in analyse_df.columns and analyse_df[col].nunique() > 1]

    if not available:
        st.info("Pas assez de variabilité dans les configs filtrées pour une analyse.")
        return

    n_cols = 2
    n_rows = (len(available) + 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[lbl for lbl, _ in available],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    for i, (lbl, col) in enumerate(available):
        r = i // n_cols + 1
        c = i % n_cols + 1
        grp = (
            analyse_df.groupby(col)
            .agg(rank_moy=(rank_col, "mean"))
            .round(2)
            .reset_index()
            .sort_values("rank_moy")
        )
        fig.add_trace(
            go.Bar(
                x=grp[col].astype(str),
                y=grp["rank_moy"],
                text=grp["rank_moy"].round(1),
                textposition="outside",
                marker_color=COLORS[:len(grp)],
                showlegend=False,
            ),
            row=r, col=c,
        )

    fig.update_layout(height=320 * n_rows, margin=dict(t=40, b=20))
    fig.update_yaxes(title_text="Rank moy")
    st.plotly_chart(fig, use_container_width=True, key="analyse_fig")

    st.divider()
    st.markdown("##### Détail par paramètre")
    detail_cols = st.columns(n_cols)
    for i, (lbl, col) in enumerate(available):
        with detail_cols[i % n_cols]:
            grp = (
                analyse_df.groupby(col)
                .agg(
                    n=("config", "count"),
                    rank_moy=(rank_col, "mean"),
                    top1=(top1_col, "mean"),
                )
                .round(2)
                .reset_index()
                .rename(columns={col: lbl})
                .sort_values("rank_moy")
            )
            st.markdown(f"**Par {lbl}**")
            st.dataframe(grp, use_container_width=True, hide_index=True)


# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="PPO-EDA Dashboard", layout="wide", page_icon="📊")
st.title("PPO-EDA — Grid Search Dashboard")

df = load_summary()
if df.empty:
    st.error(f"Aucun résultat trouvé dans `{RESULTS_DIR}`")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filtres")
    if st.button("🔄 Recalculer le cache", help="À utiliser après un rsync de nouveaux résultats"):
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        st.cache_data.clear()
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
        mask &= df[col].isin(sel)
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
tab1, tab2, tab3, tab4 = st.tabs(["📋 Classement", "📈 Courbes", "📦 Boxplots", "🔍 Analyse"])

with tab1:
    tab_classement(filtered)

with tab2:
    tab_courbes(filtered, sorted_instances)

with tab3:
    tab_boxplots(filtered, sorted_instances)

with tab4:
    tab_analyse(filtered)
