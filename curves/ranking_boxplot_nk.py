"""Horizontal boxplot of rank distributions across NK / NK3 configurations."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RANKING_DIR = ROOT / "additional_results" / "global_ranking"

SKIP = {"PPO-EDA", "Tabu", "TABU", "tabu"}

MY_KEY = "SVGD"
MY_LABEL = "SVGD-EDA"

COMPETITORS = ["DiscreteDE", "RecombiningDiscreteLenglerOnePlusOne", "MIMIC", "BOA"]

DISPLAY_NAMES = {
    "RecombiningDiscreteLenglerOnePlusOne": "RDL1+1",
}

PALETTE = {
    MY_LABEL:    "#2ca02c",
    "DiscreteDE": "#1f77b4",
    "RDL1+1":     "#e377c2",
    "MIMIC":      "#9467bd",
    "BOA":        "#8c564b",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _last_score(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    try:
        with csv_path.open() as f:
            lines = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
            reader = csv.reader(lines)
            header = next(reader, None)
            if not header:
                return None
            last = None
            for row in reader:
                last = row
            if last is None:
                return None
            d = dict(zip(header, last))
            for key in ("mean", "avg_score", "score", "avg_fitness", "fitness", "best_fitness"):
                v = d.get(key, "").strip()
                if v:
                    try:
                        return float(v)
                    except ValueError:
                        continue
    except Exception:
        pass
    return None


def _load_ranking(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name_algo") or row.get("name") or row.get("algo") or "").strip()
            if not name or name in SKIP:
                continue
            try:
                score = float(row["score"])
            except (KeyError, ValueError):
                continue
            rows.append((name, score))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def collect_ranks(config_dir: Path) -> dict[str, list[int]]:
    nk_files: list[tuple[str, int, int, Path]] = []
    for f in sorted(RANKING_DIR.glob("NK_N_*_K_*_ranks.csv")):
        m = re.match(r"NK_N_(\d+)_K_(\d+)_ranks\.csv", f.name)
        if m:
            nk_files.append(("NK", int(m.group(1)), int(m.group(2)), f))
    for f in sorted(RANKING_DIR.glob("NK3_N_*_K_*_ranks.csv")):
        m = re.match(r"NK3_N_(\d+)_K_(\d+)_ranks\.csv", f.name)
        if m:
            nk_files.append(("NK3", int(m.group(1)), int(m.group(2)), f))

    ranks: dict[str, list[int]] = {MY_LABEL: [], **{c: [] for c in COMPETITORS}}

    for problem, dim, t, rank_path in nk_files:
        ranking = _load_ranking(rank_path)

        svgd_score = _last_score(config_dir / f"{problem}_dim{dim}_t{t}" / "best_metrics.csv")
        combined = list(ranking)
        if svgd_score is not None:
            combined.append((MY_KEY, svgd_score))
        combined.sort(key=lambda x: x[1], reverse=True)
        rank_map = {name: idx + 1 for idx, (name, _) in enumerate(combined)}

        if MY_KEY in rank_map:
            ranks[MY_LABEL].append(rank_map[MY_KEY])
        for comp in COMPETITORS:
            if comp in rank_map:
                ranks[comp].append(rank_map[comp])

    return ranks


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(ranks: dict[str, list[int]], output_path: Path) -> None:
    comps_sorted = sorted(
        [c for c in COMPETITORS if ranks.get(c)],
        key=lambda c: sorted(ranks[c])[len(ranks[c]) // 2],
    )
    algo_order = [MY_LABEL] + comps_sorted

    def display(algo: str) -> str:
        return DISPLAY_NAMES.get(algo, algo)

    data = [ranks[a] for a in algo_order]
    colors = [PALETTE.get(display(a), "#aaaaaa") for a in algo_order]
    n = len(algo_order)

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=300)

    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        positions=list(range(n)),
        widths=0.52,
        whis=(5, 95),
        medianprops=dict(color="red", linewidth=1.0),
        whiskerprops=dict(linewidth=1.4, color="#333333"),
        capprops=dict(linewidth=1.4, color="#333333"),
        flierprops=dict(visible=False),
        showfliers=False,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    for i, d in enumerate(data):
        ax.plot(sum(d) / len(d), i, "ko", markersize=5, zorder=5, clip_on=False)

    svgd_mean = sum(ranks[MY_LABEL]) / len(ranks[MY_LABEL])
    ax.axvline(svgd_mean, color=PALETTE[MY_LABEL], linestyle="--", linewidth=1.5, alpha=0.85, zorder=3)
    ax.text(
        svgd_mean, 0 - 0.42,
        f"{svgd_mean:.1f}",
        ha="center", va="bottom",
        fontsize=8, color=PALETTE[MY_LABEL],
        clip_on=False,
    )

    ax.set_yticks(list(range(n)))
    ax.set_yticklabels([display(a) for a in algo_order], fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Rank over 24 NK/NK3 benchmark settings (↓ better)", fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
    ax.set_xlim(left=0.5)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=[
            Line2D([0], [0], color="red", linewidth=2, label="Median"),
            Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=5, label="Mean"),
        ],
        loc="upper right",
        fontsize=9,
        frameon=True,
        framealpha=0.85,
        edgecolor="#cccccc",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# Public entry point (called from main_curves_overall.py)
# ---------------------------------------------------------------------------

def generate(config_dir: Path) -> None:
    ranks = collect_ranks(config_dir)
    output_path = config_dir / "ranking_boxplot_nk_nk3.png"
    plot(ranks, output_path)


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

def main() -> None:
    DEFAULT = (
        ROOT / "results" / "config"
        / "krbf__advglobalrankweighted__M7__L13__eps0p08__g0p015__ds0p03__dm0p01"
    )
    config_dir = DEFAULT
    ranks = collect_ranks(config_dir)

    print(f"\n{'Algorithm':<40} {'n':>3}  {'min':>4}  {'Q1':>5}  {'med':>5}  {'Q3':>5}  {'max':>4}  {'wins':>5}")
    print("-" * 80)
    for algo in [MY_LABEL] + COMPETITORS:
        r = sorted(ranks.get(algo, []))
        if not r:
            print(f"{algo:<40}  no data")
            continue
        n = len(r)
        q1 = r[int(0.25 * (n - 1))]
        med = r[int(0.50 * (n - 1))]
        q3 = r[int(0.75 * (n - 1))]
        w = sum(1 for x in r if x == 1)
        print(f"{algo:<40} {n:>3}  {r[0]:>4}  {q1:>5}  {med:>5}  {q3:>5}  {r[-1]:>4}  {w:>5}")

    plot(ranks, config_dir / "ranking_boxplot_nk_nk3.png")


if __name__ == "__main__":
    main()
