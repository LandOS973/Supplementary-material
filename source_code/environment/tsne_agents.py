"""t-SNE visualization for MultiAgentUnivariateEDA agents using Hamming distances.

Usage example (inside your training/analysis code):

    from environment.tsne_agents import plot_agents_tsne

    # model is an instance of MultiAgentUnivariateEDA after training/update
    fig, ax, embedding, distances = plot_agents_tsne(model, output_path="agents_tsne.png")
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from environment.metrics import MetricsCalculator
from eda_strategies.MultiAgentUnivariate.MultiAgentUnivariateEDA import MultiAgentUnivariateEDA

try:
    from sklearn.manifold import TSNE
except Exception as exc:  # pragma: no cover - import-time guard
    raise SystemExit(
        "scikit-learn is required for t-SNE. Install it with `pip install scikit-learn`."
    ) from exc


def _compute_hamming_distance_matrix(
    model: MultiAgentUnivariateEDA,
    metrics: Optional[MetricsCalculator] = None,
) -> Tuple[np.ndarray, float]:
    if metrics is None:
        metrics = MetricsCalculator()

    if hasattr(model, "_refresh_agent_views"):
        model._refresh_agent_views()

    avg, distances = metrics.compute_average_hamming(model.agents)
    if distances is None:
        raise ValueError("Need at least two agents to compute Hamming distances.")

    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError(f"Expected a square distance matrix, got {distances.shape}.")

    # t-SNE expects a zero diagonal for precomputed distances.
    np.fill_diagonal(distances, 0.0)
    return distances, float(avg)


def _tsne_from_distance(
    distances: np.ndarray,
    perplexity: Optional[float] = None,
    random_state: int = 0,
) -> np.ndarray:
    num_agents = distances.shape[0]
    if num_agents < 2:
        raise ValueError("t-SNE requires at least two agents.")

    if perplexity is None:
        perplexity = min(30, num_agents - 1)
    if perplexity <= 0 or perplexity >= num_agents:
        raise ValueError(
            f"perplexity must be in [1, {num_agents - 1}], got {perplexity}."
        )

    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        perplexity=perplexity,
        random_state=random_state,
    )
    return tsne.fit_transform(distances)


def plot_agents_tsne(
    model: MultiAgentUnivariateEDA,
    output_path: Optional[str] = None,
    perplexity: Optional[float] = None,
    random_state: int = 0,
    title: Optional[str] = None,
):
    distances, avg = _compute_hamming_distance_matrix(model)
    embedding = _tsne_from_distance(distances, perplexity=perplexity, random_state=random_state)

    num_agents = embedding.shape[0]
    colors = plt.cm.get_cmap("tab20", max(num_agents, 1))(np.arange(num_agents))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    for idx in range(num_agents):
        x, y = embedding[idx, 0], embedding[idx, 1]
        ax.scatter(
            x,
            y,
            s=70,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            str(idx),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    ax.set_title(title or f"t-SNE agents (Hamming, avg={avg:.3f})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

    return fig, ax, embedding, distances
