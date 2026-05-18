"""2D curvature atlas: UMAP of hidden states coloured by curvature proxy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from curvature_semantics.visualization.plot_utils import setup_style, save_figure
from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)


def plot_curvature_atlas(
    hidden_states: np.ndarray,
    curvature_values: np.ndarray,
    output_path: Path,
    title: str = "Curvature Atlas",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    """UMAP projection of hidden states coloured by curvature proxy."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib not installed; skipping curvature atlas")
        return

    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        embedding = reducer.fit_transform(hidden_states)
    except ImportError:
        logger.warning("umap-learn not installed; using PCA fallback")
        from sklearn.decomposition import PCA
        embedding = PCA(n_components=2, random_state=42).fit_transform(hidden_states)

    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=curvature_values, cmap="RdYlBu_r", alpha=0.7, s=20,
    )
    plt.colorbar(sc, ax=ax, label="Curvature proxy")
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    save_figure(fig, output_path)
    logger.info("Curvature atlas saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate curvature atlas")
    parser.add_argument("--hidden-states", required=True, help="Path to .npy hidden states (n, d)")
    parser.add_argument("--curvature", required=True, help="Path to .npy curvature values (n,)")
    parser.add_argument("--output", default="results/figures/curvature_atlas")
    args = parser.parse_args()

    hs = np.load(args.hidden_states)
    curv = np.load(args.curvature)
    plot_curvature_atlas(hs, curv, Path(args.output))


if __name__ == "__main__":
    main()
