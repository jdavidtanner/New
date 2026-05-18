"""Line plots of curvature/ID/distortion across layers; heatmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from curvature_semantics.visualization.plot_utils import setup_style, save_figure
from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)


def plot_layerwise_curvature(
    df: pd.DataFrame,
    curvature_cols: list[str],
    output_path: Path,
    title: str = "Curvature Across Layers",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping layerwise plot")
        return

    if "layer_idx" not in df.columns:
        logger.warning("No layer_idx column; skipping layerwise plot")
        return

    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in curvature_cols:
        if col not in df.columns:
            continue
        mean = df.groupby("layer_idx")[col].mean()
        std = df.groupby("layer_idx")[col].std().fillna(0)
        ax.plot(mean.index, mean.values, label=col, linewidth=2)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Curvature proxy")
    ax.set_title(title)
    ax.legend(fontsize=9)
    save_figure(fig, output_path)
    logger.info("Layerwise geometry plot saved to %s", output_path)


def plot_curvature_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    output_path: Path,
    title: str = "Curvature Heatmap",
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed; skipping heatmap")
        return

    setup_style()
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels)), max(4, len(row_labels) // 2)))
    sns.heatmap(matrix, xticklabels=col_labels, yticklabels=row_labels, ax=ax, cmap="RdYlBu_r", center=0)
    ax.set_title(title)
    save_figure(fig, output_path)
