"""Barcode and persistence diagram plots from TDA results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from curvature_semantics.visualization.plot_utils import setup_style, save_figure
from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)


def plot_persistence_diagram(
    diagrams: list[np.ndarray],
    output_path: Path,
    title: str = "Persistence Diagram",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    setup_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    for dim, (dgm, color) in enumerate(zip(diagrams, colors)):
        if dgm.size == 0:
            continue
        finite = dgm[dgm[:, 1] < np.inf]
        ax.scatter(finite[:, 0], finite[:, 1], c=color, label=f"H{dim}", alpha=0.7, s=30)

    lim_max = max(
        max(d[d[:, 1] < np.inf][:, 1].max() for d in diagrams if d.size > 0 and d[d[:, 1] < np.inf].size > 0),
        1.0,
    )
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max * 1.05)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.legend()
    save_figure(fig, output_path)


def plot_barcode(
    diagrams: list[np.ndarray],
    output_path: Path,
    title: str = "Persistence Barcode",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    y = 0
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    for dim, (dgm, color) in enumerate(zip(diagrams, colors)):
        if dgm.size == 0:
            continue
        finite = dgm[dgm[:, 1] < np.inf]
        for (birth, death) in finite:
            ax.plot([birth, death], [y, y], c=color, linewidth=2, alpha=0.8)
            y += 1

    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Feature index")
    ax.set_title(title)
    save_figure(fig, output_path)
