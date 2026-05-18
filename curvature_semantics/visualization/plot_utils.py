"""Shared style config, color palettes, figure export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    _MPL = True
except ImportError:
    _MPL = False

PALETTE = {
    "low_curvature": "#2196F3",
    "medium_curvature": "#FF9800",
    "high_curvature": "#F44336",
    "neutral": "#9E9E9E",
    "entailment": "#4CAF50",
    "contradiction": "#F44336",
}


def setup_style() -> None:
    if not _MPL:
        return
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })


def save_figure(fig: Any, path: Path | str, formats: list[str] | None = None) -> None:
    if not _MPL:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    formats = formats or ["pdf", "png"]
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)
