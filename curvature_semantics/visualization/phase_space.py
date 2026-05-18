"""Scatter: curvature vs completeness; phase boundary estimation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from curvature_semantics.visualization.plot_utils import setup_style, save_figure
from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)


def plot_phase_space(
    df: pd.DataFrame,
    curvature_col: str = "trajectory_divergence",
    completeness_col: str = "nli_entailment",
    domain_col: str = "domain",
    output_path: Path = Path("results/figures/phase_space"),
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping phase space plot")
        return

    if curvature_col not in df.columns or completeness_col not in df.columns:
        logger.warning("Required columns not found for phase space plot")
        return

    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    domains = df[domain_col].unique() if domain_col in df.columns else ["all"]
    cmap = plt.cm.get_cmap("tab10", len(domains))

    for i, domain in enumerate(domains):
        mask = df[domain_col] == domain if domain_col in df.columns else pd.Series([True] * len(df))
        sub = df[mask]
        ax.scatter(sub[curvature_col], sub[completeness_col], label=domain, alpha=0.6, s=20, color=cmap(i))

    # Fit and plot a regression line
    x = df[curvature_col].fillna(0).values
    y = df[completeness_col].fillna(0).values
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, np.polyval(z, xline), "k--", linewidth=1.5, label="trend")

    ax.set_xlabel(curvature_col.replace("_", " ").title())
    ax.set_ylabel(completeness_col.replace("_", " ").title())
    ax.set_title("Phase Space: Curvature vs Semantic Completeness")
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, output_path)
    logger.info("Phase space plot saved to %s", output_path)


def plot_intervention_delta(
    before: pd.DataFrame,
    after: pd.DataFrame,
    curvature_col: str = "trajectory_divergence",
    completeness_col: str = "nli_entailment",
    output_path: Path = Path("results/figures/intervention_delta"),
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, label in zip(axes, [curvature_col, completeness_col], ["Curvature", "Completeness"]):
        b = before[col].fillna(0).values if col in before.columns else np.zeros(len(before))
        a = after[col].fillna(0).values if col in after.columns else np.zeros(len(after))
        ax.bar(["Before", "After"], [b.mean(), a.mean()], yerr=[b.std(), a.std()], capsize=5)
        ax.set_title(f"{label} Before/After Intervention")
        ax.set_ylabel(label)
    save_figure(fig, output_path)
