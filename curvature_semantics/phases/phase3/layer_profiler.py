"""Classifies transformer layers into regions and aligns them across architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LayerProfile:
    n_layers: int
    embedding_indices: list[int]
    early_indices: list[int]
    middle_indices: list[int]
    late_indices: list[int]
    logit_index: int | None = None

    def region_of(self, layer_idx: int) -> str:
        if layer_idx in self.embedding_indices:
            return "embedding"
        if layer_idx in self.early_indices:
            return "early"
        if layer_idx in self.middle_indices:
            return "middle"
        if layer_idx in self.late_indices:
            return "late"
        return "unknown"


def build_layer_profile(
    n_layers: int,
    early_fraction: float = 0.25,
    middle_fraction: float = 0.50,
    late_fraction: float = 0.25,
    extract_logits: bool = True,
) -> LayerProfile:
    """Partition n_layers into embedding / early / middle / late regions."""
    assert abs(early_fraction + middle_fraction + late_fraction - 1.0) < 1e-6

    n_early = max(1, int(n_layers * early_fraction))
    n_middle = max(1, int(n_layers * middle_fraction))
    n_late = n_layers - n_early - n_middle

    embedding_indices = [0]
    early_indices = list(range(1, n_early + 1))
    middle_indices = list(range(n_early + 1, n_early + n_middle + 1))
    late_indices = list(range(n_early + n_middle + 1, n_layers))
    logit_index = n_layers - 1 if extract_logits else None

    return LayerProfile(
        n_layers=n_layers,
        embedding_indices=embedding_indices,
        early_indices=early_indices,
        middle_indices=middle_indices,
        late_indices=late_indices,
        logit_index=logit_index,
    )


def compute_region_statistics(
    df: "pandas.DataFrame",
    profile: LayerProfile,
    curvature_col: str = "trajectory_divergence",
    completeness_col: str = "nli_entailment",
) -> dict[str, Any]:
    """Compute mean curvature and completeness per layer region."""
    import pandas as pd

    df = df.copy()
    df["region"] = df["layer_idx"].map(profile.region_of)
    stats: dict[str, Any] = {}
    for region in ["embedding", "early", "middle", "late"]:
        sub = df[df["region"] == region]
        if sub.empty:
            continue
        stats[region] = {
            "n": len(sub),
            curvature_col + "_mean": float(sub[curvature_col].mean()) if curvature_col in sub.columns else None,
            curvature_col + "_std": float(sub[curvature_col].std()) if curvature_col in sub.columns else None,
            completeness_col + "_mean": float(sub[completeness_col].mean()) if completeness_col in sub.columns else None,
        }
    return stats
