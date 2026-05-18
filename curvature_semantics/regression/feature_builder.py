"""Joins CurvatureBundle + SemanticCompletenessBundle + metadata into flat DataFrame."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from curvature_semantics.curvature.curvature_aggregator import CurvatureBundle
from curvature_semantics.semantics.completeness_aggregator import SemanticCompletenessBundle


def build_feature_df(
    curvature_bundles: list[CurvatureBundle],
    completeness_bundles: list[SemanticCompletenessBundle],
    model_metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a flat DataFrame from lists of bundles.

    Each row is one (layer, example) pair.
    """
    assert len(curvature_bundles) == len(completeness_bundles), "Bundle length mismatch"
    rows = []
    for cb, sb in zip(curvature_bundles, completeness_bundles):
        row: dict[str, Any] = {}
        row.update(cb.to_flat_dict())
        row.update(sb.to_flat_dict())
        if model_metadata:
            row.update({f"model_{k}": v for k, v in model_metadata.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def add_entropy_and_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Derive entropy and confidence columns if logit columns are present."""
    if "max_prob" in df.columns:
        df = df.copy()
        df["confidence"] = df["max_prob"]
    if "entropy" not in df.columns:
        df["entropy"] = np.nan
    return df


def build_regression_matrix(
    df: pd.DataFrame,
    target: str = "nli_entailment",
    predictors: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for regression."""
    predictors = predictors or [
        "trajectory_divergence",
        "intrinsic_dimension",
        "neighborhood_distortion",
        "entropy",
        "confidence",
        "model_params_billions",
    ]
    available = [p for p in predictors if p in df.columns]
    X = df[available].fillna(df[available].mean())
    y = df[target].fillna(0.0) if target in df.columns else pd.Series(np.zeros(len(df)))
    return X, y
