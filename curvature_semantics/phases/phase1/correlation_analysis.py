"""Spearman/Pearson/partial correlations between curvature proxies and completeness metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from curvature_semantics.utils.statistical_utils import partial_correlation, fdr_correct
from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)

CURVATURE_COLS = [
    "trajectory_divergence", "intrinsic_dimension", "neighborhood_distortion",
    "geodesic_deviation", "ricci_graph_curvature",
]
COMPLETENESS_COLS = [
    "nli_entailment", "nli_contradiction", "factuality", "self_consistency",
    "calibration_error", "ood_uncertainty_entropy",
]
CONTROL_COLS = ["entropy", "confidence", "model_params_billions"]


def compute_correlation_matrix(
    df: pd.DataFrame,
    curvature_cols: list[str] | None = None,
    completeness_cols: list[str] | None = None,
    method: str = "spearman",
) -> pd.DataFrame:
    """Return correlation matrix: curvature_cols x completeness_cols."""
    c_cols = [c for c in (curvature_cols or CURVATURE_COLS) if c in df.columns]
    s_cols = [c for c in (completeness_cols or COMPLETENESS_COLS) if c in df.columns]
    rows = []
    for c in c_cols:
        row = {"curvature_proxy": c}
        for s in s_cols:
            x = df[c].fillna(0).values
            y = df[s].fillna(0).values
            if method == "spearman":
                r, p = stats.spearmanr(x, y)
            else:
                r, p = stats.pearsonr(x, y)
            row[s] = float(r) if np.isfinite(r) else 0.0
            row[f"{s}_p"] = float(p) if np.isfinite(p) else 1.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("curvature_proxy")


def compute_partial_correlations(
    df: pd.DataFrame,
    curvature_cols: list[str] | None = None,
    completeness_cols: list[str] | None = None,
    control_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Partial correlations controlling for entropy, confidence, model size."""
    c_cols = [c for c in (curvature_cols or CURVATURE_COLS) if c in df.columns]
    s_cols = [c for c in (completeness_cols or COMPLETENESS_COLS) if c in df.columns]
    controls_available = [c for c in (control_cols or CONTROL_COLS) if c in df.columns]
    controls = df[controls_available].fillna(0).values if controls_available else None

    rows = []
    p_values = []
    for c in c_cols:
        row = {"curvature_proxy": c}
        for s in s_cols:
            x = df[c].fillna(0).values
            y = df[s].fillna(0).values
            if controls is not None and len(controls) > 0:
                r, p = partial_correlation(x, y, controls)
            else:
                r, p = stats.pearsonr(x, y)
            row[s] = float(r) if np.isfinite(r) else 0.0
            row[f"{s}_p"] = float(p) if np.isfinite(p) else 1.0
            p_values.append(float(p) if np.isfinite(p) else 1.0)
        rows.append(row)

    result = pd.DataFrame(rows).set_index("curvature_proxy")
    _, adj_p = fdr_correct(np.array(p_values))
    # Attach FDR-corrected p-values
    for i, (c, s) in enumerate([(c, s) for c in c_cols for s in s_cols]):
        if f"{s}_p_fdr" not in result.columns:
            result[f"{s}_p_fdr"] = np.nan
        if c in result.index:
            result.loc[c, f"{s}_p_fdr"] = adj_p[i]
    return result


def summarise_findings(corr_df: pd.DataFrame, partial_df: pd.DataFrame) -> dict:
    """Return key findings from correlation analysis."""
    findings = {}
    for col in COMPLETENESS_COLS:
        if col not in corr_df.columns:
            continue
        series = corr_df[col].abs().dropna()
        if series.empty:
            findings[f"top_curvature_predictor_of_{col}"] = {"predictor": None, "r_spearman": None}
            continue
        top_predictor = series.idxmax()
        findings[f"top_curvature_predictor_of_{col}"] = {
            "predictor": top_predictor,
            "r_spearman": float(corr_df.loc[top_predictor, col]),
        }
    return findings
