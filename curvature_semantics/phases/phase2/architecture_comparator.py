"""Cross-architecture comparison: effect sizes and correlation tables across model families."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from curvature_semantics.utils.statistical_utils import cohens_d, eta_squared, fdr_correct
from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)

CURVATURE_COLS = ["trajectory_divergence", "intrinsic_dimension", "neighborhood_distortion"]
COMPLETENESS_COLS = ["nli_entailment", "nli_contradiction", "self_consistency"]


def compare_across_architectures(
    results_by_model: dict[str, pd.DataFrame],
    curvature_col: str = "trajectory_divergence",
    completeness_col: str = "nli_entailment",
) -> pd.DataFrame:
    """Compute per-model summary stats and cross-model effect sizes."""
    rows = []
    for alias, df in results_by_model.items():
        if curvature_col not in df.columns or completeness_col not in df.columns:
            continue
        curv = df[curvature_col].dropna().values
        comp = df[completeness_col].dropna().values
        r, p = stats.spearmanr(curv, comp) if len(curv) > 1 else (0.0, 1.0)
        rows.append({
            "model_alias": alias,
            f"{curvature_col}_mean": float(curv.mean()),
            f"{curvature_col}_std": float(curv.std()),
            f"{completeness_col}_mean": float(comp.mean()),
            f"{completeness_col}_std": float(comp.std()),
            "spearman_r": float(r),
            "spearman_p": float(p),
            "n": len(curv),
        })
    return pd.DataFrame(rows)


def compute_effect_sizes(
    results_by_model: dict[str, pd.DataFrame],
    split_col: str = "model_tuning",
    target_col: str = "trajectory_divergence",
    metric: str = "cohens_d",
) -> dict[str, float]:
    """Compare groups (e.g. base vs instruct) on curvature proxy."""
    grouped: dict[str, list[np.ndarray]] = {}
    for alias, df in results_by_model.items():
        if target_col not in df.columns:
            continue
        group = df[split_col].iloc[0] if split_col in df.columns else alias
        grouped.setdefault(str(group), []).append(df[target_col].dropna().values)

    merged = {k: np.concatenate(v) for k, v in grouped.items() if v}
    keys = list(merged.keys())
    if len(keys) < 2:
        return {}

    if metric == "cohens_d" and len(keys) == 2:
        d = cohens_d(merged[keys[0]], merged[keys[1]])
        return {f"cohens_d_{keys[0]}_vs_{keys[1]}": float(d)}
    elif metric == "eta_squared":
        groups = list(merged.values())
        return {"eta_squared": float(eta_squared(groups))}
    return {}


def replication_summary(
    results_by_model: dict[str, pd.DataFrame],
    min_replications: int = 4,
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """Summarise replication success across architectures.

    Tests two signals found in Phase 1:
      (A) trajectory_divergence → nli_entailment
      (B) intrinsic_dimension   → nli_contradiction  (strongest in Phase 1: r=−0.735 at layer 12)
    """
    tests = [
        ("trajectory_divergence", "nli_entailment"),
        ("intrinsic_dimension", "nli_contradiction"),
    ]
    results: dict[str, Any] = {"n_models_tested": len(results_by_model), "signals": {}}

    for curv_col, comp_col in tests:
        significant_models = []
        all_r: list[float] = []
        for alias, df in results_by_model.items():
            if curv_col not in df.columns or comp_col not in df.columns:
                continue
            x = df[curv_col].replace([np.inf, -np.inf], np.nan).dropna().values
            y = df[comp_col].dropna().values
            n = min(len(x), len(y))
            if n < 5:
                continue
            r, p = stats.spearmanr(x[:n], y[:n])
            if np.isfinite(r):
                all_r.append(float(r))
                if p < significance_level:
                    significant_models.append(alias)

        key = f"{curv_col}__{comp_col}"
        results["signals"][key] = {
            "n_significant": len(significant_models),
            "significant_models": significant_models,
            "replication_rate": len(significant_models) / max(len(results_by_model), 1),
            "mean_spearman_r": float(np.mean(all_r)) if all_r else 0.0,
            "effect_direction_consistent": (
                all(r < 0 for r in all_r) or all(r > 0 for r in all_r)
            ) if all_r else False,
        }

    # Primary verdict: either signal replicates in ≥ min_replications models
    n_sig_primary = results["signals"].get("trajectory_divergence__nli_entailment", {}).get("n_significant", 0)
    n_sig_secondary = results["signals"].get("intrinsic_dimension__nli_contradiction", {}).get("n_significant", 0)
    results["hypothesis_supported"] = max(n_sig_primary, n_sig_secondary) >= min_replications
    results["replication_rate"] = max(
        results["signals"][k]["replication_rate"] for k in results["signals"]
    ) if results["signals"] else 0.0
    return results


def per_layer_correlations(
    results_by_model: dict[str, pd.DataFrame],
    curvature_col: str = "intrinsic_dimension",
    completeness_col: str = "nli_contradiction",
) -> pd.DataFrame:
    """Compute curvature→completeness correlation at each relative layer depth (0.0–1.0).

    Normalises layer_idx by total layers per model so curves are comparable across
    architectures with different numbers of layers.
    """
    rows = []
    for alias, df in results_by_model.items():
        if curvature_col not in df.columns or completeness_col not in df.columns:
            continue
        if "layer_idx" not in df.columns:
            continue
        n_layers = df["layer_idx"].max() + 1
        for layer_idx, grp in df.groupby("layer_idx"):
            x = grp[curvature_col].replace([np.inf, -np.inf], np.nan).dropna().values
            y = grp[completeness_col].dropna().values
            n = min(len(x), len(y))
            if n < 3:
                continue
            r, p = stats.spearmanr(x[:n], y[:n])
            rows.append({
                "model_alias": alias,
                "layer_idx": int(layer_idx),
                "layer_rel": float(layer_idx) / max(n_layers - 1, 1),
                "spearman_r": float(r) if np.isfinite(r) else 0.0,
                "p_value": float(p) if np.isfinite(p) else 1.0,
                "n": n,
            })
    return pd.DataFrame(rows)
