"""Generates coefficient tables, R², partial correlations, and LaTeX output."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from curvature_semantics.core.artifact_store import ArtifactStore
from curvature_semantics.core.logging_utils import get_logger
from curvature_semantics.regression.feature_builder import build_regression_matrix
from curvature_semantics.regression.ols_regression import OLSRegression
from curvature_semantics.regression.mixed_effects import MixedEffectsRegression
from curvature_semantics.utils.statistical_utils import partial_correlation, fdr_correct

logger = get_logger(__name__)


def generate_report(df: pd.DataFrame, output_dir: Path, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or {}
    target = cfg.get("target", "nli_entailment")
    predictors = cfg.get("predictors", [
        "trajectory_divergence", "intrinsic_dimension", "neighborhood_distortion",
        "entropy", "confidence", "model_params_billions",
    ])

    report: dict[str, Any] = {}

    # OLS
    ols = OLSRegression(target=target, predictors=predictors).fit(df)
    report["ols"] = ols.summary()

    # Mixed effects
    lmm = MixedEffectsRegression(target=target, fixed=predictors).fit(df)
    report["mixed_effects"] = lmm.summary()

    # Partial correlations
    X, y = build_regression_matrix(df, target=target, predictors=predictors)
    controls = X.values
    partial_corrs = {}
    p_values = []
    predictor_names = list(X.columns)
    for col in predictor_names:
        col_idx = list(X.columns).index(col)
        other_idx = [i for i in range(len(predictor_names)) if i != col_idx]
        if other_idx:
            r, p = partial_correlation(X.iloc[:, col_idx].values, y.values, controls[:, other_idx])
        else:
            from scipy import stats
            r, p = stats.pearsonr(X.iloc[:, col_idx].values, y.values)
        partial_corrs[col] = {"r": float(r), "p": float(p)}
        p_values.append(float(p))

    # FDR correction
    _, adj_p = fdr_correct(np.array(p_values))
    for i, col in enumerate(predictor_names):
        partial_corrs[col]["p_fdr"] = float(adj_p[i])

    report["partial_correlations"] = partial_corrs

    # LaTeX table
    latex = _make_latex_table(partial_corrs, ols.summary())
    (output_dir / "regression_table.tex").write_text(latex)
    logger.info("Regression report written to %s", output_dir)
    return report


def _make_latex_table(partial_corrs: dict, ols_summary: dict) -> str:
    coefs = ols_summary.get("coefficients", {})
    pvals = ols_summary.get("p_values", {})
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Regression: Semantic Completeness $\sim$ Curvature Proxies}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Predictor & $\beta$ (OLS) & $r_{\text{partial}}$ & $p_{\text{FDR}}$ \\",
        r"\hline",
    ]
    for pred, pc in partial_corrs.items():
        beta = coefs.get(pred, float("nan"))
        r = pc["r"]
        p_fdr = pc.get("p_fdr", float("nan"))
        sig = "**" if p_fdr < 0.01 else ("*" if p_fdr < 0.05 else "")
        lines.append(f"  {pred.replace('_', r'\_')} & {beta:.3f} & {r:.3f} & {p_fdr:.3f}{sig} \\\\")
    r2 = ols_summary.get("r_squared", float("nan"))
    lines += [r"\hline", f"$R^2$ & {r2:.3f} & & \\\\", r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate regression report")
    parser.add_argument("--data", required=True, help="Path to parquet feature DataFrame")
    parser.add_argument("--output-dir", default="results/regression")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = generate_report(df, output_dir)
    import json
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    logger.info("Done.")


if __name__ == "__main__":
    main()
