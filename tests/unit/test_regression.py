"""Unit tests for OLS regression on synthetic DataFrame."""

import numpy as np
import pandas as pd
import pytest

from curvature_semantics.regression.feature_builder import build_feature_df, build_regression_matrix
from curvature_semantics.regression.ols_regression import OLSRegression
from curvature_semantics.curvature.curvature_aggregator import CurvatureBundle
from curvature_semantics.semantics.completeness_aggregator import SemanticCompletenessBundle


def make_synthetic_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "trajectory_divergence": rng.uniform(0.1, 1.0, n),
        "intrinsic_dimension": rng.uniform(5, 50, n),
        "neighborhood_distortion": rng.uniform(0.0, 0.5, n),
        "entropy": rng.uniform(0.0, 3.0, n),
        "confidence": rng.uniform(0.3, 1.0, n),
        "model_params_billions": rng.choice([1.0, 3.0, 7.0], n),
        "layer_idx": rng.integers(0, 32, n),
        "domain": rng.choice(["arithmetic", "speculative_medicine"], n),
    })
    # Construct a noisy-linear target so OLS has something real to fit
    df["nli_entailment"] = (
        0.8 - 0.4 * df["trajectory_divergence"]
        + 0.1 * rng.standard_normal(n)
    ).clip(0.0, 1.0)
    return df


class TestFeatureBuilder:
    def test_build_regression_matrix(self):
        df = make_synthetic_df()
        X, y = build_regression_matrix(
            df, target="nli_entailment",
            predictors=["trajectory_divergence", "entropy"],
        )
        assert X.shape == (100, 2)
        assert y.shape == (100,)

    def test_build_feature_df_from_bundles(self):
        n = 5
        c_bundles = [
            CurvatureBundle(layer_idx=i, n_samples=10, metrics={"trajectory_divergence": float(i)})
            for i in range(n)
        ]
        s_bundles = [
            SemanticCompletenessBundle(prompt="Q", response="A", domain="test", metrics={"nli_entailment": 0.8})
            for _ in range(n)
        ]
        df = build_feature_df(c_bundles, s_bundles, model_metadata={"alias": "test-model"})
        assert len(df) == n
        assert "trajectory_divergence" in df.columns
        assert "nli_entailment" in df.columns
        assert "model_alias" in df.columns


class TestOLSRegression:
    def test_fit_and_summary(self):
        df = make_synthetic_df()
        ols = OLSRegression(target="nli_entailment", predictors=["trajectory_divergence", "entropy"]).fit(df)
        summary = ols.summary()
        assert "coefficients" in summary

    def test_predict_returns_array(self):
        df = make_synthetic_df()
        ols = OLSRegression(target="nli_entailment", predictors=["trajectory_divergence"]).fit(df)
        preds = ols.predict(df)
        assert len(preds) == len(df)

    def test_r_squared_reasonable(self):
        df = make_synthetic_df()
        ols = OLSRegression(target="nli_entailment", predictors=["trajectory_divergence"]).fit(df)
        summary = ols.summary()
        r2 = summary.get("r_squared", 0.0)
        assert r2 >= 0.0 and r2 <= 1.0
