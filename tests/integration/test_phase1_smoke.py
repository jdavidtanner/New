"""Smoke test: run Phase 1 on tiny synthetic data (no real model required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from curvature_semantics.curvature.curvature_aggregator import CurvatureAggregator
from curvature_semantics.semantics.completeness_aggregator import CompletenessAggregator
from curvature_semantics.phases.phase1.correlation_analysis import (
    compute_correlation_matrix, compute_partial_correlations,
)
from curvature_semantics.regression.feature_builder import build_feature_df
from curvature_semantics.curvature.curvature_aggregator import CurvatureBundle
from curvature_semantics.semantics.completeness_aggregator import SemanticCompletenessBundle


@pytest.fixture
def synthetic_feature_df():
    rng = np.random.default_rng(0)
    n = 50
    return pd.DataFrame({
        "trajectory_divergence": rng.uniform(0.1, 1.0, n),
        "intrinsic_dimension": rng.uniform(5, 40, n),
        "neighborhood_distortion": rng.uniform(0.0, 0.5, n),
        "entropy": rng.uniform(0.5, 3.0, n),
        "confidence": rng.uniform(0.4, 1.0, n),
        "model_params_billions": [1.0] * n,
        "layer_idx": list(range(n)),
        "domain": ["arithmetic"] * 25 + ["speculative_medicine"] * 25,
        "nli_entailment": rng.uniform(0.2, 0.9, n),
        "nli_contradiction": rng.uniform(0.0, 0.3, n),
        "self_consistency": rng.uniform(0.5, 1.0, n),
    })


def test_correlation_matrix_runs(synthetic_feature_df):
    df = synthetic_feature_df
    result = compute_correlation_matrix(df, method="spearman")
    assert len(result) > 0
    assert "nli_entailment" in result.columns


def test_partial_correlations_run(synthetic_feature_df):
    df = synthetic_feature_df
    result = compute_partial_correlations(df)
    assert len(result) > 0


def test_curvature_aggregator_on_batch():
    rng = np.random.default_rng(1)
    agg = CurvatureAggregator(proxy_names=["trajectory_divergence", "intrinsic_dimension"])
    hs = rng.standard_normal((20, 64)).astype(np.float32)
    bundle = agg.compute_layer(hs, layer_idx=5)
    assert bundle.layer_idx == 5
    assert "trajectory_divergence" in bundle.metrics


def test_completeness_aggregator_on_example():
    agg = CompletenessAggregator(metric_names=["nli_entailment", "evidence_utilization"])
    bundle = agg.score(
        prompt="What is photosynthesis?",
        response="Photosynthesis is the process by which plants convert sunlight into energy.",
        context="Plants use sunlight to produce glucose through photosynthesis.",
        reference="Photosynthesis converts sunlight to energy.",
        domain="basic_physics",
    )
    assert "nli_entailment" in bundle.metrics
    assert "evidence_utilization" in bundle.metrics
    assert 0.0 <= bundle.primary() <= 1.0


def test_feature_df_assembly():
    rng = np.random.default_rng(2)
    n = 10
    c_bundles = [CurvatureBundle(i, 5, {"trajectory_divergence": float(rng.random())}) for i in range(n)]
    s_bundles = [SemanticCompletenessBundle("Q", "A", "test", {"nli_entailment": float(rng.random())}) for _ in range(n)]
    df = build_feature_df(c_bundles, s_bundles)
    assert len(df) == n
    assert "trajectory_divergence" in df.columns
    assert "nli_entailment" in df.columns
