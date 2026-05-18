"""Unit tests for held-out curvature prediction reports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from curvature_semantics.prediction.predictive_report import (
    build_failure_target,
    generate_predictive_report,
)


def make_predictive_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    curvature = np.linspace(0.0, 1.0, n)
    entropy = rng.uniform(0.0, 0.2, n)
    entailment = np.clip(0.9 - 0.7 * curvature + rng.normal(0.0, 0.03, n), 0.0, 1.0)
    return pd.DataFrame(
        {
            "prompt": [f"prompt-{i // 4}" for i in range(n)],
            "trajectory_divergence": curvature,
            "intrinsic_dimension": 1.0 + curvature,
            "entropy": entropy,
            "confidence": 1.0 - entropy,
            "layer_idx": np.arange(n) % 4,
            "nli_entailment": entailment,
        }
    )


def test_build_failure_target_thresholds_low_scores():
    df = pd.DataFrame({"nli_entailment": [0.2, 0.5, 0.8]})
    labels = build_failure_target(df, threshold=0.5)
    assert labels.tolist() == [1.0, 0.0, 0.0]


def test_generate_predictive_report_compares_baseline_and_curvature():
    df = make_predictive_df()
    report = generate_predictive_report(
        df,
        target_col="nli_entailment",
        failure_threshold=0.5,
        split_col="prompt",
        test_fraction=0.25,
        seed=1,
    )
    assert report["goal"].startswith("Test whether curvature-derived features")
    assert report["n_train"] > 0
    assert report["n_test"] > 0
    assert "trajectory_divergence" in report["features"]["curvature"]
    assert "baseline_model" in report
    assert "curvature_model" in report
    assert "auroc" in report["lift"]
