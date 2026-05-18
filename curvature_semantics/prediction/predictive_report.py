"""Held-out predictive tests for the core curvature-semantics hypothesis.

The project goal is not just to observe curvature/completeness correlations; it is to test
whether curvature features predict semantic failure on held-out examples and add signal beyond
standard uncertainty/control baselines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_BASELINE_FEATURES = [
    "entropy",
    "confidence",
    "max_prob",
    "model_params_billions",
    "layer_idx",
]
DEFAULT_CURVATURE_FEATURES = [
    "trajectory_divergence",
    "intrinsic_dimension",
    "neighborhood_distortion",
    "geodesic_deviation",
    "ricci_graph_curvature",
    "persistent_homology_h0_count",
    "persistent_homology_h1_count",
]


def build_failure_target(
    df: pd.DataFrame,
    target_col: str = "nli_entailment",
    threshold: float = 0.5,
) -> np.ndarray:
    """Return binary semantic-failure labels where lower target values are worse."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in feature DataFrame")
    return (df[target_col].fillna(0.0).to_numpy(dtype=float) < threshold).astype(float)


def generate_predictive_report(
    df: pd.DataFrame,
    target_col: str = "nli_entailment",
    failure_threshold: float = 0.5,
    baseline_features: list[str] | None = None,
    curvature_features: list[str] | None = None,
    split_col: str = "prompt",
    test_fraction: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare baseline-only vs baseline+curvature held-out prediction.

    Returns a JSON-serializable report with regression metrics for the continuous target and
    classification metrics for low-completeness/failure labels. Splitting is group-aware when
    ``split_col`` is available, preventing layer/perturbation rows from the same prompt from
    leaking across train and test sets.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in feature DataFrame")
    if len(df) < 4:
        raise ValueError("At least four rows are required for a held-out predictive report")

    baseline_features = baseline_features or DEFAULT_BASELINE_FEATURES
    curvature_features = curvature_features or DEFAULT_CURVATURE_FEATURES
    available_baseline = [col for col in baseline_features if col in df.columns]
    available_curvature = [col for col in curvature_features if col in df.columns]

    train_mask, test_mask = _make_split(
        df,
        split_col=split_col,
        test_fraction=test_fraction,
        seed=seed,
    )
    y = df[target_col].fillna(0.0).to_numpy(dtype=float)
    failure = build_failure_target(df, target_col=target_col, threshold=failure_threshold)

    baseline = _fit_and_evaluate(df, y, failure, train_mask, test_mask, available_baseline)
    curvature = _fit_and_evaluate(
        df,
        y,
        failure,
        train_mask,
        test_mask,
        available_baseline + available_curvature,
    )

    report = {
        "goal": (
            "Test whether curvature-derived features predict semantic failure on held-out "
            "examples and add signal beyond baseline uncertainty/control features."
        ),
        "target_col": target_col,
        "failure_threshold": failure_threshold,
        "split_col": split_col if split_col in df.columns else None,
        "n_rows": int(len(df)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "features": {
            "baseline": available_baseline,
            "curvature": available_curvature,
        },
        "baseline_model": baseline,
        "curvature_model": curvature,
        "lift": {
            "r2": curvature["regression"]["r2"] - baseline["regression"]["r2"],
            "mae": baseline["regression"]["mae"] - curvature["regression"]["mae"],
            "auroc": curvature["classification"]["auroc"] - baseline["classification"]["auroc"],
            "accuracy": (
                curvature["classification"]["accuracy"]
                - baseline["classification"]["accuracy"]
            ),
        },
        "interpretation": _interpret_lift(baseline, curvature),
    }
    return report


def save_predictive_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    """Write a predictive report to ``predictive_report.json``."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "predictive_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Predictive report written to %s", report_path)
    return report_path


def _make_split(
    df: pd.DataFrame,
    split_col: str,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    test_fraction = min(max(test_fraction, 0.05), 0.95)
    if split_col in df.columns and df[split_col].nunique() > 1:
        groups = np.array(sorted(df[split_col].dropna().unique()))
        rng.shuffle(groups)
        n_test_groups = max(1, int(round(len(groups) * test_fraction)))
        test_groups = set(groups[:n_test_groups].tolist())
        test_mask = df[split_col].isin(test_groups).to_numpy()
    else:
        indices = np.arange(len(df))
        rng.shuffle(indices)
        n_test = max(1, int(round(len(df) * test_fraction)))
        test_indices = set(indices[:n_test].tolist())
        test_mask = np.array([idx in test_indices for idx in range(len(df))])
    train_mask = ~test_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Train/test split produced an empty partition")
    return train_mask, test_mask


def _fit_and_evaluate(
    df: pd.DataFrame,
    y: np.ndarray,
    failure: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    features: list[str],
) -> dict[str, Any]:
    x_all = _feature_matrix(df, features)
    x_train, x_test = x_all[train_mask], x_all[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    failure_test = failure[test_mask]

    x_train_std, x_test_std = _standardize(x_train, x_test)
    coefficients = _fit_ridge(x_train_std, y_train)
    predictions = _predict_with_intercept(x_test_std, coefficients)
    failure_scores = -predictions
    failure_pred = (predictions < 0.5).astype(float)

    return {
        "features": features,
        "coefficients": {
            "intercept": float(coefficients[0]),
            **{name: float(value) for name, value in zip(features, coefficients[1:])},
        },
        "regression": {
            "r2": _r2_score(y_test, predictions),
            "mae": float(np.abs(y_test - predictions).mean()),
        },
        "classification": {
            "accuracy": float((failure_pred == failure_test).mean()),
            "auroc": _auroc(failure_test, failure_scores),
            "positive_rate": float(failure_test.mean()),
        },
    }


def _feature_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    if not features:
        return np.zeros((len(df), 0), dtype=float)
    frame = df[features].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.fillna(frame.mean()).fillna(0.0).to_numpy(dtype=float)


def _standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x_train.shape[1] == 0:
        return x_train, x_test
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + penalty) @ design.T @ y


def _predict_with_intercept(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    return design @ coefficients


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.square(y_true - y_pred).sum())
    ss_tot = float(np.square(y_true - y_true.mean()).sum())
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    wins = 0.0
    for pos_score in positives:
        wins += float((pos_score > negatives).sum())
        wins += 0.5 * float((pos_score == negatives).sum())
    return float(wins / (len(positives) * len(negatives)))


def _interpret_lift(baseline: dict[str, Any], curvature: dict[str, Any]) -> str:
    auroc_lift = curvature["classification"]["auroc"] - baseline["classification"]["auroc"]
    r2_lift = curvature["regression"]["r2"] - baseline["regression"]["r2"]
    if auroc_lift > 0.02 or r2_lift > 0.02:
        return "curvature_features_improved_held_out_prediction"
    if auroc_lift < -0.02 and r2_lift < -0.02:
        return "curvature_features_hurt_held_out_prediction"
    return "no_clear_incremental_curvature_signal"


def _read_dataframe(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)
    if data_path.suffix == ".pkl":
        return pd.read_pickle(data_path)
    return pd.read_parquet(data_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether curvature predicts semantic failure"
    )
    parser.add_argument("--data", required=True, help="Path to Phase 1 feature parquet/pickle")
    parser.add_argument("--output-dir", default="results/prediction")
    parser.add_argument("--target-col", default="nli_entailment")
    parser.add_argument("--failure-threshold", type=float, default=0.5)
    parser.add_argument("--split-col", default="prompt")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = _read_dataframe(args.data)
    report = generate_predictive_report(
        df,
        target_col=args.target_col,
        failure_threshold=args.failure_threshold,
        split_col=args.split_col,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    save_predictive_report(report, args.output_dir)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
