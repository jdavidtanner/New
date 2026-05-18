"""Unified evaluation harness for all Phase 6 systems."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from curvature_semantics.semantics.nli_scorer import NLIScorer
from curvature_semantics.semantics.abstention_accuracy import AbstentionAccuracy
from curvature_semantics.semantics.calibration import CalibrationError
from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_system(
    results: list[dict[str, Any]],
    gold_answers: list[str],
    should_abstain_labels: list[bool] | None = None,
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
) -> dict[str, float]:
    """Compute all evaluation metrics for a system's outputs."""
    assert len(results) == len(gold_answers), "Results/gold length mismatch"
    n = len(results)

    nli = NLIScorer(nli_model)
    abstainer = AbstentionAccuracy()

    responses = [r.get("response", "") for r in results]
    actions = [r.get("action", "answer") for r in results]

    # NLI-based accuracy against gold
    pairs = list(zip(gold_answers, responses))
    nli_scores = nli.batch_score(pairs)
    entailment_scores = np.array([s["nli_entailment"] for s in nli_scores])
    contradiction_scores = np.array([s["nli_contradiction"] for s in nli_scores])

    # Abstention metrics
    abstained = [a == "abstain" for a in actions]
    if should_abstain_labels is not None:
        abs_metrics = abstainer.compute_corpus_metrics(
            [{"response": r} for r in responses], should_abstain_labels
        )
    else:
        abs_metrics = {"abstention_rate": float(np.mean(abstained))}

    # Coverage: fraction of questions where a substantive answer was given
    coverage = float(1.0 - np.mean(abstained))

    # Token F1 as accuracy proxy
    from curvature_semantics.utils.nlp_utils import token_overlap_f1
    token_f1s = [token_overlap_f1(g, r) for g, r in zip(gold_answers, responses)]
    accuracy = float(np.mean(token_f1s))

    metrics = {
        "accuracy": accuracy,
        "nli_entailment_mean": float(entailment_scores.mean()),
        "nli_contradiction_mean": float(contradiction_scores.mean()),
        "coverage": coverage,
        **abs_metrics,
    }
    return metrics


def compare_systems(
    system_results: dict[str, list[dict[str, Any]]],
    gold_answers: list[str],
    should_abstain_labels: list[bool] | None = None,
) -> pd.DataFrame:
    """Compare multiple systems on unified metrics."""
    rows = []
    for system_name, results in system_results.items():
        logger.info("Evaluating system: %s", system_name)
        metrics = evaluate_system(results, gold_answers, should_abstain_labels)
        rows.append({"system": system_name, **metrics})
    return pd.DataFrame(rows).set_index("system")
