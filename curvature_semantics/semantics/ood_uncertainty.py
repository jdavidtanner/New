"""OOD uncertainty: entropy of output distribution and Mahalanobis distance."""

from __future__ import annotations

from typing import Any

import numpy as np

from curvature_semantics.semantics.base import SemanticMetric


class OODUncertainty(SemanticMetric):
    """Estimates out-of-distribution uncertainty from logit distribution."""
    name = "ood_uncertainty"

    def score(
        self,
        prompt: str,
        response: str,
        context: str = "",
        reference: str = "",
        logits: np.ndarray | None = None,
        hidden: np.ndarray | None = None,
        train_mean: np.ndarray | None = None,
        train_cov_inv: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        result: dict[str, float] = {}

        if logits is not None:
            probs = _softmax(logits.astype(np.float64))
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
            max_prob = float(probs.max())
            result["entropy"] = entropy
            result["max_prob"] = max_prob
            result["ood_uncertainty_entropy"] = entropy

        if hidden is not None and train_mean is not None and train_cov_inv is not None:
            delta = hidden.flatten() - train_mean.flatten()
            mahal = float(delta @ train_cov_inv @ delta)
            result["mahalanobis_distance"] = mahal
            result["ood_uncertainty_mahalanobis"] = mahal

        if not result:
            result["ood_uncertainty_entropy"] = 0.0

        return result

    @staticmethod
    def compute_from_token_logprobs(token_logprobs: np.ndarray) -> dict[str, float]:
        """Compute sequence-level uncertainty from per-token log-probabilities."""
        entropy = float(-token_logprobs.mean())
        perplexity = float(np.exp(-token_logprobs.mean()))
        return {
            "sequence_entropy": entropy,
            "perplexity": perplexity,
            "ood_uncertainty_entropy": entropy,
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
