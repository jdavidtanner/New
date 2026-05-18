"""Expected calibration error (ECE) and reliability diagram data."""

from __future__ import annotations

from typing import Any

import numpy as np

from curvature_semantics.semantics.base import SemanticMetric


class CalibrationError(SemanticMetric):
    """Computes ECE given a list of (confidence, correctness) pairs."""
    name = "calibration_error"

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def score(
        self,
        prompt: str,
        response: str,
        context: str = "",
        reference: str = "",
        confidence: float | None = None,
        correct: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        if confidence is None or correct is None:
            return {"calibration_error": 0.0}
        ece = abs(confidence - float(correct))
        return {"calibration_error": float(ece)}

    def compute_ece(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
    ) -> dict[str, float]:
        """Compute ECE from arrays.

        Args:
            confidences: (n,) float in [0, 1]
            correctness: (n,) binary 0/1
        """
        n = len(confidences)
        bins = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        mce = 0.0
        reliability: list[dict[str, float]] = []

        for i in range(self.n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (confidences >= lo) & (confidences < hi) if i < self.n_bins - 1 else (confidences >= lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            acc = correctness[mask].mean()
            conf = confidences[mask].mean()
            gap = abs(acc - conf)
            ece += (mask.sum() / n) * gap
            mce = max(mce, gap)
            reliability.append({"bin_mid": float((lo + hi) / 2), "accuracy": float(acc), "confidence": float(conf)})

        return {
            "calibration_error": float(ece),
            "max_calibration_error": float(mce),
            "overconfidence": float((confidences - correctness).mean()),
        }
