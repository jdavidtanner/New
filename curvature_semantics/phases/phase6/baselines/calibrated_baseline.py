"""Baseline: confidence-calibrated routing (abstain when confidence < threshold)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


class CalibratedBaseline:
    name = "calibrated"

    def __init__(self, confidence_threshold: float = 0.5):
        self.threshold = confidence_threshold

    def run(
        self,
        prompt: str,
        generate_fn: Callable[[str], str],
        logits: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        confidence = 1.0
        if logits is not None:
            probs = _softmax(logits.astype(np.float64))
            confidence = float(probs.max())

        if confidence < self.threshold:
            return {
                "action": "abstain",
                "response": "I'm not confident enough to answer this.",
                "baseline": self.name,
                "confidence": confidence,
            }
        response = generate_fn(prompt)
        return {"action": "answer", "response": response, "baseline": self.name, "confidence": confidence}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
