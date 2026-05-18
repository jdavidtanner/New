"""Abstention accuracy: precision/recall of model's refusal vs known unknowables."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from curvature_semantics.semantics.base import SemanticMetric

_ABSTENTION_PATTERNS = [
    r"\bI don'?t know\b",
    r"\bI'?m not sure\b",
    r"\bI cannot\b",
    r"\bcannot be determined\b",
    r"\bunable to\b",
    r"\bunclear\b",
    r"\bI'?m uncertain\b",
    r"\bno information\b",
    r"\boutside my knowledge\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _ABSTENTION_PATTERNS]


class AbstentionAccuracy(SemanticMetric):
    """Measures whether the model abstains appropriately on unknowable queries."""
    name = "abstention_accuracy"

    def score(
        self,
        prompt: str,
        response: str,
        context: str = "",
        reference: str = "",
        should_abstain: bool = False,
        **kwargs: Any,
    ) -> dict[str, float]:
        did_abstain = self.detect_abstention(response)
        if should_abstain:
            # True positive: correctly abstained
            tp = 1.0 if did_abstain else 0.0
            return {"abstention_accuracy": tp, "abstention_precision": tp, "abstention_recall": tp}
        else:
            # False positive: incorrectly abstained when answer exists
            fp = 1.0 if did_abstain else 0.0
            return {"abstention_accuracy": 1.0 - fp, "false_abstention_rate": fp}

    @staticmethod
    def detect_abstention(text: str) -> bool:
        return any(p.search(text) for p in _COMPILED)

    @staticmethod
    def compute_corpus_metrics(
        predictions: list[dict[str, Any]],
        should_abstain_labels: list[bool],
    ) -> dict[str, float]:
        """Compute precision/recall/F1 over a corpus."""
        tp = fp = fn = tn = 0
        for pred, label in zip(predictions, should_abstain_labels):
            did = AbstentionAccuracy.detect_abstention(pred.get("response", ""))
            if label and did:
                tp += 1
            elif not label and did:
                fp += 1
            elif label and not did:
                fn += 1
            else:
                tn += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        return {
            "abstention_precision": float(precision),
            "abstention_recall": float(recall),
            "abstention_f1": float(f1),
            "abstention_accuracy": float((tp + tn) / max(tp + fp + fn + tn, 1)),
        }
