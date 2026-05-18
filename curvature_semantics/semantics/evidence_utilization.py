"""Evidence utilization: attribution overlap between retrieved context and answer."""

from __future__ import annotations

from typing import Any

import numpy as np

from curvature_semantics.semantics.base import SemanticMetric


class EvidenceUtilization(SemanticMetric):
    """Measures how much of the retrieved context is reflected in the answer."""
    name = "evidence_utilization"

    def __init__(self, min_token_len: int = 3):
        self.min_token_len = min_token_len

    def score(
        self,
        prompt: str,
        response: str,
        context: str = "",
        reference: str = "",
        **kwargs: Any,
    ) -> dict[str, float]:
        if not context or not response:
            return {"evidence_utilization": 0.0}
        ctx_tokens = self._tokenize(context)
        resp_tokens = self._tokenize(response)
        if not ctx_tokens or not resp_tokens:
            return {"evidence_utilization": 0.0}
        overlap = ctx_tokens & resp_tokens
        precision = len(overlap) / max(len(resp_tokens), 1)
        recall = len(overlap) / max(len(ctx_tokens), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        return {
            "evidence_utilization": float(f1),
            "evidence_precision": float(precision),
            "evidence_recall": float(recall),
        }

    def _tokenize(self, text: str) -> set[str]:
        return {
            w.lower().strip(".,;:!?\"'")
            for w in text.split()
            if len(w) >= self.min_token_len
        }
