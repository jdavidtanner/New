"""Claim-level factual consistency via sentence-level NLI."""

from __future__ import annotations

import re
from typing import Any

from curvature_semantics.semantics.base import SemanticMetric
from curvature_semantics.semantics.nli_scorer import NLIScorer


class FactualityScorer(SemanticMetric):
    """Computes factuality as the fraction of response sentences entailed by context."""
    name = "factuality"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", threshold: float = 0.5):
        self.nli = NLIScorer(model_name)
        self.threshold = threshold

    def score(
        self,
        prompt: str,
        response: str,
        context: str = "",
        reference: str = "",
        **kwargs: Any,
    ) -> dict[str, float]:
        if not response.strip() or not (context or reference):
            return {"factuality": 0.5}  # neutral when no context

        premise = context or reference
        sentences = _split_sentences(response)
        if not sentences:
            return {"factuality": 0.5}

        pairs = [(premise, s) for s in sentences]
        results = self.nli.batch_score(pairs)
        entailed = sum(1 for r in results if r["nli_entailment"] >= self.threshold)
        contradicted = sum(1 for r in results if r["nli_contradiction"] >= self.threshold)
        n = len(sentences)
        return {
            "factuality": float(entailed / n),
            "factuality_contradiction_rate": float(contradicted / n),
            "factuality_n_sentences": float(n),
        }


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]
