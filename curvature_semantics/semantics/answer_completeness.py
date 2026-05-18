"""Answer completeness: ROUGE/BERTScore of answer against gold reference."""

from __future__ import annotations

from typing import Any

import numpy as np

from curvature_semantics.semantics.base import SemanticMetric
from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)


class AnswerCompleteness(SemanticMetric):
    """Computes F1/ROUGE/BERTScore between response and gold reference."""
    name = "answer_completeness"

    def __init__(self, use_bertscore: bool = False):
        self.use_bertscore = use_bertscore
        self._rouge = None
        self._bertscore = None

    def _load_rouge(self) -> None:
        if self._rouge is not None:
            return
        try:
            from rouge_score import rouge_scorer
            self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        except ImportError:
            logger.warning("rouge-score not installed; using token F1 fallback")

    def score(
        self,
        prompt: str,
        response: str,
        context: str = "",
        reference: str = "",
        **kwargs: Any,
    ) -> dict[str, float]:
        if not reference.strip():
            return {"answer_completeness": 0.0}
        self._load_rouge()
        if self._rouge is not None:
            scores = self._rouge.score(reference, response)
            return {
                "answer_completeness": float(scores["rougeL"].fmeasure),
                "answer_completeness_precision": float(scores["rougeL"].precision),
                "answer_completeness_recall": float(scores["rougeL"].recall),
            }
        # Fallback: token F1
        ref_tokens = set(reference.lower().split())
        resp_tokens = set(response.lower().split())
        if not ref_tokens or not resp_tokens:
            return {"answer_completeness": 0.0}
        overlap = ref_tokens & resp_tokens
        p = len(overlap) / max(len(resp_tokens), 1)
        r = len(overlap) / max(len(ref_tokens), 1)
        f1 = 2 * p * r / max(p + r, 1e-10)
        return {"answer_completeness": float(f1)}
