"""NLI-based entailment and contradiction scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

from curvature_semantics.semantics.base import SemanticMetric
from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)


class NLIScorer(SemanticMetric):
    """Scores entailment and contradiction between premise and hypothesis via NLI model."""
    name = "nli"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        self.model_name = model_name
        self._model = None
        self._labels = ["contradiction", "entailment", "neutral"]

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=512)
            logger.info("Loaded NLI model: %s", self.model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed; NLI scores will be random placeholders")
            self._model = None

    def score(
        self,
        prompt: str,
        response: str,
        context: str = "",
        reference: str = "",
        **kwargs: Any,
    ) -> dict[str, float]:
        self._load_model()
        premise = context or prompt
        hypothesis = response
        if not hypothesis.strip():
            return {"nli_entailment": 0.0, "nli_contradiction": 0.0, "nli_neutral": 1.0}

        if self._model is None:
            rng = np.random.default_rng(abs(hash(hypothesis)) % (2**31))
            scores = rng.dirichlet([1, 1, 1])
            return {
                "nli_entailment": float(scores[1]),
                "nli_contradiction": float(scores[0]),
                "nli_neutral": float(scores[2]),
            }

        try:
            logits = self._model.predict([(premise, hypothesis)], apply_softmax=True)[0]
            # DeBERTa NLI labels: contradiction=0, entailment=1, neutral=2
            return {
                "nli_entailment": float(logits[1]),
                "nli_contradiction": float(logits[0]),
                "nli_neutral": float(logits[2]),
            }
        except Exception as exc:
            logger.warning("NLI scoring failed: %s", exc)
            return {"nli_entailment": 0.0, "nli_contradiction": 0.0, "nli_neutral": 1.0}

    def batch_score(
        self,
        premise_hypothesis_pairs: list[tuple[str, str]],
    ) -> list[dict[str, float]]:
        self._load_model()
        if self._model is None:
            return [self.score("", h, context=p) for p, h in premise_hypothesis_pairs]
        try:
            logits = self._model.predict(premise_hypothesis_pairs, apply_softmax=True)
            return [
                {
                    "nli_entailment": float(row[1]),
                    "nli_contradiction": float(row[0]),
                    "nli_neutral": float(row[2]),
                }
                for row in logits
            ]
        except Exception as exc:
            logger.warning("NLI batch scoring failed: %s", exc)
            return [{"nli_entailment": 0.0, "nli_contradiction": 0.0, "nli_neutral": 1.0}] * len(premise_hypothesis_pairs)
