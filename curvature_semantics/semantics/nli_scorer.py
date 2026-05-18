"""NLI-based entailment and contradiction scoring."""

from __future__ import annotations

import re
from typing import Any

from curvature_semantics.core.logging_utils import get_logger
from curvature_semantics.semantics.base import SemanticMetric

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
            logger.warning(
                "sentence-transformers not installed; using deterministic lexical NLI fallback"
            )
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
            return _lexical_nli(premise, hypothesis)

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


def _token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 1}


def _lexical_nli(premise: str, hypothesis: str) -> dict[str, float]:
    """Deterministic, dependency-free approximation used when no NLI model is installed."""
    p_tokens = _token_set(premise)
    h_tokens = _token_set(hypothesis)
    if not h_tokens:
        return {"nli_entailment": 0.0, "nli_contradiction": 0.0, "nli_neutral": 1.0}

    overlap = len(p_tokens & h_tokens) / max(len(h_tokens), 1)
    combined = f" {premise.lower()} {hypothesis.lower()} "
    contradiction_cues = (
        " not ", " never ", " no ", " false", " incorrect", "contradict", "opposite"
    )
    contradiction = 0.65 if any(cue in combined for cue in contradiction_cues) and overlap < 0.8 else 0.0
    entailment = max(0.0, min(0.95, overlap * (1.0 - 0.5 * contradiction)))
    neutral = max(0.0, 1.0 - entailment - contradiction)
    total = entailment + contradiction + neutral
    return {
        "nli_entailment": float(entailment / total),
        "nli_contradiction": float(contradiction / total),
        "nli_neutral": float(neutral / total),
    }
