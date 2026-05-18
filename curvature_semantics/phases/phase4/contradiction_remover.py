"""NLI-guided contradiction removal from context."""

from __future__ import annotations

import re
from typing import Any


def remove_contradictions(
    prompt: str,
    context: str,
    nli_scorer: Any,
    threshold: float = 0.7,
) -> str:
    """Remove sentences from context that contradict the prompt."""
    if not context:
        return context
    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    kept = []
    pairs = [(prompt, s) for s in sentences]
    scores = nli_scorer.batch_score(pairs)
    for sentence, score in zip(sentences, scores):
        if score.get("nli_contradiction", 0.0) < threshold:
            kept.append(sentence)
    return " ".join(kept)


def build_decontradicted_variants(
    examples: list[dict[str, Any]],
    nli_scorer: Any,
    threshold: float = 0.7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    augmented = []
    for ex in examples:
        clean_context = remove_contradictions(
            ex["prompt"], ex.get("context", ""), nli_scorer, threshold
        )
        aug = dict(ex)
        aug["context"] = clean_context
        aug["intervention"] = "contradiction_removal"
        augmented.append(aug)
    return examples, augmented
