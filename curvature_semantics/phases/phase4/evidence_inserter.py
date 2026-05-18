"""Targeted evidence insertion: prepends gold-context sentences to prompts."""

from __future__ import annotations

from typing import Any


def insert_evidence(
    prompt: str,
    context: str,
    num_sentences: int = 3,
) -> str:
    """Prepend the top-N context sentences before the prompt."""
    if not context:
        return prompt
    import re
    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    selected = sentences[:num_sentences]
    prefix = " ".join(selected)
    return f"Context: {prefix}\n\nQuestion: {prompt}"


def build_evidence_variants(
    examples: list[dict[str, Any]],
    num_sentences: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (original_examples, evidence_augmented_examples) pairs."""
    augmented = []
    for ex in examples:
        aug = dict(ex)
        aug["prompt"] = insert_evidence(ex["prompt"], ex.get("context", ""), num_sentences)
        aug["intervention"] = "evidence_insertion"
        augmented.append(aug)
    return examples, augmented
