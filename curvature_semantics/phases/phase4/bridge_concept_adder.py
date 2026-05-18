"""Bridge concept insertion: adds intermediate reasoning steps to prompts."""

from __future__ import annotations

from typing import Any


_BRIDGE_TEMPLATES = [
    "Note that {concept} connects these ideas.",
    "Consider that {concept} is a key intermediary concept here.",
    "The relationship between these concepts is mediated by {concept}.",
]


def add_bridge_concepts(
    prompt: str,
    concepts: list[str],
    max_concepts: int = 3,
) -> str:
    """Append bridge concept hints to the prompt."""
    if not concepts:
        return prompt
    selected = concepts[:max_concepts]
    hints = " ".join(
        _BRIDGE_TEMPLATES[i % len(_BRIDGE_TEMPLATES)].format(concept=c)
        for i, c in enumerate(selected)
    )
    return f"{prompt}\n\n[Reasoning hints: {hints}]"


def build_bridge_variants(
    examples: list[dict[str, Any]],
    concept_map: dict[str, list[str]] | None = None,
    max_concepts: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    concept_map = concept_map or {}
    augmented = []
    for ex in examples:
        domain = ex.get("domain", "")
        concepts = concept_map.get(domain, [f"{domain}_bridge_{i}" for i in range(max_concepts)])
        aug = dict(ex)
        aug["prompt"] = add_bridge_concepts(ex["prompt"], concepts, max_concepts)
        aug["intervention"] = "bridge_concept"
        augmented.append(aug)
    return examples, augmented
