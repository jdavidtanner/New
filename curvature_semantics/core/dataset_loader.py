"""Dataset loading: HuggingFace datasets and custom JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)

# Mapping domain names → (hf_dataset_name, config, split)
DOMAIN_DATASET_MAP: dict[str, tuple[str, str | None, str]] = {
    "arithmetic": ("allenai/math_qa", None, "train"),
    "geography": ("coastalcph/geo880", None, "test"),
    "basic_physics": ("allenai/openbookqa", "main", "test"),
    "biological_taxonomy": ("allenai/openbookqa", "main", "test"),
    "historical_dates": ("trivia_qa", "rc", "validation"),
    "legal_reasoning": ("nguyen-brat/legalbench", None, "train"),
    "medical_advice": ("bigbio/med_qa", None, "test"),
    "historical_interpretation": ("trivia_qa", "rc", "validation"),
    "speculative_medicine": ("allenai/qasper", None, "validation"),
    "synthetic_biography": ("trivia_qa", "rc", "validation"),
    "fictional_canon_blending": ("trivia_qa", "rc", "validation"),
}


def load_domain_examples(
    domain: str,
    n: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load up to n examples for a given domain."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets not installed; returning synthetic examples")
        return _synthetic_examples(domain, n)

    mapping = DOMAIN_DATASET_MAP.get(domain)
    if mapping is None:
        logger.warning("No dataset mapping for domain '%s'; using synthetic", domain)
        return _synthetic_examples(domain, n)

    ds_name, config, split = mapping
    try:
        logger.info("Loading dataset %s (config=%s, split=%s)", ds_name, config, split)
        ds = load_dataset(ds_name, config, split=split, trust_remote_code=True)
        ds = ds.shuffle(seed=seed)
        examples = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            examples.append(_normalize_row(domain, row))
        return examples
    except Exception as exc:
        logger.warning("Failed to load %s: %s — using synthetic", ds_name, exc)
        return _synthetic_examples(domain, n)


def _normalize_row(domain: str, row: dict[str, Any]) -> dict[str, Any]:
    """Convert various dataset schemas to a unified dict."""
    prompt = (
        row.get("question")
        or row.get("input")
        or row.get("text")
        or str(row)
    )
    answer = (
        row.get("answer")
        or row.get("output")
        or row.get("label")
        or ""
    )
    context = row.get("context") or row.get("passage") or ""
    return {"domain": domain, "prompt": str(prompt), "answer": str(answer), "context": str(context)}


def _synthetic_examples(domain: str, n: int) -> list[dict[str, Any]]:
    """Generate placeholder examples for development/testing."""
    templates = {
        "arithmetic": "What is {a} + {b}?",
        "geography": "What is the capital of {country}?",
        "speculative_medicine": "Is it safe to combine {drugA} and {drugB} for treating {condition}?",
    }
    import random
    rng = random.Random(42)
    template = templates.get(domain, f"[{domain}] Question {{i}}?")
    examples = []
    for i in range(n):
        prompt = template.format(
            i=i, a=rng.randint(1, 100), b=rng.randint(1, 100),
            country="France", drugA="aspirin", drugB="ibuprofen",
            condition="headache",
        )
        examples.append({"domain": domain, "prompt": prompt, "answer": "42", "context": ""})
    return examples


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
