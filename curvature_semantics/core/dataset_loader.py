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
        ds = load_dataset(ds_name, config, split=split)
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
    # Openbookqa / multiple-choice: build a readable prompt from question_stem + choices
    if "question_stem" in row:
        choices = row.get("choices", {})
        choice_texts = choices.get("text", []) if isinstance(choices, dict) else []
        choice_labels = choices.get("label", []) if isinstance(choices, dict) else []
        options_str = "  ".join(f"({l}) {t}" for l, t in zip(choice_labels, choice_texts))
        prompt = row["question_stem"] + (f"  Options: {options_str}" if options_str else "")
        answer = str(row.get("answerKey", ""))
        return {"domain": domain, "prompt": prompt, "answer": answer, "context": ""}

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
    """Generate varied placeholder examples for development/testing."""
    import random
    rng = random.Random(42)

    _COUNTRIES = ["France", "Japan", "Brazil", "Australia", "Canada", "India", "Germany", "Egypt"]
    _DRUGS_A = ["aspirin", "ibuprofen", "metformin", "lisinopril", "warfarin", "methotrexate"]
    _DRUGS_B = ["acetaminophen", "naproxen", "atorvastatin", "amlodipine", "clopidogrel", "prednisone"]
    _CONDITIONS = ["headache", "hypertension", "diabetes", "arthritis", "heart failure", "asthma"]
    _PHYSICS = [
        "What is the speed of light in a vacuum?",
        "How does a lever work?",
        "What causes a rainbow?",
        "Why does ice float on water?",
        "What is Newton's third law?",
        "How does a magnetic field form?",
        "What is kinetic energy?",
        "Why do objects fall at the same rate in a vacuum?",
        "What is the Doppler effect?",
        "How does a prism separate white light?",
    ]
    _MEDICAL = [
        "What are the symptoms of type 2 diabetes?",
        "How is hypertension diagnosed?",
        "What causes chronic migraines?",
        "Is ibuprofen safe for long-term use?",
        "What is the difference between viral and bacterial pneumonia?",
        "How do beta-blockers work?",
        "What are the risk factors for stroke?",
        "Can stress cause heart disease?",
        "What is the recommended treatment for mild depression?",
        "How does the flu vaccine work?",
    ]
    _SPEC_MED = [
        f"Is it safe to combine {rng.choice(_DRUGS_A)} and {rng.choice(_DRUGS_B)} for treating {rng.choice(_CONDITIONS)}?"
        for _ in range(max(n, 20))
    ]
    _BIO = [
        f"Who was the {rng.choice(['first','second','third','last'])} person to {rng.choice(['climb Everest','walk on the moon','win the Nobel Prize in Physics','sail around the world'])}?",
        *[f"Describe the life of person number {i} in the synthetic biography dataset." for i in range(20)]
    ]

    domain_pools: dict[str, list[str]] = {
        "arithmetic": [f"What is {rng.randint(1,999)} {'+'  if rng.random()>0.5 else '*'} {rng.randint(1,99)}?" for _ in range(max(n, 20))],
        "geography": [f"What is the capital of {rng.choice(_COUNTRIES)}?" for _ in range(max(n, 20))],
        "basic_physics": (_PHYSICS * ((n // len(_PHYSICS)) + 2))[:max(n, 20)],
        "medical_advice": (_MEDICAL * ((n // len(_MEDICAL)) + 2))[:max(n, 20)],
        "speculative_medicine": _SPEC_MED,
        "synthetic_biography": _BIO * ((n // len(_BIO)) + 2),
    }

    pool = domain_pools.get(domain)
    if pool is None:
        pool = [f"[{domain}] Question {i}: {rng.randint(0,9999)}?" for i in range(max(n, 20))]

    rng.shuffle(pool)
    examples = []
    for i in range(n):
        prompt = pool[i % len(pool)]
        examples.append({"domain": domain, "prompt": prompt, "answer": "", "context": ""})
    return examples


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
