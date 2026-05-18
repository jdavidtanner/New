"""NLP helpers: tokenization, sentence splitting, paraphrase detection."""

from __future__ import annotations

import re


def split_sentences(text: str, min_len: int = 5) -> list[str]:
    """Split text into sentences using regex."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= min_len]


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def token_overlap_f1(a: str, b: str) -> float:
    """Token-level F1 between two strings (EM-style)."""
    a_tokens = set(simple_tokenize(a))
    b_tokens = set(simple_tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = a_tokens & b_tokens
    p = len(overlap) / len(b_tokens)
    r = len(overlap) / len(a_tokens)
    return 2 * p * r / max(p + r, 1e-10)


def truncate_to_tokens(text: str, max_tokens: int = 512) -> str:
    """Naive word-based truncation."""
    words = text.split()
    return " ".join(words[:max_tokens])


def is_paraphrase(a: str, b: str, threshold: float = 0.8) -> bool:
    """Rough paraphrase detection via token overlap."""
    return token_overlap_f1(a, b) >= threshold
