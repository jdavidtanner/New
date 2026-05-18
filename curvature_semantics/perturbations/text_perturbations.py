"""Text-level semantic-preserving perturbations."""

from __future__ import annotations

import random
import re
from typing import ClassVar

import numpy as np

from curvature_semantics.perturbations.base import TextPerturbation

# Lightweight synonym table (subset — expands at runtime via WordNet if available)
_SYNONYMS: dict[str, list[str]] = {
    "large": ["big", "sizable", "substantial"],
    "small": ["tiny", "little", "compact"],
    "fast": ["quick", "rapid", "swift"],
    "slow": ["gradual", "unhurried", "leisurely"],
    "important": ["significant", "crucial", "essential"],
    "show": ["demonstrate", "indicate", "reveal"],
    "increase": ["rise", "grow", "expand"],
    "decrease": ["fall", "drop", "decline"],
    "because": ["since", "as", "given that"],
    "however": ["nevertheless", "yet", "but"],
}


def _wordnet_synonyms(word: str) -> list[str]:
    try:
        from nltk.corpus import wordnet
        syns = set()
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                s = lemma.name().replace("_", " ")
                if s.lower() != word.lower():
                    syns.add(s)
        return list(syns)[:5]
    except Exception:
        return []


class SynonymSubstitution(TextPerturbation):
    name = "synonym_substitution"

    def __init__(self, p: float = 0.15):
        self.p = p

    def apply(self, text: str, rng: np.random.Generator | None = None) -> str:
        rng = rng or np.random.default_rng()
        words = text.split()
        result = []
        for w in words:
            key = w.lower().strip(".,;:!?\"'")
            syns = _SYNONYMS.get(key) or _wordnet_synonyms(key)
            if syns and rng.random() < self.p:
                replacement = rng.choice(syns)
                # Preserve capitalisation
                if w[0].isupper():
                    replacement = replacement.capitalize()
                result.append(replacement)
            else:
                result.append(w)
        return " ".join(result)


class SentenceReorder(TextPerturbation):
    name = "sentence_reorder"

    def apply(self, text: str, rng: np.random.Generator | None = None) -> str:
        rng = rng or np.random.default_rng()
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentences) <= 1:
            return text
        idx = rng.permutation(len(sentences)).tolist()
        return " ".join(sentences[i] for i in idx)


class ActivePassive(TextPerturbation):
    """Simple heuristic active/passive toggle (covers common patterns)."""
    name = "active_passive"

    _ACTIVE = re.compile(
        r"^(\w[\w\s]*?)\s+(was|were|is|are|has been|have been)\s+(\w[\w\s]*?)\s+by\s+(\w[\w\s]*?)[.!?]?$",
        re.IGNORECASE,
    )

    def apply(self, text: str, rng: np.random.Generator | None = None) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        result = []
        for s in sentences:
            m = self._ACTIVE.match(s.strip())
            if m:
                subj, verb, obj_, agent = m.groups()
                active_verb = {"was": "did", "were": "did", "is": "does", "are": "do",
                               "has been": "has", "have been": "have"}.get(verb.lower(), verb)
                result.append(f"{agent.strip()} {active_verb} {obj_.strip()} {subj.strip()}.")
            else:
                result.append(s)
        return " ".join(result)


class DistractorInsertion(TextPerturbation):
    """Inserts a semantically irrelevant distractor sentence."""
    name = "distractor_insertion"

    _DISTRACTORS: ClassVar[list[str]] = [
        "The weather today is partly cloudy.",
        "Many researchers find this topic fascinating.",
        "This has been a subject of discussion for years.",
        "Various opinions exist on related matters.",
        "It is worth noting that context matters greatly.",
    ]

    def apply(self, text: str, rng: np.random.Generator | None = None) -> str:
        rng = rng or np.random.default_rng()
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        pos = int(rng.integers(0, len(sentences) + 1))
        distractor = self._DISTRACTORS[int(rng.integers(0, len(self._DISTRACTORS)))]
        sentences.insert(pos, distractor)
        return " ".join(sentences)


class EvidenceReorder(TextPerturbation):
    """Shuffles evidence sentences while keeping the question prefix intact."""
    name = "evidence_reorder"

    def apply(self, text: str, rng: np.random.Generator | None = None) -> str:
        rng = rng or np.random.default_rng()
        if "Context:" in text and "Question:" in text:
            q_idx = text.find("Question:")
            header = text[:q_idx]
            question = text[q_idx:]
            # Shuffle sentences in the context portion
            ctx_start = header.find("Context:") + len("Context:")
            ctx_text = header[ctx_start:].strip()
            sents = re.split(r"(?<=[.!?])\s+", ctx_text)
            idx = rng.permutation(len(sents)).tolist()
            shuffled = " ".join(sents[i] for i in idx)
            return header[: header.find("Context:") + len("Context:")] + " " + shuffled + "\n" + question
        # Fallback: treat whole text as evidence
        return SentenceReorder().apply(text, rng)
