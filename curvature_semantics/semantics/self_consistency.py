"""Self-consistency: agreement among multiple generation samples."""

from __future__ import annotations

from typing import Any

import numpy as np

from curvature_semantics.semantics.base import SemanticMetric


class SelfConsistency(SemanticMetric):
    """Measures self-consistency as pairwise semantic similarity across N generations."""
    name = "self_consistency"

    def __init__(self, similarity_threshold: float = 0.8):
        self.threshold = similarity_threshold
        self._embedder = None

    def _load_embedder(self) -> None:
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            self._embedder = None

    def score(
        self,
        prompt: str,
        response: str,
        context: str = "",
        reference: str = "",
        generations: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        if not generations or len(generations) < 2:
            return {"self_consistency": 1.0}

        self._load_embedder()
        if self._embedder is None:
            # Fallback: lexical overlap
            return {"self_consistency": float(self._lexical_consistency(generations))}

        embeddings = self._embedder.encode(generations, convert_to_numpy=True)
        sims = _cosine_similarity_matrix(embeddings)
        n = len(generations)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pairwise = sims[mask]
        return {
            "self_consistency": float(pairwise.mean()),
            "self_consistency_std": float(pairwise.std()),
            "self_consistency_min": float(pairwise.min()),
        }

    @staticmethod
    def _lexical_consistency(texts: list[str]) -> float:
        def tokens(t: str) -> set[str]:
            return set(t.lower().split())
        pairs = [(tokens(texts[i]), tokens(texts[j])) for i in range(len(texts)) for j in range(i + 1, len(texts))]
        overlaps = [len(a & b) / max(len(a | b), 1) for a, b in pairs]
        return float(np.mean(overlaps)) if overlaps else 1.0


def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.maximum(norms, 1e-10)
    return X_norm @ X_norm.T
