"""RAG retriever that scores candidates by expected curvature reduction."""

from __future__ import annotations

from typing import Any

import numpy as np

from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)


class CurvatureAwareRetriever:
    """Blends semantic similarity with estimated curvature reduction for retrieval scoring."""

    def __init__(
        self,
        documents: list[str],
        retrieval_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        curvature_weight: float = 0.4,
        top_k: int = 5,
    ):
        self.documents = documents
        self.curvature_weight = curvature_weight
        self.top_k = top_k
        self._embedder = None
        self._doc_embeddings: np.ndarray | None = None
        self._retrieval_model = retrieval_model

    def _load_embedder(self) -> None:
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._retrieval_model)
            self._doc_embeddings = self._embedder.encode(self.documents, convert_to_numpy=True)
            logger.info("Loaded retrieval model: %s, indexed %d docs", self._retrieval_model, len(self.documents))
        except ImportError:
            logger.warning("sentence-transformers not installed; retrieval will use random scores")

    def retrieve(
        self,
        query: str,
        query_curvature: float = 0.0,
        curvature_estimates: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k documents using curvature-aware scoring.

        Score = (1 - w) * sim_score + w * curvature_reduction_score
        """
        self._load_embedder()
        n = len(self.documents)

        if self._embedder is not None and self._doc_embeddings is not None:
            q_emb = self._embedder.encode([query], convert_to_numpy=True)
            norms_q = np.linalg.norm(q_emb, axis=1, keepdims=True)
            norms_d = np.linalg.norm(self._doc_embeddings, axis=1, keepdims=True)
            sim_scores = (q_emb / np.maximum(norms_q, 1e-10)) @ (self._doc_embeddings / np.maximum(norms_d, 1e-10)).T
            sim_scores = sim_scores.flatten()
        else:
            sim_scores = np.random.default_rng(42).random(n)

        # Curvature reduction: documents that are expected to reduce local curvature
        if curvature_estimates is not None and len(curvature_estimates) == n:
            curv_reduction = 1.0 - (curvature_estimates / max(curvature_estimates.max(), 1e-10))
        else:
            # Proxy: longer documents tend to provide more bridging context
            curv_reduction = np.array([min(len(d.split()) / 100.0, 1.0) for d in self.documents])

        w = self.curvature_weight
        final_scores = (1 - w) * sim_scores + w * curv_reduction

        top_idx = np.argsort(final_scores)[::-1][: self.top_k]
        return [
            {
                "document": self.documents[i],
                "score": float(final_scores[i]),
                "sim_score": float(sim_scores[i]),
                "curvature_reduction": float(curv_reduction[i]),
                "rank": rank + 1,
            }
            for rank, i in enumerate(top_idx)
        ]
