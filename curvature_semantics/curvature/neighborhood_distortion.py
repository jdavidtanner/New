"""Neighborhood distortion ratio: kNN rank preservation across layer transitions."""

from __future__ import annotations

import numpy as np

from curvature_semantics.curvature.base import CurvatureProxy


class NeighborhoodDistortionRatio(CurvatureProxy):
    """Measures how much local kNN structure is preserved between two layers.

    A score of 1.0 means perfect preservation; 0.0 means total disruption.
    Higher distortion → higher curvature-like behaviour.
    """
    name = "neighborhood_distortion"

    def __init__(self, k: int = 15):
        self.k = k

    def compute(self, hidden_states: np.ndarray) -> dict[str, float]:
        """Compute self-distortion within a single layer (intra-layer stability).

        Args:
            hidden_states: (n_samples, hidden_dim)
        """
        n = hidden_states.shape[0]
        k = min(self.k, n - 1)
        if n < 3:
            return {"neighborhood_distortion": 0.0}
        dists = _pairwise_l2(hidden_states)
        np.fill_diagonal(dists, np.inf)
        nn_indices = np.argsort(dists, axis=1)[:, :k]

        # Compute trustworthiness: fraction of true kNNs present in kNN graph
        rank_matrix = np.argsort(np.argsort(dists, axis=1), axis=1)
        T = 0.0
        for i in range(n):
            for j in nn_indices[i]:
                r = rank_matrix[j, i]
                if r > k:
                    T += r - k
        n_term = 2.0 / (n * k * (2 * n - 3 * k - 1))
        trustworthiness = 1.0 - n_term * T
        return {
            "neighborhood_distortion": float(1.0 - trustworthiness),
            "trustworthiness": float(trustworthiness),
        }

    def compute_between_layers(
        self,
        layer_a: np.ndarray,
        layer_b: np.ndarray,
    ) -> dict[str, float]:
        """Measure kNN rank preservation from layer_a to layer_b.

        Args:
            layer_a: (n_samples, hidden_dim_a)
            layer_b: (n_samples, hidden_dim_b)
        """
        n = layer_a.shape[0]
        k = min(self.k, n - 1)
        if n < 3:
            return {"neighborhood_distortion": 0.0}

        dists_a = _pairwise_l2(layer_a)
        dists_b = _pairwise_l2(layer_b)
        np.fill_diagonal(dists_a, np.inf)
        np.fill_diagonal(dists_b, np.inf)

        nn_a = set(map(tuple, np.argsort(dists_a, axis=1)[:, :k]))
        nn_b = set(map(tuple, np.argsort(dists_b, axis=1)[:, :k]))

        overlap = len(nn_a & nn_b) / max(len(nn_a | nn_b), 1)
        return {
            "neighborhood_distortion": float(1.0 - overlap),
            "nn_overlap": float(overlap),
        }


def _pairwise_l2(X: np.ndarray) -> np.ndarray:
    sq = (X ** 2).sum(axis=1, keepdims=True)
    return np.sqrt(np.maximum(sq + sq.T - 2 * X @ X.T, 0.0))
