"""Geodesic deviation proxy: shortest-path vs Euclidean distance ratio."""

from __future__ import annotations

import numpy as np

from curvature_semantics.curvature.base import CurvatureProxy


class GeodesicDeviationProxy(CurvatureProxy):
    """Estimates curvature by comparing graph geodesic distance to Euclidean distance.

    On a flat manifold these are equal; positive curvature causes geodesic < Euclidean
    (sphere-like); negative curvature causes geodesic > Euclidean (hyperbolic-like).
    We return the signed ratio deviation from 1.
    """
    name = "geodesic_deviation"

    def __init__(self, n_samples: int = 50, k: int = 7):
        self.n_samples = n_samples
        self.k = k

    def compute(self, hidden_states: np.ndarray) -> dict[str, float]:
        """Compute geodesic deviation.

        Args:
            hidden_states: (n_samples, hidden_dim)
        """
        n = hidden_states.shape[0]
        k = min(self.k, n - 1)
        if n < 4:
            return {"geodesic_deviation": 0.0, "geodesic_deviation_std": 0.0}

        euc_dists = _pairwise_l2(hidden_states)

        # Build kNN graph (adjacency with Euclidean edge weights)
        adj = np.full((n, n), np.inf)
        np.fill_diagonal(adj, 0.0)
        nn_idx = np.argsort(euc_dists, axis=1)[:, 1: k + 1]
        for i in range(n):
            for j in nn_idx[i]:
                adj[i, j] = euc_dists[i, j]
                adj[j, i] = euc_dists[j, i]

        geo_dists = _floyd_warshall(adj)

        # Sample random pairs and compute ratio
        rng = np.random.default_rng(42)
        m = min(self.n_samples, n * (n - 1) // 2)
        pairs = rng.choice(n, size=(m, 2), replace=True)
        mask = pairs[:, 0] != pairs[:, 1]
        pairs = pairs[mask][:m]

        ratios = []
        for i, j in pairs:
            g = geo_dists[i, j]
            e = euc_dists[i, j]
            if e > 1e-10 and g < np.inf:
                ratios.append(g / e)

        if not ratios:
            return {"geodesic_deviation": 0.0, "geodesic_deviation_std": 0.0}

        ratios_arr = np.array(ratios)
        deviation = ratios_arr.mean() - 1.0  # negative = positive curvature
        return {
            "geodesic_deviation": float(deviation),
            "geodesic_deviation_std": float(ratios_arr.std()),
            "geodesic_ratio_mean": float(ratios_arr.mean()),
        }


def _pairwise_l2(X: np.ndarray) -> np.ndarray:
    sq = (X ** 2).sum(axis=1, keepdims=True)
    return np.sqrt(np.maximum(sq + sq.T - 2 * X @ X.T, 0.0))


def _floyd_warshall(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    dist = adj.copy()
    for k in range(n):
        new_dist = dist[:, k:k+1] + dist[k:k+1, :]
        dist = np.minimum(dist, new_dist)
    return dist
