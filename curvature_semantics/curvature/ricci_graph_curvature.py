"""Ollivier-Ricci curvature on kNN graphs of hidden states."""

from __future__ import annotations

import numpy as np

from curvature_semantics.curvature.base import CurvatureProxy


class RicciGraphCurvature(CurvatureProxy):
    """Computes Ollivier-Ricci curvature on the kNN graph of hidden states.

    Uses a discrete approximation via the earth-mover's distance between
    neighbourhood probability distributions. Falls back to a linear-program
    (POT library) if available, otherwise uses a closed-form approximation.
    """
    name = "ricci_graph_curvature"

    def __init__(self, k: int = 10, alpha: float = 0.5):
        self.k = k
        self.alpha = alpha  # idleness parameter

    def compute(self, hidden_states: np.ndarray) -> dict[str, float]:
        """Compute mean Ollivier-Ricci curvature over all edges.

        Args:
            hidden_states: (n_samples, hidden_dim)
        """
        n = hidden_states.shape[0]
        k = min(self.k, n - 1)
        if n < 3:
            return {"ricci_graph_curvature": 0.0, "ricci_graph_curvature_std": 0.0}

        dists = _pairwise_l2(hidden_states)
        np.fill_diagonal(dists, np.inf)
        nn_idx = np.argsort(dists, axis=1)[:, :k]
        np.fill_diagonal(dists, 0.0)

        edge_curvatures = []
        for i in range(n):
            for j in nn_idx[i]:
                kappa = self._edge_curvature(i, j, dists, nn_idx)
                edge_curvatures.append(kappa)

        if not edge_curvatures:
            return {"ricci_graph_curvature": 0.0, "ricci_graph_curvature_std": 0.0}

        arr = np.array(edge_curvatures)
        return {
            "ricci_graph_curvature": float(arr.mean()),
            "ricci_graph_curvature_std": float(arr.std()),
            "ricci_graph_curvature_min": float(arr.min()),
            "ricci_graph_curvature_max": float(arr.max()),
        }

    def _edge_curvature(
        self,
        i: int,
        j: int,
        dists: np.ndarray,
        nn_idx: np.ndarray,
    ) -> float:
        d_ij = dists[i, j]
        if d_ij < 1e-10:
            return 0.0

        # Probability distributions on neighbourhoods (lazy random walk)
        mu_i = self._lazy_dist(i, nn_idx, dists)
        mu_j = self._lazy_dist(j, nn_idx, dists)

        W = self._wasserstein_approx(mu_i, mu_j, dists)
        return float(1.0 - W / d_ij)

    def _lazy_dist(self, i: int, nn_idx: np.ndarray, dists: np.ndarray) -> dict[int, float]:
        neighbors = nn_idx[i].tolist()
        k = len(neighbors)
        dist = {i: self.alpha}
        weight = (1 - self.alpha) / k
        for nb in neighbors:
            dist[nb] = dist.get(nb, 0.0) + weight
        return dist

    @staticmethod
    def _wasserstein_approx(
        mu: dict[int, float],
        nu: dict[int, float],
        dists: np.ndarray,
    ) -> float:
        """Approximate W1 using POT if available, else closed-form lower bound."""
        try:
            import ot
            nodes_mu = sorted(mu.keys())
            nodes_nu = sorted(nu.keys())
            a = np.array([mu[n] for n in nodes_mu])
            b = np.array([nu[n] for n in nodes_nu])
            M = dists[np.ix_(nodes_mu, nodes_nu)]
            return float(ot.emd2(a, b, M))
        except ImportError:
            pass

        # Closed-form approximation: mean displacement
        all_nodes = set(mu) | set(nu)
        total = 0.0
        for node in all_nodes:
            p = mu.get(node, 0.0)
            q = nu.get(node, 0.0)
            total += abs(p - q)
        return total * 0.5  # crude lower bound


def _pairwise_l2(X: np.ndarray) -> np.ndarray:
    sq = (X ** 2).sum(axis=1, keepdims=True)
    return np.sqrt(np.maximum(sq + sq.T - 2 * X @ X.T, 0.0))
