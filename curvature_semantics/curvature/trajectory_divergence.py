"""Local trajectory divergence: how perturbed paths separate across layers."""

from __future__ import annotations

import numpy as np

from curvature_semantics.curvature.base import CurvatureProxy


class LocalTrajectoryDivergence(CurvatureProxy):
    """Measures divergence between original and perturbed hidden-state trajectories.

    Given two sets of layerwise representations (original and perturbed), this
    computes the mean L2 distance across layers, normalised by the initial distance.
    """
    name = "trajectory_divergence"

    def __init__(self, n_neighbors: int = 10):
        self.n_neighbors = n_neighbors

    def compute(self, hidden_states: np.ndarray) -> dict[str, float]:
        """Compute intra-batch trajectory divergence.

        Args:
            hidden_states: (n_samples, hidden_dim)
        Returns:
            {"trajectory_divergence": float, "trajectory_divergence_std": float}
        """
        n = hidden_states.shape[0]
        if n < 2:
            return {"trajectory_divergence": 0.0, "trajectory_divergence_std": 0.0}

        # Pairwise L2 distances
        diff = hidden_states[:, np.newaxis, :] - hidden_states[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(-1))  # (n, n)
        np.fill_diagonal(dists, np.inf)
        # For each point, mean distance to its k nearest neighbours
        k = min(self.n_neighbors, n - 1)
        nn_dists = np.sort(dists, axis=1)[:, :k]
        mean_nn = nn_dists.mean(axis=1)
        return {
            "trajectory_divergence": float(mean_nn.mean()),
            "trajectory_divergence_std": float(mean_nn.std()),
        }

    def compute_from_trajectory_pair(
        self,
        original: np.ndarray,
        perturbed: np.ndarray,
    ) -> dict[str, float]:
        """Compare original vs perturbed trajectories across layers.

        Args:
            original:  (n_layers, hidden_dim)
            perturbed: (n_layers, hidden_dim)
        """
        assert original.shape == perturbed.shape, "Shape mismatch"
        diffs = np.linalg.norm(original - perturbed, axis=-1)  # (n_layers,)
        init_dist = float(diffs[0]) if diffs[0] > 1e-10 else 1.0
        normalised = diffs / init_dist
        return {
            "trajectory_divergence": float(normalised.mean()),
            "trajectory_divergence_std": float(normalised.std()),
            "trajectory_divergence_max": float(normalised.max()),
            "trajectory_divergence_final": float(normalised[-1]),
        }
