"""Persistent homology features: Betti numbers, persistence entropy, lifetime stats."""

from __future__ import annotations

import numpy as np

from curvature_semantics.curvature.base import CurvatureProxy


class PersistentHomologyFeatures(CurvatureProxy):
    """Extracts topological features via persistent homology (ripser or gudhi)."""
    name = "persistent_homology"

    def __init__(self, max_dim: int = 2, metric: str = "euclidean", max_points: int = 200):
        self.max_dim = max_dim
        self.metric = metric
        self.max_points = max_points  # subsample for speed

    def compute(self, hidden_states: np.ndarray) -> dict[str, float]:
        """Compute TDA features.

        Args:
            hidden_states: (n_samples, hidden_dim)
        """
        n = hidden_states.shape[0]
        if n < 4:
            return self._empty_features()

        # Subsample if needed
        X = hidden_states
        if n > self.max_points:
            idx = np.random.default_rng(42).choice(n, size=self.max_points, replace=False)
            X = X[idx]

        diagrams = self._compute_diagrams(X)
        return self._extract_features(diagrams)

    def _compute_diagrams(self, X: np.ndarray) -> list[np.ndarray]:
        """Return list of persistence diagrams (one per homology dimension)."""
        try:
            import ripser
            result = ripser.ripser(X, maxdim=self.max_dim, metric=self.metric)
            return result["dgms"]
        except ImportError:
            pass

        try:
            import gudhi
            rips = gudhi.RipsComplex(points=X.tolist(), max_edge_length=np.ptp(X) * 2)
            st = rips.create_simplex_tree(max_dimension=self.max_dim + 1)
            st.compute_persistence()
            diagrams = []
            for dim in range(self.max_dim + 1):
                pairs = [(b, d) for (d_, (b, d)) in st.persistence() if d_ == dim and d != float("inf")]
                diagrams.append(np.array(pairs) if pairs else np.zeros((0, 2)))
            return diagrams
        except ImportError:
            pass

        return [np.zeros((0, 2)) for _ in range(self.max_dim + 1)]

    def _extract_features(self, diagrams: list[np.ndarray]) -> dict[str, float]:
        features: dict[str, float] = {}
        for dim, dgm in enumerate(diagrams):
            finite = dgm[dgm[:, 1] < np.inf] if dgm.size > 0 else dgm
            lifetimes = finite[:, 1] - finite[:, 0] if finite.size > 0 else np.array([])
            features[f"betti_{dim}"] = float(len(lifetimes))
            features[f"persistence_entropy_{dim}"] = float(self._entropy(lifetimes))
            features[f"lifetime_mean_{dim}"] = float(lifetimes.mean()) if len(lifetimes) > 0 else 0.0
            features[f"lifetime_max_{dim}"] = float(lifetimes.max()) if len(lifetimes) > 0 else 0.0
        return features

    @staticmethod
    def _entropy(lifetimes: np.ndarray) -> float:
        if len(lifetimes) == 0:
            return 0.0
        total = lifetimes.sum()
        if total < 1e-10:
            return 0.0
        probs = lifetimes / total
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    def _empty_features(self) -> dict[str, float]:
        features: dict[str, float] = {}
        for dim in range(self.max_dim + 1):
            features[f"betti_{dim}"] = 0.0
            features[f"persistence_entropy_{dim}"] = 0.0
            features[f"lifetime_mean_{dim}"] = 0.0
            features[f"lifetime_max_{dim}"] = 0.0
        return features
