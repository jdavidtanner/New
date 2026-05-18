"""Intrinsic dimension estimation: TwoNN and MLE estimators."""

from __future__ import annotations

import numpy as np

from curvature_semantics.curvature.base import CurvatureProxy


class IntrinsicDimension(CurvatureProxy):
    """Estimates the intrinsic dimension of a point cloud.

    TwoNN estimator: Facco et al. (2017) — uses ratio of 1st and 2nd nearest-neighbor distances.
    MLE estimator: Levina & Bickel (2004) — local log-distance averaging.
    """
    name = "intrinsic_dimension"

    def __init__(self, method: str = "twonn"):
        if method not in ("twonn", "mle"):
            raise ValueError(f"Unknown method: {method!r}")
        self.method = method

    def compute(self, hidden_states: np.ndarray) -> dict[str, float]:
        """Estimate intrinsic dimension.

        Args:
            hidden_states: (n_samples, hidden_dim)
        """
        n = hidden_states.shape[0]
        if n < 3:
            return {"intrinsic_dimension": 0.0}
        if self.method == "twonn":
            return {"intrinsic_dimension": float(self._twonn(hidden_states))}
        return {"intrinsic_dimension": float(self._mle(hidden_states))}

    @staticmethod
    def _twonn(X: np.ndarray) -> float:
        """Two-nearest-neighbour intrinsic dimension estimator."""
        n = X.shape[0]
        dists = _pairwise_l2(X)
        np.fill_diagonal(dists, np.inf)
        sorted_d = np.sort(dists, axis=1)
        r1 = sorted_d[:, 0]
        r2 = sorted_d[:, 1]
        mask = (r1 > 1e-10) & (r2 > 1e-10)
        if mask.sum() < 2:
            return 0.0
        mu = r2[mask] / r1[mask]
        # MLE of dimension given exponential distribution of mu
        return float(1.0 / np.log(mu).mean())

    @staticmethod
    def _mle(X: np.ndarray, k: int = 5) -> float:
        """Levina-Bickel MLE intrinsic dimension."""
        n = X.shape[0]
        k = min(k, n - 1)
        dists = _pairwise_l2(X)
        np.fill_diagonal(dists, np.inf)
        sorted_d = np.sort(dists, axis=1)[:, :k]
        log_r_k = np.log(sorted_d[:, -1])
        log_r = np.log(sorted_d[:, :-1])
        m = (log_r_k[:, np.newaxis] - log_r).mean(axis=1)
        m = np.where(m > 1e-10, m, np.nan)
        return float(np.nanmean(1.0 / m))


def _pairwise_l2(X: np.ndarray) -> np.ndarray:
    sq = (X ** 2).sum(axis=1, keepdims=True)
    return np.sqrt(np.maximum(sq + sq.T - 2 * X @ X.T, 0.0))
