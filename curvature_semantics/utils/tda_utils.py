"""TDA helpers: ripser/gudhi wrappers, persistence diagram to feature vector."""

from __future__ import annotations

import numpy as np


def persistence_to_feature_vector(diagrams: list[np.ndarray], n_bins: int = 20) -> np.ndarray:
    """Convert persistence diagrams to a fixed-length feature vector via landscape sampling."""
    features = []
    for dgm in diagrams:
        if dgm.size == 0:
            features.extend([0.0] * n_bins)
            continue
        finite = dgm[dgm[:, 1] < np.inf] if dgm.size > 0 else np.zeros((0, 2))
        if finite.size == 0:
            features.extend([0.0] * n_bins)
            continue
        lifetimes = finite[:, 1] - finite[:, 0]
        # Persistence landscape approximation: histogram of lifetimes
        hist, _ = np.histogram(lifetimes, bins=n_bins, range=(0, lifetimes.max() + 1e-10))
        features.extend(hist.astype(float).tolist())
    return np.array(features)


def wasserstein_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Approximate Wasserstein distance between two persistence diagrams."""
    try:
        from persim import wasserstein
        return float(wasserstein(dgm1, dgm2))
    except ImportError:
        pass
    # Fallback: use L2 between sorted lifetimes
    l1 = np.sort(dgm1[:, 1] - dgm1[:, 0]) if dgm1.size else np.array([])
    l2 = np.sort(dgm2[:, 1] - dgm2[:, 0]) if dgm2.size else np.array([])
    n = max(len(l1), len(l2))
    l1 = np.pad(l1, (0, n - len(l1)))
    l2 = np.pad(l2, (0, n - len(l2)))
    return float(np.linalg.norm(l1 - l2))
