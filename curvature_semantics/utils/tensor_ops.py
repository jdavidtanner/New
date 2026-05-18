"""Batch tensor operations: cosine similarity, pairwise distance, kNN, PCA."""

from __future__ import annotations

import numpy as np
import torch


def pairwise_cosine(X: np.ndarray) -> np.ndarray:
    """(n, d) -> (n, n) cosine similarity matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.maximum(norms, 1e-10)
    return X_norm @ X_norm.T


def pairwise_l2(X: np.ndarray) -> np.ndarray:
    """(n, d) -> (n, n) Euclidean distance matrix."""
    sq = (X ** 2).sum(axis=1, keepdims=True)
    return np.sqrt(np.maximum(sq + sq.T - 2 * X @ X.T, 0.0))


def knn_indices(X: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    """Return (n, k) array of kNN indices (excluding self)."""
    if metric == "cosine":
        dists = 1.0 - pairwise_cosine(X)
    else:
        dists = pairwise_l2(X)
    np.fill_diagonal(dists, np.inf)
    return np.argsort(dists, axis=1)[:, :k]


def pca_reduce(X: np.ndarray, n_components: int = 50) -> np.ndarray:
    """PCA dimensionality reduction. Returns (n, n_components)."""
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    mean = X.mean(0)
    centered = X - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:n_components].T


def batch_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between corresponding rows of a and b."""
    a_norm = torch.nn.functional.normalize(a, dim=-1)
    b_norm = torch.nn.functional.normalize(b, dim=-1)
    return (a_norm * b_norm).sum(dim=-1)


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy from logits. (batch, seq, vocab) -> (batch, seq)."""
    probs = torch.softmax(logits, dim=-1)
    return -(probs * (probs + 1e-12).log()).sum(dim=-1)
