"""Embedding-space perturbations applied directly to hidden states."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from curvature_semantics.perturbations.base import EmbeddingPerturbation


class GaussianNoise(EmbeddingPerturbation):
    name = "gaussian_noise"

    def __init__(self, std: float = 0.01):
        self.std = std

    def apply(self, hidden: torch.Tensor, rng: np.random.Generator | None = None, **kwargs: Any) -> torch.Tensor:
        noise = torch.randn_like(hidden) * self.std
        return hidden + noise


class PCADirection(EmbeddingPerturbation):
    """Perturb along the top-k PCA direction of the hidden states."""
    name = "pca_direction"

    def __init__(self, scale: float = 0.05, n_components: int = 1):
        self.scale = scale
        self.n_components = n_components

    def apply(self, hidden: torch.Tensor, rng: np.random.Generator | None = None, **kwargs: Any) -> torch.Tensor:
        orig_shape = hidden.shape
        flat = hidden.reshape(-1, orig_shape[-1]).float()
        mean = flat.mean(0, keepdim=True)
        centered = flat - mean
        try:
            _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
            direction = Vt[0]  # top PC
            rng = rng or np.random.default_rng()
            sign = float(rng.choice([-1, 1]))
            perturbation = sign * self.scale * direction
            perturbed = (flat + perturbation).reshape(orig_shape)
            return perturbed.to(hidden.dtype)
        except Exception:
            return hidden


class ManifoldDirection(EmbeddingPerturbation):
    """Perturb along low-density manifold directions (approx via kNN tangent space)."""
    name = "manifold_direction"

    def __init__(self, scale: float = 0.03, k: int = 5):
        self.scale = scale
        self.k = k

    def apply(self, hidden: torch.Tensor, rng: np.random.Generator | None = None, **kwargs: Any) -> torch.Tensor:
        orig_shape = hidden.shape
        flat = hidden.reshape(-1, orig_shape[-1]).float()
        n = flat.shape[0]
        if n <= self.k:
            return hidden

        # Build kNN and take random tangent direction
        dists = torch.cdist(flat, flat)
        _, nn_idx = dists.topk(self.k + 1, largest=False)
        nn_idx = nn_idx[:, 1:]  # exclude self

        rng = rng or np.random.default_rng()
        query_idx = int(rng.integers(0, n))
        neighbors = flat[nn_idx[query_idx]]
        tangent = neighbors.mean(0) - flat[query_idx]
        norm = tangent.norm()
        if norm < 1e-8:
            return hidden
        tangent = tangent / norm * self.scale
        perturbed = flat + tangent.unsqueeze(0)
        return perturbed.reshape(orig_shape).to(hidden.dtype)


class HiddenStateInterpolation(EmbeddingPerturbation):
    """Interpolate between two hidden states (requires pairs)."""
    name = "hidden_state_interpolation"

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def apply(self, hidden: torch.Tensor, rng: np.random.Generator | None = None, target: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        if target is None or target.shape != hidden.shape:
            return hidden
        return (1 - self.alpha) * hidden + self.alpha * target


class TangentSpaceProjection(EmbeddingPerturbation):
    """Project hidden states onto their local tangent space (removes normal component)."""
    name = "tangent_space_projection"

    def apply(self, hidden: torch.Tensor, rng: np.random.Generator | None = None, **kwargs: Any) -> torch.Tensor:
        orig_shape = hidden.shape
        flat = hidden.reshape(-1, orig_shape[-1]).float()
        mean = flat.mean(0, keepdim=True)
        centered = flat - mean
        try:
            _, S, Vt = torch.linalg.svd(centered, full_matrices=False)
            # Keep only tangent space directions (all but the last few)
            k = max(1, Vt.shape[0] - 2)
            tangent_basis = Vt[:k]
            projected = (centered @ tangent_basis.T) @ tangent_basis + mean
            return projected.reshape(orig_shape).to(hidden.dtype)
        except Exception:
            return hidden
