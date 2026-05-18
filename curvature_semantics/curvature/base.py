"""Abstract base for curvature proxy metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CurvatureProxy(ABC):
    name: str = "base"

    @abstractmethod
    def compute(self, hidden_states: np.ndarray) -> dict[str, float]:
        """Compute curvature proxy.

        Args:
            hidden_states: Array of shape (n_samples, hidden_dim) representing
                           embeddings at a single layer for a batch of inputs.

        Returns:
            Dict mapping metric_name -> scalar value.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
