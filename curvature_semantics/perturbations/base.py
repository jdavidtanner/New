"""Abstract base for all perturbation types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class TextPerturbation(ABC):
    name: str = "base_text"

    @abstractmethod
    def apply(self, text: str, rng: np.random.Generator | None = None) -> str:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class EmbeddingPerturbation(ABC):
    name: str = "base_embedding"

    @abstractmethod
    def apply(
        self,
        hidden: torch.Tensor,
        rng: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply perturbation to hidden states.

        Args:
            hidden: Tensor of shape (batch, seq_len, hidden_dim) or (batch, hidden_dim)
        Returns:
            Perturbed tensor of same shape.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
