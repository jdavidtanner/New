"""Abstract base for semantic completeness metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SemanticMetric(ABC):
    name: str = "base"

    @abstractmethod
    def score(
        self,
        prompt: str,
        response: str,
        context: str = "",
        reference: str = "",
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute metric score.

        Returns:
            Dict mapping metric_name -> scalar in [0, 1] (higher = better completeness).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
