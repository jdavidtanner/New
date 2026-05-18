"""Routing policy: maps curvature/completeness signals to actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RoutingAction(str, Enum):
    ANSWER = "answer"
    RETRIEVE = "retrieve"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


@dataclass
class RoutingSignal:
    curvature: float
    completeness: float
    entropy: float = 0.0
    confidence: float = 1.0
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ThresholdPolicy:
    """Threshold-based routing: curvature → action."""

    def __init__(
        self,
        low: float = 0.3,
        medium: float = 0.6,
        high: float = 0.85,
    ):
        self.low = low
        self.medium = medium
        self.high = high

    def decide(self, signal: RoutingSignal) -> RoutingAction:
        c = signal.curvature
        if c <= self.low:
            return RoutingAction.ANSWER
        elif c <= self.medium:
            return RoutingAction.RETRIEVE
        elif c <= self.high:
            return RoutingAction.CLARIFY
        else:
            return RoutingAction.ABSTAIN


class LearnedPolicy:
    """Logistic-regression routing policy trained on (signal, action) pairs."""

    def __init__(self):
        self._clf = None

    def fit(self, signals: list[RoutingSignal], actions: list[RoutingAction]) -> "LearnedPolicy":
        try:
            from sklearn.linear_model import LogisticRegression
            import numpy as np
            X = np.array([[s.curvature, s.completeness, s.entropy, s.confidence] for s in signals])
            y = [a.value for a in actions]
            self._clf = LogisticRegression(max_iter=500).fit(X, y)
        except ImportError:
            pass
        return self

    def decide(self, signal: RoutingSignal) -> RoutingAction:
        if self._clf is None:
            return ThresholdPolicy().decide(signal)
        import numpy as np
        X = np.array([[signal.curvature, signal.completeness, signal.entropy, signal.confidence]])
        pred = self._clf.predict(X)[0]
        return RoutingAction(pred)
