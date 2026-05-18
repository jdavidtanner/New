"""CurvatureAwareRouter: maps signals to routing actions and executes them."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from curvature_semantics.phases.phase6.routing_policy import RoutingAction, RoutingSignal, ThresholdPolicy
from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)


class CurvatureAwareRouter:
    """Routes queries based on live curvature/completeness estimates.

    For inference-time use (no full hidden-state extraction), it uses
    lightweight entropy-based proxies from output logits.
    """

    def __init__(
        self,
        policy: ThresholdPolicy | None = None,
        retriever: Any = None,
        curvature_proxy: str = "trajectory_divergence",
    ):
        self.policy = policy or ThresholdPolicy()
        self.retriever = retriever
        self.curvature_proxy = curvature_proxy

    def estimate_signal(
        self,
        logits: np.ndarray | None,
        hidden_state: np.ndarray | None = None,
        curvature_bundle: Any = None,
        completeness_bundle: Any = None,
    ) -> RoutingSignal:
        """Estimate routing signal from available signals."""
        curvature = 0.0
        completeness = 0.5
        entropy = 0.0
        confidence = 1.0

        if curvature_bundle is not None:
            curvature = curvature_bundle.metrics.get(self.curvature_proxy, 0.0)

        if completeness_bundle is not None:
            completeness = completeness_bundle.primary()

        if logits is not None:
            probs = _softmax(logits.astype(np.float64))
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
            confidence = float(probs.max())
            # Use entropy as lightweight curvature proxy when no hidden-state curvature
            if curvature_bundle is None:
                curvature = min(entropy / 10.0, 1.0)

        return RoutingSignal(
            curvature=float(curvature),
            completeness=float(completeness),
            entropy=float(entropy),
            confidence=float(confidence),
        )

    def route(
        self,
        prompt: str,
        signal: RoutingSignal,
        generate_fn: Callable[[str], str],
        retrieve_fn: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
        """Execute the routing decision and return result."""
        action = self.policy.decide(signal)

        if action == RoutingAction.ANSWER:
            response = generate_fn(prompt)
            return {"action": action.value, "response": response, "signal": signal}

        elif action == RoutingAction.RETRIEVE:
            context = retrieve_fn(prompt) if retrieve_fn else ""
            augmented_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
            response = generate_fn(augmented_prompt)
            return {"action": action.value, "response": response, "signal": signal, "context": context}

        elif action == RoutingAction.CLARIFY:
            response = f"I need more information to answer this accurately. Could you clarify: {prompt}"
            return {"action": action.value, "response": response, "signal": signal}

        else:  # ABSTAIN
            response = "I don't have sufficient reliable information to answer this question."
            return {"action": action.value, "response": response, "signal": signal}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
