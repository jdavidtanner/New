"""Baseline: self-consistency voting across N generations."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable


class SelfConsistencyBaseline:
    name = "self_consistency"

    def __init__(self, n_samples: int = 5, temperature: float = 0.7):
        self.n_samples = n_samples
        self.temperature = temperature

    def run(
        self,
        prompt: str,
        generate_fn: Callable[[str], str],
        generate_n_fn: Callable[[str, int], list[str]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if generate_n_fn is not None:
            responses = generate_n_fn(prompt, self.n_samples)
        else:
            responses = [generate_fn(prompt) for _ in range(self.n_samples)]

        if not responses:
            return {"action": "answer", "response": "", "baseline": self.name}

        # Majority vote by exact string (simplified — in practice use semantic clustering)
        counts = Counter(responses)
        majority = counts.most_common(1)[0][0]
        consistency_score = counts.most_common(1)[0][1] / len(responses)
        return {
            "action": "answer",
            "response": majority,
            "baseline": self.name,
            "consistency_score": consistency_score,
            "all_responses": responses,
        }
