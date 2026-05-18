"""Baseline: direct generation, no retrieval or routing."""

from __future__ import annotations

from typing import Any, Callable


class VanillaBaseline:
    name = "vanilla"

    def run(self, prompt: str, generate_fn: Callable[[str], str], **kwargs: Any) -> dict[str, Any]:
        response = generate_fn(prompt)
        return {"action": "answer", "response": response, "baseline": self.name}
