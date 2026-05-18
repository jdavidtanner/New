"""Baseline: standard RAG — always retrieve, always answer."""

from __future__ import annotations

from typing import Any, Callable


class RAGOnlyBaseline:
    name = "rag_only"

    def run(
        self,
        prompt: str,
        generate_fn: Callable[[str], str],
        retrieve_fn: Callable[[str], str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        context = retrieve_fn(prompt) if retrieve_fn else ""
        augmented = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
        response = generate_fn(augmented)
        return {"action": "answer", "response": response, "baseline": self.name, "context": context}
