"""Composes multiple perturbations and tracks metadata alongside outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from curvature_semantics.perturbations.base import EmbeddingPerturbation, TextPerturbation
from curvature_semantics.perturbations.text_perturbations import (
    ActivePassive,
    DistractorInsertion,
    EvidenceReorder,
    SentenceReorder,
    SynonymSubstitution,
)
from curvature_semantics.perturbations.embedding_perturbations import (
    GaussianNoise,
    HiddenStateInterpolation,
    ManifoldDirection,
    PCADirection,
    TangentSpaceProjection,
)

_TEXT_REGISTRY: dict[str, type[TextPerturbation]] = {
    "synonym_substitution": SynonymSubstitution,
    "sentence_reorder": SentenceReorder,
    "active_passive": ActivePassive,
    "distractor_insertion": DistractorInsertion,
    "evidence_reorder": EvidenceReorder,
}

_EMBEDDING_REGISTRY: dict[str, type[EmbeddingPerturbation]] = {
    "gaussian_noise": GaussianNoise,
    "pca_direction": PCADirection,
    "manifold_direction": ManifoldDirection,
    "hidden_state_interpolation": HiddenStateInterpolation,
    "tangent_space_projection": TangentSpaceProjection,
}


@dataclass
class PerturbedSample:
    original_text: str
    perturbed_text: str
    perturbation_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerturbedEmbedding:
    original: torch.Tensor
    perturbed: torch.Tensor
    perturbation_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PerturbationSuite:
    """Manages a collection of text and embedding perturbations."""

    def __init__(
        self,
        text_names: list[str] | None = None,
        embedding_names: list[str] | None = None,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.text_perturbers: list[TextPerturbation] = []
        self.embedding_perturbers: list[EmbeddingPerturbation] = []

        for name in text_names or []:
            cls = _TEXT_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"Unknown text perturbation: {name!r}")
            self.text_perturbers.append(cls())

        for name in embedding_names or []:
            cls = _EMBEDDING_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"Unknown embedding perturbation: {name!r}")
            self.embedding_perturbers.append(cls())

    @classmethod
    def from_config(cls, cfg: Any) -> "PerturbationSuite":
        pert_cfg = cfg.raw.get("perturbations", {})
        return cls(
            text_names=pert_cfg.get("text", []),
            embedding_names=pert_cfg.get("embedding", []),
            seed=cfg.seed,
        )

    def perturb_text(self, text: str, n: int = 1) -> list[PerturbedSample]:
        results: list[PerturbedSample] = []
        perturbers = self.text_perturbers or list(_TEXT_REGISTRY.values())[:n]
        for p in perturbers[:n]:
            perturbed = p.apply(text, rng=self.rng)
            results.append(PerturbedSample(
                original_text=text,
                perturbed_text=perturbed,
                perturbation_name=p.name,
            ))
        return results

    def perturb_embedding(
        self,
        hidden: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> list[PerturbedEmbedding]:
        results: list[PerturbedEmbedding] = []
        for p in self.embedding_perturbers:
            kwargs: dict[str, Any] = {}
            if isinstance(p, HiddenStateInterpolation) and target is not None:
                kwargs["target"] = target
            perturbed = p.apply(hidden, rng=self.rng, **kwargs)
            results.append(PerturbedEmbedding(
                original=hidden,
                perturbed=perturbed,
                perturbation_name=p.name,
            ))
        return results
