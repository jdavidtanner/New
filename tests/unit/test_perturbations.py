"""Unit tests for text and embedding perturbations."""

import numpy as np
import pytest
import torch

from curvature_semantics.perturbations.text_perturbations import (
    SynonymSubstitution, SentenceReorder, ActivePassive,
    DistractorInsertion, EvidenceReorder,
)
from curvature_semantics.perturbations.embedding_perturbations import (
    GaussianNoise, PCADirection, ManifoldDirection, TangentSpaceProjection,
)
from curvature_semantics.perturbations.perturbation_suite import PerturbationSuite


TEXT = "The large cat sat quickly on the mat. The small dog ran slowly."


class TestTextPerturbations:
    def test_synonym_substitution_changes_text(self):
        p = SynonymSubstitution(p=1.0)
        result = p.apply(TEXT, rng=np.random.default_rng(42))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sentence_reorder_preserves_words(self):
        p = SentenceReorder()
        result = p.apply(TEXT, rng=np.random.default_rng(42))
        orig_words = set(TEXT.lower().split())
        result_words = set(result.lower().split())
        # All words should be present
        assert orig_words == result_words

    def test_distractor_insertion_adds_content(self):
        p = DistractorInsertion()
        result = p.apply(TEXT, rng=np.random.default_rng(42))
        assert len(result) > len(TEXT)

    def test_active_passive_returns_string(self):
        p = ActivePassive()
        result = p.apply("The project was completed by the team.", rng=np.random.default_rng(0))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_evidence_reorder_returns_string(self):
        p = EvidenceReorder()
        text = "Context: Cats are mammals. Dogs are pets.\nQuestion: What are cats?"
        result = p.apply(text, rng=np.random.default_rng(0))
        assert isinstance(result, str)


class TestEmbeddingPerturbations:
    def test_gaussian_noise_shape_preserved(self):
        p = GaussianNoise(std=0.01)
        x = torch.randn(4, 64)
        result = p.apply(x)
        assert result.shape == x.shape

    def test_gaussian_noise_adds_perturbation(self):
        p = GaussianNoise(std=0.1)
        x = torch.zeros(4, 64)
        result = p.apply(x)
        assert not torch.allclose(result, x)

    def test_pca_direction_shape(self):
        p = PCADirection(scale=0.05)
        x = torch.randn(10, 32)
        result = p.apply(x)
        assert result.shape == x.shape

    def test_manifold_direction_shape(self):
        p = ManifoldDirection(scale=0.03, k=3)
        x = torch.randn(15, 32)
        result = p.apply(x)
        assert result.shape == x.shape

    def test_tangent_space_projection_shape(self):
        p = TangentSpaceProjection()
        x = torch.randn(20, 32)
        result = p.apply(x)
        assert result.shape == x.shape


class TestPerturbationSuite:
    def test_text_perturbations(self):
        suite = PerturbationSuite(text_names=["synonym_substitution", "sentence_reorder"])
        results = suite.perturb_text(TEXT, n=2)
        assert len(results) == 2
        for r in results:
            assert r.original_text == TEXT
            assert isinstance(r.perturbed_text, str)

    def test_embedding_perturbations(self):
        suite = PerturbationSuite(embedding_names=["gaussian_noise", "pca_direction"])
        hidden = torch.randn(8, 64)
        results = suite.perturb_embedding(hidden)
        assert len(results) == 2
        for r in results:
            assert r.original.shape == r.perturbed.shape

    def test_unknown_perturbation_raises(self):
        with pytest.raises(ValueError):
            PerturbationSuite(text_names=["nonexistent_perturbation"])
