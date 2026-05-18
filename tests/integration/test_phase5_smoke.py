"""Smoke test: build ontology, train for minimal steps, probe."""

from __future__ import annotations

import pytest

from curvature_semantics.phases.phase5.ontology_builder import build_ontology
from curvature_semantics.phases.phase5.bridge_concept_injector import remove_bridge_concepts, restore_bridge_concepts
from curvature_semantics.phases.phase5.contradiction_controller import add_contradictions, remove_contradictions_from_ontology
from curvature_semantics.phases.phase5.synthetic_probing import build_bridge_probes, build_non_bridge_probes


def test_build_ontology_structure():
    ont = build_ontology(num_entities=20, num_relations=4, num_bridge_concepts=4, num_contradictions=2, seed=0)
    assert len(ont.entities) == 20
    assert len(ont.relations) > 0
    assert len(ont.bridge_concept_ids) == 4
    assert len(ont.contradiction_pairs) == 2


def test_ontology_to_sentences():
    ont = build_ontology(num_entities=10, seed=1)
    sentences = ont.to_sentences(include_bridge=True)
    assert len(sentences) > 0
    assert all(isinstance(s, str) for s in sentences)


def test_ontology_to_sentences_without_bridges():
    ont = build_ontology(num_entities=20, num_bridge_concepts=5, seed=2)
    with_bridges = ont.to_sentences(include_bridge=True)
    without_bridges = ont.to_sentences(include_bridge=False)
    assert len(with_bridges) >= len(without_bridges)


def test_remove_bridge_concepts():
    ont = build_ontology(num_entities=20, num_bridge_concepts=5, seed=3)
    partial = remove_bridge_concepts(ont)
    bridge_ids = set(ont.bridge_concept_ids)
    for e in partial.entities:
        assert e.id not in bridge_ids
    for r in partial.relations:
        assert r.source not in bridge_ids
        assert r.target not in bridge_ids


def test_restore_bridge_concepts():
    ont = build_ontology(num_entities=20, num_bridge_concepts=5, seed=4)
    partial = remove_bridge_concepts(ont)
    restored = restore_bridge_concepts(ont, partial)
    assert len(restored.bridge_concept_ids) == len(ont.bridge_concept_ids)


def test_add_remove_contradictions():
    ont = build_ontology(num_entities=20, num_contradictions=0, seed=5)
    assert len(ont.contradiction_pairs) == 0
    with_c = add_contradictions(ont, n_additional=3, seed=0)
    assert len(with_c.contradiction_pairs) == 3
    without_c = remove_contradictions_from_ontology(with_c)
    assert len(without_c.contradiction_pairs) == 0


def test_build_bridge_probes():
    ont = build_ontology(num_entities=30, num_bridge_concepts=5, seed=6)
    probes = build_bridge_probes(ont, n_per_bridge=3)
    assert len(probes) > 0
    for p in probes:
        assert "prompt" in p
        assert "expected" in p
        assert p["requires_bridge"] is True


def test_build_non_bridge_probes():
    ont = build_ontology(num_entities=30, num_bridge_concepts=3, seed=7)
    probes = build_non_bridge_probes(ont, n=10)
    assert len(probes) > 0
    for p in probes:
        assert p["requires_bridge"] is False
        assert p["bridge_concept"] is None
