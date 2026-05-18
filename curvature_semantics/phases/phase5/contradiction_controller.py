"""Add/remove controlled contradictions in the ontology at configurable severity."""

from __future__ import annotations

import copy
import random

from curvature_semantics.phases.phase5.ontology_builder import Ontology, Relation


def add_contradictions(
    ontology: Ontology,
    n_additional: int = 5,
    seed: int = 0,
) -> Ontology:
    """Add n_additional contradiction relations to the ontology."""
    new_ont = copy.deepcopy(ontology)
    entity_ids = [e.id for e in new_ont.entities]
    rng = random.Random(seed)
    existing = {(r.source, r.target) for r in new_ont.relations}
    added = 0
    attempts = 0
    while added < n_additional and attempts < n_additional * 20:
        attempts += 1
        a, b = rng.sample(entity_ids, 2)
        if (a, b) in existing:
            continue
        new_ont.relations.append(Relation(source=a, relation_type="contradicts", target=b))
        new_ont.contradiction_pairs.append((a, "contradicts", b))
        existing.add((a, b))
        added += 1
    return new_ont


def remove_contradictions_from_ontology(ontology: Ontology) -> Ontology:
    """Strip all contradiction relations from the ontology."""
    new_ont = copy.deepcopy(ontology)
    new_ont.relations = [r for r in new_ont.relations if r.relation_type != "contradicts"]
    new_ont.contradiction_pairs = []
    return new_ont
