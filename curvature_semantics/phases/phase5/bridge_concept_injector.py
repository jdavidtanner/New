"""Programmatically insert/remove bridge concepts; track which are absent."""

from __future__ import annotations

import copy

from curvature_semantics.phases.phase5.ontology_builder import Ontology, Entity, Relation


def remove_bridge_concepts(ontology: Ontology, bridge_ids: list[str] | None = None) -> Ontology:
    """Return a new ontology with bridge concepts removed."""
    to_remove = set(bridge_ids or ontology.bridge_concept_ids)
    new_ont = copy.deepcopy(ontology)
    new_ont.entities = [e for e in new_ont.entities if e.id not in to_remove]
    new_ont.relations = [r for r in new_ont.relations if r.source not in to_remove and r.target not in to_remove]
    new_ont.bridge_concept_ids = [b for b in new_ont.bridge_concept_ids if b not in to_remove]
    return new_ont


def restore_bridge_concepts(ontology_full: Ontology, ontology_partial: Ontology) -> Ontology:
    """Restore bridge concepts from the full ontology into the partial one."""
    new_ont = copy.deepcopy(ontology_partial)
    existing_ids = {e.id for e in new_ont.entities}
    bridge_ids = set(ontology_full.bridge_concept_ids)

    for entity in ontology_full.entities:
        if entity.id in bridge_ids and entity.id not in existing_ids:
            new_ont.entities.append(copy.deepcopy(entity))

    existing_pairs = {(r.source, r.relation_type, r.target) for r in new_ont.relations}
    for rel in ontology_full.relations:
        key = (rel.source, rel.relation_type, rel.target)
        if key not in existing_pairs and rel.is_bridge:
            new_ont.relations.append(copy.deepcopy(rel))

    new_ont.bridge_concept_ids = list(ontology_full.bridge_concept_ids)
    return new_ont
