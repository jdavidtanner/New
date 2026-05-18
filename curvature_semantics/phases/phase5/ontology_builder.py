"""Constructs a toy knowledge world: Entity/Relation/Rule dataclasses, exports to graph."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entity:
    id: str
    name: str
    category: str = "concept"
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    source: str
    relation_type: str
    target: str
    weight: float = 1.0
    is_bridge: bool = False


@dataclass
class Rule:
    antecedents: list[tuple[str, str]]   # list of (relation_type, entity_id)
    consequent_relation: str
    consequent_entity: str
    confidence: float = 1.0


@dataclass
class Ontology:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    bridge_concept_ids: list[str] = field(default_factory=list)
    contradiction_pairs: list[tuple[str, str, str]] = field(default_factory=list)

    def entity_ids(self) -> list[str]:
        return [e.id for e in self.entities]

    def to_triples(self, include_bridge: bool = True) -> list[tuple[str, str, str]]:
        triples = []
        for r in self.relations:
            if not include_bridge and r.source in self.bridge_concept_ids:
                continue
            if not include_bridge and r.target in self.bridge_concept_ids:
                continue
            triples.append((r.source, r.relation_type, r.target))
        return triples

    def to_sentences(self, include_bridge: bool = True) -> list[str]:
        """Convert triples to natural-language sentences for training corpus."""
        templates = {
            "is_a": "{s} is a type of {t}.",
            "has_property": "{s} has the property of {t}.",
            "causes": "{s} causes {t}.",
            "related_to": "{s} is related to {t}.",
            "part_of": "{s} is part of {t}.",
            "implies": "If {s}, then {t}.",
            "contradicts": "{s} contradicts {t}.",
            "bridges": "{s} bridges the gap between concepts and {t}.",
        }
        sentences = []
        for src, rel, tgt in self.to_triples(include_bridge=include_bridge):
            tmpl = templates.get(rel, "{s} {r} {t}.")
            sentences.append(tmpl.format(s=src.replace("_", " "), r=rel.replace("_", " "), t=tgt.replace("_", " ")))
        return sentences


def build_ontology(
    num_entities: int = 50,
    num_relations: int = 8,
    num_rules: int = 20,
    num_bridge_concepts: int = 10,
    num_contradictions: int = 5,
    graph_density: float = 0.15,
    seed: int = 42,
) -> Ontology:
    """Build a synthetic ontology with controlled bridge concepts and contradictions."""
    rng = random.Random(seed)
    categories = ["animal", "plant", "mineral", "concept", "process", "property"]
    relation_types = ["is_a", "has_property", "causes", "related_to", "part_of", "implies", "bridges", "contradicts"]
    relation_types = relation_types[:num_relations]

    entities = [
        Entity(id=f"entity_{i:03d}", name=f"concept_{i}", category=rng.choice(categories))
        for i in range(num_entities)
    ]

    # Select bridge concepts
    bridge_ids = [f"entity_{i:03d}" for i in rng.sample(range(num_entities), min(num_bridge_concepts, num_entities))]
    for e in entities:
        if e.id in bridge_ids:
            e.attributes["is_bridge"] = True

    # Build edges with given density
    relations = []
    max_edges = int(num_entities * (num_entities - 1) * graph_density / 2)
    entity_ids = [e.id for e in entities]
    seen_pairs: set[tuple[str, str]] = set()
    attempts = 0
    while len(relations) < max_edges and attempts < max_edges * 10:
        attempts += 1
        src = rng.choice(entity_ids)
        tgt = rng.choice(entity_ids)
        if src == tgt or (src, tgt) in seen_pairs:
            continue
        seen_pairs.add((src, tgt))
        rel_type = rng.choice(relation_types)
        is_bridge = src in bridge_ids or tgt in bridge_ids
        relations.append(Relation(source=src, relation_type=rel_type, target=tgt, is_bridge=is_bridge))

    # Rules: A ->rel1-> B, B ->rel2-> C implies A ->implied_rel-> C
    rules = []
    for _ in range(num_rules):
        if len(entity_ids) < 3:
            break
        a, b, c = rng.sample(entity_ids, 3)
        rule = Rule(
            antecedents=[(rng.choice(relation_types), b)],
            consequent_relation=rng.choice(relation_types),
            consequent_entity=c,
            confidence=rng.uniform(0.7, 1.0),
        )
        rules.append(rule)

    # Contradictions: pairs of entities that conflict
    contradiction_pairs = []
    for _ in range(num_contradictions):
        a, b = rng.sample(entity_ids, 2)
        contradiction_pairs.append((a, "contradicts", b))
        relations.append(Relation(source=a, relation_type="contradicts", target=b, is_bridge=False))

    return Ontology(
        entities=entities,
        relations=relations,
        rules=rules,
        bridge_concept_ids=bridge_ids,
        contradiction_pairs=contradiction_pairs,
    )
