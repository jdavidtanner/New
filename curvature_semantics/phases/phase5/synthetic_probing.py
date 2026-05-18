"""Probes trained toy models at bridge-concept boundaries; extracts curvature signatures."""

from __future__ import annotations

from typing import Any

import numpy as np

from curvature_semantics.core.hidden_state_extractor import HiddenStateExtractor
from curvature_semantics.curvature.curvature_aggregator import CurvatureAggregator
from curvature_semantics.semantics.completeness_aggregator import CompletenessAggregator
from curvature_semantics.phases.phase5.ontology_builder import Ontology
from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)


def build_bridge_probes(ontology: Ontology, n_per_bridge: int = 20) -> list[dict[str, Any]]:
    """Generate probing questions that span bridge concepts."""
    probes = []
    bridge_ids = set(ontology.bridge_concept_ids)
    bridge_relations = [r for r in ontology.relations if r.source in bridge_ids or r.target in bridge_ids]

    import random
    rng = random.Random(42)
    sampled = rng.sample(bridge_relations, min(n_per_bridge * len(ontology.bridge_concept_ids), len(bridge_relations)))

    for rel in sampled:
        prompt = f"What is the relationship between {rel.source.replace('_', ' ')} and {rel.target.replace('_', ' ')}?"
        expected = f"{rel.source.replace('_', ' ')} {rel.relation_type.replace('_', ' ')} {rel.target.replace('_', ' ')}."
        probes.append({
            "prompt": prompt,
            "expected": expected,
            "bridge_concept": rel.source if rel.source in bridge_ids else rel.target,
            "relation_type": rel.relation_type,
            "requires_bridge": True,
        })
    return probes


def build_non_bridge_probes(ontology: Ontology, n: int = 50) -> list[dict[str, Any]]:
    """Generate control probes that don't span bridge concepts."""
    bridge_ids = set(ontology.bridge_concept_ids)
    non_bridge = [r for r in ontology.relations if r.source not in bridge_ids and r.target not in bridge_ids]
    import random
    rng = random.Random(0)
    sampled = rng.sample(non_bridge, min(n, len(non_bridge)))
    return [
        {
            "prompt": f"What is the relationship between {r.source.replace('_', ' ')} and {r.target.replace('_', ' ')}?",
            "expected": f"{r.source.replace('_', ' ')} {r.relation_type.replace('_', ' ')} {r.target.replace('_', ' ')}.",
            "bridge_concept": None,
            "relation_type": r.relation_type,
            "requires_bridge": False,
        }
        for r in sampled
    ]


def probe_model(
    model: Any,
    tokenizer: Any,
    probes: list[dict[str, Any]],
    curv_agg: CurvatureAggregator,
    sem_agg: CompletenessAggregator,
    device: str = "cpu",
    max_new_tokens: int = 64,
) -> list[dict[str, Any]]:
    """Run probes through the model and collect curvature + completeness signals."""
    import torch
    extractor = HiddenStateExtractor(model, extract_logits=True)
    results = []

    for probe in probes:
        prompt = probe["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        bundle = extractor.extract(inputs["input_ids"].to(device), inputs.get("attention_mask", {}).get("input_ids"))
        hs_array = bundle.last_token_array()  # (n_layers, 1, hidden_dim)

        with torch.no_grad():
            gen_ids = model.generate(
                inputs["input_ids"].to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        mid = hs_array.shape[0] // 2
        hs = hs_array[mid, :, :]  # (1, hidden_dim)
        curv = curv_agg.compute_layer(hs, layer_idx=mid)
        sem = sem_agg.score(prompt, response, reference=probe["expected"])

        results.append({
            **probe,
            "response": response,
            **curv.metrics,
            **sem.metrics,
        })
    return results
