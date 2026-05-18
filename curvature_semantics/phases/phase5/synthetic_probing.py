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
    """Run probes through the model and collect curvature + completeness signals.

    Curvature is computed over the full probe batch (not per-example) so that
    proximity-based proxies (trajectory divergence, intrinsic dimension) have
    enough samples to produce meaningful estimates.
    """
    import torch
    extractor = HiddenStateExtractor(model, extract_logits=True)

    # --- Pass 1: collect hidden states and generate responses for all probes ---
    all_hs: list[np.ndarray] = []   # list of (n_layers, hidden_dim) per probe
    responses: list[str] = []

    for probe in probes:
        prompt = probe["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        attn_mask = inputs.get("attention_mask")
        bundle = extractor.extract(
            inputs["input_ids"].to(device),
            attn_mask.to(device) if attn_mask is not None else None,
        )
        # last_token_array: (n_layers, batch=1, hidden_dim) → (n_layers, hidden_dim)
        hs = bundle.last_token_array()[:, 0, :]
        all_hs.append(hs)

        with torch.no_grad():
            gen_ids = model.generate(
                inputs["input_ids"].to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append(response)

    if not all_hs:
        return []

    # --- Pass 2: compute curvature over the full batch at the middle layer ---
    n_layers = all_hs[0].shape[0]
    mid = n_layers // 2
    # Stack across probes: (n_probes, hidden_dim)
    batch_hs = np.stack([hs[mid] for hs in all_hs], axis=0)
    curv_bundle = curv_agg.compute_layer(batch_hs, layer_idx=mid)
    logger.info(
        "Curvature (layer %d, n=%d): traj_div=%.4f  ID=%.4f",
        mid, len(probes),
        curv_bundle.metrics.get("trajectory_divergence", 0.0),
        curv_bundle.metrics.get("intrinsic_dimension", 0.0),
    )

    # --- Pass 3: per-probe semantic completeness + assemble results ---
    results: list[dict[str, Any]] = []
    for probe, response in zip(probes, responses):
        sem = sem_agg.score(probe["prompt"], response, reference=probe["expected"])
        results.append({
            **probe,
            "response": response,
            **curv_bundle.metrics,   # same batch-level curvature for all probes in this group
            **sem.metrics,
        })
    return results
