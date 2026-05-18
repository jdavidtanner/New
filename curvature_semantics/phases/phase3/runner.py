"""Phase 3 orchestrator: iterates all transformer layers, stores per-layer geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from curvature_semantics.core.config_manager import ExperimentConfig
from curvature_semantics.core.artifact_store import ArtifactStore
from curvature_semantics.core.dataset_loader import load_domain_examples
from curvature_semantics.core.logging_utils import get_logger
from curvature_semantics.core.model_loader import load_model_and_tokenizer, get_num_layers
from curvature_semantics.core.hidden_state_extractor import HiddenStateExtractor
from curvature_semantics.curvature.curvature_aggregator import CurvatureAggregator
from curvature_semantics.semantics.completeness_aggregator import CompletenessAggregator
from curvature_semantics.phases.phase3.layer_profiler import build_layer_profile, compute_region_statistics
from curvature_semantics.visualization.layerwise_geometry import plot_layerwise_curvature

logger = get_logger(__name__)


def run(cfg: ExperimentConfig) -> dict[str, Any]:
    store = ArtifactStore.from_config(cfg)
    store.save_config(cfg.to_dict())

    model_spec = cfg.model
    model, tokenizer = load_model_and_tokenizer(
        model_spec.id, device=cfg.device, dtype=cfg.dtype,
        load_in_8bit=model_spec.load_in_8bit, load_in_4bit=model_spec.load_in_4bit,
    )
    n_layers = get_num_layers(model)
    logger.info("Model has %d transformer layers", n_layers)

    layer_cfg = cfg.raw.get("layer_schedule", {})
    mode = layer_cfg.get("mode", "all")
    if mode == "all":
        layer_indices = list(range(n_layers))
    elif mode == "strided":
        stride = layer_cfg.get("stride", 2)
        layer_indices = list(range(0, n_layers, stride))
    else:
        layer_indices = [i % n_layers for i in layer_cfg.get("indices", list(range(n_layers)))]

    extract_logits = cfg.raw.get("layer_schedule", {}).get("extract_logits", True)
    extractor = HiddenStateExtractor(model, layer_indices=layer_indices, extract_logits=extract_logits)
    curv_agg = CurvatureAggregator.from_config(cfg)
    sem_agg = CompletenessAggregator.from_config(cfg)

    domain_cfg = cfg.raw.get("domains", {"low_curvature": ["arithmetic"], "high_curvature": ["speculative_medicine"]})
    all_domains = [d for domains in domain_cfg.values() for d in domains] if isinstance(domain_cfg, dict) else domain_cfg

    n_prompts = cfg.raw.get("prompts_per_domain", 20)
    all_rows: list[dict[str, Any]] = []

    for domain in all_domains:
        examples = load_domain_examples(domain, n=n_prompts, seed=cfg.seed)
        for ex in examples:
            prompt, context, reference = ex["prompt"], ex.get("context", ""), ex.get("answer", "")
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
            bundle = extractor.extract(inputs["input_ids"], inputs.get("attention_mask"))
            hs_array = bundle.last_token_array()  # (n_layers, 1, hidden_dim)

            with torch.no_grad():
                gen_ids = model.generate(
                    inputs["input_ids"].to(cfg.device),
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=None,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            sem_bundle = sem_agg.score(prompt, response, context=context, reference=reference, domain=domain)

            for li, layer_idx in enumerate(bundle.layer_indices):
                hs = hs_array[li, 0, :][np.newaxis, :]  # (1, hidden_dim)
                curv_bundle = curv_agg.compute_layer(hs, layer_idx=layer_idx)
                row: dict[str, Any] = {
                    "domain": domain,
                    "layer_idx": layer_idx,
                    **curv_bundle.metrics,
                    **sem_bundle.metrics,
                    "model_alias": model_spec.alias,
                }
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    store.save_df("layerwise_features", df)

    layer_region_cfg = cfg.raw.get("layer_regions", {})
    profile = build_layer_profile(
        n_layers,
        early_fraction=layer_region_cfg.get("early_fraction", 0.25),
        middle_fraction=layer_region_cfg.get("middle_fraction", 0.50),
        late_fraction=layer_region_cfg.get("late_fraction", 0.25),
    )
    region_stats = compute_region_statistics(df, profile)
    store.save_json("region_statistics", region_stats)

    figures_dir = store.path("figures")
    plot_layerwise_curvature(
        df, curvature_cols=["trajectory_divergence", "intrinsic_dimension", "neighborhood_distortion"],
        output_path=figures_dir / "layerwise_curvature",
        title=f"Layerwise Curvature — {model_spec.alias}",
    )

    hypothesis_region = cfg.raw.get("hypothesis", {}).get("obstruction_region", "middle")
    middle_curv = region_stats.get("middle", {}).get("trajectory_divergence_mean", 0.0) or 0.0
    late_curv = region_stats.get("late", {}).get("trajectory_divergence_mean", 0.0) or 0.0
    hypothesis_supported = middle_curv > late_curv if middle_curv and late_curv else None

    logger.info(
        "Phase 3 complete. Middle curv=%.4f, Late curv=%.4f, Hypothesis supported=%s",
        middle_curv, late_curv, hypothesis_supported,
    )
    return {
        "n_rows": len(all_rows),
        "region_statistics": region_stats,
        "hypothesis_supported": hypothesis_supported,
        "output_dir": str(store.phase_dir),
    }
