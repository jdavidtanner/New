"""Phase 1 orchestrator: extract states → perturb → curvature+completeness → correlate → save."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from curvature_semantics.core.artifact_store import ArtifactStore
from curvature_semantics.core.config_manager import ExperimentConfig
from curvature_semantics.core.dataset_loader import load_domain_examples
from curvature_semantics.core.hidden_state_extractor import HiddenStateExtractor
from curvature_semantics.core.logging_utils import get_logger, log_config
from curvature_semantics.core.model_loader import load_model_and_tokenizer
from curvature_semantics.curvature.curvature_aggregator import CurvatureAggregator
from curvature_semantics.curvature.trajectory_divergence import LocalTrajectoryDivergence
from curvature_semantics.perturbations.perturbation_suite import PerturbationSuite
from curvature_semantics.phases.phase1.correlation_analysis import (
    compute_correlation_matrix,
    compute_partial_correlations,
    summarise_findings,
)
from curvature_semantics.semantics.completeness_aggregator import CompletenessAggregator
from curvature_semantics.visualization.layerwise_geometry import plot_layerwise_curvature
from curvature_semantics.visualization.phase_space import plot_phase_space

logger = get_logger(__name__)


def _generate(model: Any, tokenizer: Any, inputs: dict[str, Any], device: torch.device, cfg: ExperimentConfig) -> str:
    with torch.no_grad():
        gen_ids = model.generate(
            inputs["input_ids"].to(device),
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature if cfg.temperature > 0 else None,
            do_sample=cfg.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _extract_last_token(
    extractor: HiddenStateExtractor,
    tokenizer: Any,
    prompt: str,
) -> tuple[dict[str, Any], np.ndarray]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
    bundle = extractor.extract(inputs["input_ids"], inputs.get("attention_mask"))
    return inputs, bundle.last_token_array()[:, 0, :]


def run(cfg: ExperimentConfig) -> dict[str, Any]:
    store = ArtifactStore.from_config(cfg)
    store.save_config(cfg.to_dict())
    log_config(logger, cfg.to_dict())

    model_spec = cfg.model
    model, tokenizer = load_model_and_tokenizer(
        model_spec.id,
        device=cfg.device,
        dtype=cfg.dtype,
        load_in_8bit=model_spec.load_in_8bit,
        load_in_4bit=model_spec.load_in_4bit,
    )
    device = next(model.parameters()).device
    extractor = HiddenStateExtractor(model, extract_logits=True)
    perturb_suite = PerturbationSuite.from_config(cfg)
    curv_agg = CurvatureAggregator.from_config(cfg)
    sem_agg = CompletenessAggregator.from_config(cfg)
    traj = LocalTrajectoryDivergence()

    domains_cfg = cfg.raw.get("domains", {})
    all_domains = [d for tier_domains in domains_cfg.values() for d in tier_domains]
    n_prompts = cfg.raw.get("prompts_per_domain", 10)
    n_pert = cfg.raw.get("perturbations_per_prompt", 3)

    all_rows: list[dict[str, Any]] = []

    for domain in all_domains:
        logger.info("Processing domain: %s", domain)
        examples = load_domain_examples(domain, n=n_prompts, seed=cfg.seed)

        for ex in examples:
            prompt = ex["prompt"]
            context = ex.get("context", "")
            reference = ex.get("answer", "")

            variants: list[dict[str, Any]] = [
                {"prompt": prompt, "perturbation": "none", "context": context, "reference": reference}
            ]
            for sample in perturb_suite.perturb_text(prompt, n=n_pert):
                variants.append({
                    "prompt": sample.perturbed_text,
                    "perturbation": sample.perturbation_name,
                    "context": context,
                    "reference": reference,
                })

            variant_states: list[np.ndarray] = []
            variant_meta: list[dict[str, Any]] = []
            original_states: np.ndarray | None = None

            for variant in variants:
                inputs, states = _extract_last_token(extractor, tokenizer, variant["prompt"])
                response = _generate(model, tokenizer, inputs, device, cfg)
                sem_bundle = sem_agg.score(
                    variant["prompt"],
                    response,
                    context=variant["context"],
                    reference=variant["reference"],
                    domain=domain,
                )
                if variant["perturbation"] == "none":
                    original_states = states
                variant_states.append(states)
                variant_meta.append({
                    "domain": domain,
                    "prompt": variant["prompt"][:100],
                    "response": response[:100],
                    "model_alias": model_spec.alias,
                    "model_params_billions": model_spec.params_billions,
                    "perturbation": variant["perturbation"],
                    **sem_bundle.metrics,
                })

            if not variant_states:
                continue
            stacked = np.stack(variant_states, axis=1)  # (n_layers, n_variants, hidden_dim)
            for variant_idx, meta in enumerate(variant_meta):
                pair_metrics: dict[str, float] = {}
                if original_states is not None and meta["perturbation"] != "none":
                    pair_metrics = traj.compute_from_trajectory_pair(
                        original_states,
                        variant_states[variant_idx],
                    )
                for layer_idx in range(stacked.shape[0]):
                    curv_bundle = curv_agg.compute_layer(stacked[layer_idx], layer_idx=layer_idx)
                    all_rows.append({
                        **meta,
                        "layer_idx": layer_idx,
                        **curv_bundle.metrics,
                        **pair_metrics,
                    })

    df = pd.DataFrame(all_rows)
    store.save_df("features", df)

    corr_df = compute_correlation_matrix(df) if not df.empty else pd.DataFrame()
    partial_df = compute_partial_correlations(df) if not df.empty else pd.DataFrame()
    findings = summarise_findings(corr_df, partial_df) if not corr_df.empty else {}
    store.save_json("correlation_matrix", corr_df.to_dict())
    store.save_json("partial_correlations", partial_df.to_dict())
    store.save_json("findings", findings)

    if not df.empty:
        figures_dir = store.path("figures")
        plot_phase_space(df, output_path=figures_dir / "phase_space")
        plot_layerwise_curvature(
            df,
            curvature_cols=["trajectory_divergence", "intrinsic_dimension"],
            output_path=figures_dir / "layerwise",
        )

    logger.info("Phase 1 complete. %d rows collected.", len(all_rows))
    return {"n_rows": len(all_rows), "findings": findings, "output_dir": str(store.phase_dir)}
