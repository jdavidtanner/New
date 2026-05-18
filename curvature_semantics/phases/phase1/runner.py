"""Phase 1 orchestrator: extract states → perturb → curvature+completeness → correlate → save."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from curvature_semantics.core.config_manager import ConfigManager, ExperimentConfig
from curvature_semantics.core.artifact_store import ArtifactStore
from curvature_semantics.core.dataset_loader import load_domain_examples
from curvature_semantics.core.logging_utils import get_logger, log_config
from curvature_semantics.core.model_loader import load_model_and_tokenizer, get_num_layers
from curvature_semantics.core.hidden_state_extractor import HiddenStateExtractor
from curvature_semantics.perturbations.perturbation_suite import PerturbationSuite
from curvature_semantics.curvature.curvature_aggregator import CurvatureAggregator
from curvature_semantics.semantics.completeness_aggregator import CompletenessAggregator
from curvature_semantics.regression.feature_builder import build_feature_df
from curvature_semantics.phases.phase1.correlation_analysis import (
    compute_correlation_matrix, compute_partial_correlations, summarise_findings,
)
from curvature_semantics.visualization.phase_space import plot_phase_space
from curvature_semantics.visualization.layerwise_geometry import plot_layerwise_curvature

logger = get_logger(__name__)


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
    n_layers = get_num_layers(model)
    extractor = HiddenStateExtractor(model, extract_logits=True)
    perturb_suite = PerturbationSuite.from_config(cfg)
    curv_agg = CurvatureAggregator.from_config(cfg)
    sem_agg = CompletenessAggregator.from_config(cfg)

    domains_cfg = cfg.raw.get("domains", {})
    all_domains = []
    for tier_domains in domains_cfg.values():
        all_domains.extend(tier_domains)

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

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)

            # Extract hidden states
            bundle = extractor.extract(inputs["input_ids"], inputs.get("attention_mask"))
            hs_array = bundle.last_token_array()  # (n_layers, 1, hidden_dim)
            hs_array = hs_array[:, 0, :]  # (n_layers, hidden_dim)

            # Generate response
            with torch.no_grad():
                gen_ids = model.generate(
                    inputs["input_ids"].to(cfg.device),
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature if cfg.temperature > 0 else None,
                    do_sample=cfg.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Semantic completeness on original
            sem_bundle = sem_agg.score(prompt, response, context=context, reference=reference, domain=domain)

            # Curvature per layer for original
            for layer_idx in range(hs_array.shape[0]):
                hs_layer = hs_array[layer_idx:layer_idx + 1]  # (1, hidden_dim)
                curv_bundle = curv_agg.compute_layer(hs_layer, layer_idx=layer_idx)
                row = {
                    "domain": domain,
                    "prompt": prompt[:100],
                    "response": response[:100],
                    "layer_idx": layer_idx,
                    **curv_bundle.metrics,
                    **sem_bundle.metrics,
                    "model_alias": model_spec.alias,
                    "model_params_billions": model_spec.params_billions,
                    "perturbation": "none",
                }
                all_rows.append(row)

            # Perturbations
            pert_samples = perturb_suite.perturb_text(prompt, n=n_pert)
            for ps in pert_samples:
                p_inputs = tokenizer(ps.perturbed_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                p_bundle = extractor.extract(p_inputs["input_ids"], p_inputs.get("attention_mask"))
                p_hs = p_bundle.last_token_array()[:, 0, :]

                with torch.no_grad():
                    p_gen_ids = model.generate(
                        p_inputs["input_ids"].to(cfg.device),
                        max_new_tokens=cfg.max_new_tokens,
                        temperature=cfg.temperature if cfg.temperature > 0 else None,
                        do_sample=cfg.temperature > 0,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                p_response = tokenizer.decode(p_gen_ids[0][p_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                p_sem = sem_agg.score(ps.perturbed_text, p_response, context=context, reference=reference, domain=domain)

                # Trajectory divergence between original and perturbed
                from curvature_semantics.curvature.trajectory_divergence import LocalTrajectoryDivergence
                traj = LocalTrajectoryDivergence()
                traj_metrics = traj.compute_from_trajectory_pair(hs_array, p_hs)

                for layer_idx in range(p_hs.shape[0]):
                    p_curv = curv_agg.compute_layer(p_hs[layer_idx:layer_idx + 1], layer_idx=layer_idx)
                    row = {
                        "domain": domain,
                        "prompt": ps.perturbed_text[:100],
                        "response": p_response[:100],
                        "layer_idx": layer_idx,
                        **p_curv.metrics,
                        **p_sem.metrics,
                        **traj_metrics,
                        "model_alias": model_spec.alias,
                        "model_params_billions": model_spec.params_billions,
                        "perturbation": ps.perturbation_name,
                    }
                    all_rows.append(row)

    df = pd.DataFrame(all_rows)
    store.save_df("features", df)

    # Correlation analysis
    corr_df = compute_correlation_matrix(df)
    partial_df = compute_partial_correlations(df)
    findings = summarise_findings(corr_df, partial_df)
    store.save_json("correlation_matrix", corr_df.to_dict())
    store.save_json("partial_correlations", partial_df.to_dict())
    store.save_json("findings", findings)

    # Visualisations
    figures_dir = store.path("figures")
    plot_phase_space(df, output_path=figures_dir / "phase_space")
    plot_layerwise_curvature(df, curvature_cols=["trajectory_divergence", "intrinsic_dimension"], output_path=figures_dir / "layerwise")

    logger.info("Phase 1 complete. %d rows collected.", len(all_rows))
    return {"n_rows": len(all_rows), "findings": findings, "output_dir": str(store.phase_dir)}
