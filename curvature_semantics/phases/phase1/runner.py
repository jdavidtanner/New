"""Phase 1 orchestrator: extract states → perturb → curvature+completeness → correlate → save."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from curvature_semantics.core.config_manager import ExperimentConfig
from curvature_semantics.core.artifact_store import ArtifactStore
from curvature_semantics.core.dataset_loader import load_domain_examples
from curvature_semantics.core.logging_utils import get_logger, log_config
from curvature_semantics.core.model_loader import load_model_and_tokenizer, get_num_layers
from curvature_semantics.core.hidden_state_extractor import HiddenStateExtractor
from curvature_semantics.curvature.trajectory_divergence import LocalTrajectoryDivergence
from curvature_semantics.perturbations.perturbation_suite import PerturbationSuite
from curvature_semantics.curvature.curvature_aggregator import CurvatureAggregator
from curvature_semantics.semantics.completeness_aggregator import CompletenessAggregator
from curvature_semantics.phases.phase1.correlation_analysis import (
    compute_correlation_matrix, compute_partial_correlations, summarise_findings,
)
from curvature_semantics.visualization.phase_space import plot_phase_space
from curvature_semantics.visualization.layerwise_geometry import plot_layerwise_curvature

logger = get_logger(__name__)


def _generate(model, tokenizer, input_ids, cfg):
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids.to(cfg.device),
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature if cfg.temperature > 0 else None,
            do_sample=cfg.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


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

    # Resolve which layers to extract
    layers_cfg = cfg.raw.get("layers", {})
    if layers_cfg.get("extract_all", True):
        layer_indices = None  # HiddenStateExtractor defaults to all
    else:
        raw_indices = layers_cfg.get("indices") or None
        layer_indices = [i % n_layers for i in raw_indices] if raw_indices else None

    extractor = HiddenStateExtractor(model, layer_indices=layer_indices, extract_logits=True)
    perturb_suite = PerturbationSuite.from_config(cfg)
    curv_agg = CurvatureAggregator.from_config(cfg)
    sem_agg = CompletenessAggregator.from_config(cfg)
    traj_proxy = LocalTrajectoryDivergence()

    domains_cfg = cfg.raw.get("domains", {})
    all_domains = [d for tier in domains_cfg.values() for d in tier]
    n_prompts = cfg.raw.get("prompts_per_domain", 10)
    n_pert = cfg.raw.get("perturbations_per_prompt", 3)

    all_rows: list[dict[str, Any]] = []

    for domain in all_domains:
        logger.info("=== Domain: %s ===", domain)
        examples = load_domain_examples(domain, n=n_prompts, seed=cfg.seed)

        # ----------------------------------------------------------------
        # Pass 1: collect hidden states + responses for all examples
        # ----------------------------------------------------------------
        # orig_hs: (n_examples, n_extracted_layers, hidden_dim)
        orig_hs_list: list[np.ndarray] = []          # per-example (n_layers, hidden_dim)
        pert_hs_list: list[list[np.ndarray]] = []     # per-example, per-perturbation
        responses: list[str] = []
        pert_responses: list[list[str]] = []
        pert_names: list[list[str]] = []

        for idx, ex in enumerate(examples):
            if idx % 5 == 0:
                logger.info("  [%s] %d/%d", domain, idx + 1, len(examples))
            prompt = ex["prompt"]

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding=False)
            bundle = extractor.extract(inputs["input_ids"], inputs.get("attention_mask"))
            # last_token_array: (n_layers, 1, hidden_dim) → (n_layers, hidden_dim)
            hs = bundle.last_token_array()[:, 0, :]
            orig_hs_list.append(hs)
            responses.append(_generate(model, tokenizer, inputs["input_ids"], cfg))

            # Perturbations
            pert_samples = perturb_suite.perturb_text(prompt, n=n_pert)
            p_hs_for_ex, p_resp_for_ex, p_names_for_ex = [], [], []
            for ps in pert_samples:
                p_inputs = tokenizer(ps.perturbed_text, return_tensors="pt", truncation=True, max_length=256, padding=False)
                p_bundle = extractor.extract(p_inputs["input_ids"], p_inputs.get("attention_mask"))
                p_hs = p_bundle.last_token_array()[:, 0, :]
                p_hs_for_ex.append(p_hs)
                p_resp_for_ex.append(_generate(model, tokenizer, p_inputs["input_ids"], cfg))
                p_names_for_ex.append(ps.perturbation_name)
            pert_hs_list.append(p_hs_for_ex)
            pert_responses.append(p_resp_for_ex)
            pert_names.append(p_names_for_ex)

        logger.info("  [%s] extraction complete, computing curvature over %d examples", domain, len(orig_hs_list))

        # ----------------------------------------------------------------
        # Pass 2: batch curvature over all examples — per extracted layer
        # ----------------------------------------------------------------
        n_examples = len(orig_hs_list)
        extracted_layer_indices = bundle.layer_indices  # actual indices that were extracted

        # batch_hs[li] = (n_examples, hidden_dim) for extracted layer li
        batch_orig = np.stack(orig_hs_list, axis=0)   # (n_examples, n_layers, hidden_dim)
        # Also stack all perturbed: (n_examples * n_pert, n_layers, hidden_dim)
        all_pert_flat = [p for ex_p in pert_hs_list for p in ex_p]
        if all_pert_flat:
            batch_pert = np.stack(all_pert_flat, axis=0)
        else:
            batch_pert = batch_orig

        # Curvature on combined set (original + perturbed) per layer
        combined = np.concatenate([batch_orig, batch_pert], axis=0)  # (N_all, n_layers, hidden_dim)

        layer_curv: dict[int, dict] = {}  # layer_idx → curvature metrics dict
        for li, layer_idx in enumerate(extracted_layer_indices):
            layer_combined = combined[:, li, :]  # (N_all, hidden_dim)
            cb = curv_agg.compute_layer(layer_combined, layer_idx=layer_idx)
            layer_curv[layer_idx] = cb.metrics

        # Per-example pairwise trajectory divergence (orig vs each perturbation)
        # traj_div_per_example[i] = mean divergence across perturbations for example i
        traj_div_per_example: list[float] = []
        for i, ex_p_list in enumerate(pert_hs_list):
            if not ex_p_list:
                traj_div_per_example.append(0.0)
                continue
            divs = []
            for p_hs in ex_p_list:
                td = traj_proxy.compute_from_trajectory_pair(orig_hs_list[i], p_hs)
                divs.append(td.get("trajectory_divergence", 0.0))
            traj_div_per_example.append(float(np.mean(divs)))

        # ----------------------------------------------------------------
        # Pass 3: semantic completeness + assemble rows
        # ----------------------------------------------------------------
        for i, ex in enumerate(examples):
            prompt = ex["prompt"]
            context = ex.get("context", "")
            reference = ex.get("answer", "")
            response = responses[i]

            # Logits for OOD/entropy if available (last extracted layer bundle)
            # (we re-use last bundle's logits field — approximate)
            sem_kwargs: dict[str, Any] = {}
            sem_bundle = sem_agg.score(prompt, response, context=context, reference=reference, domain=domain, **sem_kwargs)

            for layer_idx, layer_metrics in layer_curv.items():
                row: dict[str, Any] = {
                    "domain": domain,
                    "prompt_hash": hash(prompt) % 10**8,
                    "layer_idx": layer_idx,
                    "perturbation": "none",
                    "trajectory_divergence_pairwise": traj_div_per_example[i],
                    **layer_metrics,
                    **sem_bundle.metrics,
                    "model_alias": model_spec.alias,
                    "model_params_billions": model_spec.params_billions,
                }
                all_rows.append(row)

            # Rows for perturbations
            for pi, (p_hs, p_resp, p_name) in enumerate(zip(pert_hs_list[i], pert_responses[i], pert_names[i])):
                p_sem = sem_agg.score(prompt, p_resp, context=context, reference=reference, domain=domain)
                td = traj_proxy.compute_from_trajectory_pair(orig_hs_list[i], p_hs)
                for layer_idx, layer_metrics in layer_curv.items():
                    row = {
                        "domain": domain,
                        "prompt_hash": hash(prompt) % 10**8,
                        "layer_idx": layer_idx,
                        "perturbation": p_name,
                        "trajectory_divergence_pairwise": td.get("trajectory_divergence", 0.0),
                        **layer_metrics,
                        **p_sem.metrics,
                        "model_alias": model_spec.alias,
                        "model_params_billions": model_spec.params_billions,
                    }
                    all_rows.append(row)

    # ----------------------------------------------------------------
    # Save features
    # ----------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    store.save_df("features", df)
    logger.info("Saved %d rows to feature DataFrame", len(df))

    # ----------------------------------------------------------------
    # Correlation analysis
    # ----------------------------------------------------------------
    corr_df = compute_correlation_matrix(df)
    partial_df = compute_partial_correlations(df)
    findings = summarise_findings(corr_df, partial_df)
    store.save_json("correlation_matrix", corr_df.reset_index().to_dict(orient="records"))
    store.save_json("partial_correlations", partial_df.reset_index().to_dict(orient="records"))
    store.save_json("findings", findings)

    # Print readable summary
    logger.info("=== Correlation matrix (Spearman) ===")
    for col in ["nli_entailment", "nli_contradiction", "self_consistency"]:
        if col in corr_df.columns:
            row_str = "  ".join(
                f"{idx}: {corr_df.loc[idx, col]:.3f}"
                for idx in corr_df.index
                if col in corr_df.columns
            )
            logger.info("  %s <- %s", col, row_str)

    # ----------------------------------------------------------------
    # Visualisations
    # ----------------------------------------------------------------
    try:
        figures_dir = store.path("figures")
        plot_phase_space(df, output_path=figures_dir / "phase_space")
        curv_cols = [c for c in ["trajectory_divergence", "intrinsic_dimension", "neighborhood_distortion"] if c in df.columns]
        plot_layerwise_curvature(df, curvature_cols=curv_cols, output_path=figures_dir / "layerwise")
    except Exception as e:
        logger.warning("Visualisation failed (non-fatal): %s", e)

    logger.info("Phase 1 complete. %d rows collected.", len(all_rows))
    return {
        "n_rows": len(all_rows),
        "n_examples": sum(n_prompts for _ in all_domains),
        "domains": all_domains,
        "findings": findings,
        "output_dir": str(store.phase_dir),
    }
