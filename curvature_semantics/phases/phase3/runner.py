"""Phase 3 runner: causal activation-patching intervention.

For each (domain, target_layer, noise_alpha):
  1. Inject alpha-scaled noise at target_layer during prefill
  2. Continue autoregressive generation from the perturbed representation
  3. Measure NLI semantic quality of the patched response
  4. Capture batch curvature of the patched activations

Dose-response analysis: does quality decrease monotonically as alpha increases?
Layer sensitivity: which layer's geometry is most causally linked to output quality?
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats

from curvature_semantics.core.config_manager import ExperimentConfig
from curvature_semantics.core.artifact_store import ArtifactStore
from curvature_semantics.core.dataset_loader import load_domain_examples
from curvature_semantics.core.logging_utils import get_logger
from curvature_semantics.core.model_loader import load_model_and_tokenizer
from curvature_semantics.curvature.curvature_aggregator import CurvatureAggregator
from curvature_semantics.semantics.completeness_aggregator import CompletenessAggregator
from curvature_semantics.phases.phase3.activation_patcher import ActivationPatcher

logger = get_logger(__name__)


def _generate(model, tokenizer, input_ids, cfg) -> str:
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids.to(cfg.device),
            max_new_tokens=cfg.max_new_tokens,
            temperature=None,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


def run(cfg: ExperimentConfig) -> dict[str, Any]:
    store = ArtifactStore.from_config(cfg)
    store.save_config(cfg.to_dict())

    model_spec = cfg.model
    model, tokenizer = load_model_and_tokenizer(
        model_spec.id, device=cfg.device, dtype=cfg.dtype,
        load_in_8bit=model_spec.load_in_8bit, load_in_4bit=model_spec.load_in_4bit,
    )

    patcher = ActivationPatcher(model, rng_seed=cfg.seed)
    n_layers = patcher.n_layers()
    logger.info("Model has %d transformer layers", n_layers)

    curv_agg = CurvatureAggregator.from_config(cfg)
    sem_agg = CompletenessAggregator.from_config(cfg)

    raw_layers = cfg.raw.get("target_layers", [n_layers // 4, n_layers // 2, 3 * n_layers // 4])
    target_layers = [int(l) % n_layers for l in raw_layers]
    noise_alphas: list[float] = cfg.raw.get("noise_alphas", [0.0, 0.05, 0.1, 0.3, 1.0])

    domains_cfg = cfg.raw.get("domains", {})
    all_domains = [d for tier in domains_cfg.values() for d in tier] if isinstance(domains_cfg, dict) else list(domains_cfg)
    n_prompts = cfg.raw.get("prompts_per_domain", 10)

    all_rows: list[dict[str, Any]] = []

    for domain in all_domains:
        logger.info("=== Domain: %s ===", domain)
        examples = load_domain_examples(domain, n=n_prompts, seed=cfg.seed)

        for target_layer in target_layers:
            logger.info("  Layer %d / %d", target_layer, n_layers)

            for alpha in noise_alphas:
                # --------------------------------------------------------
                # Pass 1: collect patched responses + activations
                # --------------------------------------------------------
                patched_hs_list: list[np.ndarray] = []
                responses: list[str] = []

                for ex in examples:
                    inputs = tokenizer(
                        ex["prompt"], return_tensors="pt",
                        truncation=True, max_length=256, padding=False,
                    )
                    with patcher.patch(target_layer, alpha):
                        response = _generate(model, tokenizer, inputs["input_ids"], cfg)

                    captured = patcher.captured_activation()
                    if captured is not None:
                        patched_hs_list.append(captured[0])
                    responses.append(response)

                if not patched_hs_list:
                    continue

                # --------------------------------------------------------
                # Pass 2: batch curvature on patched activations
                # --------------------------------------------------------
                batch_hs = np.stack(patched_hs_list).astype(np.float32)
                curv_bundle = curv_agg.compute_layer(batch_hs, layer_idx=target_layer)

                # --------------------------------------------------------
                # Pass 3: semantic scoring + row assembly
                # --------------------------------------------------------
                for ex, response in zip(examples, responses):
                    sem = sem_agg.score(
                        ex["prompt"], response,
                        context=ex.get("context", ""),
                        reference=ex.get("answer", ""),
                        domain=domain,
                    )
                    row: dict[str, Any] = {
                        "domain": domain,
                        "prompt_hash": hash(ex["prompt"]) % 10**8,
                        "target_layer": target_layer,
                        "layer_rel": target_layer / max(n_layers - 1, 1),
                        "noise_alpha": alpha,
                        **curv_bundle.metrics,
                        **sem.metrics,
                        "model_alias": model_spec.alias,
                    }
                    all_rows.append(row)

                logger.info(
                    "    alpha=%.2f  traj_div=%.3f  n_examples=%d",
                    alpha,
                    curv_bundle.metrics.get("trajectory_divergence", 0.0),
                    len(patched_hs_list),
                )

    df = pd.DataFrame(all_rows)
    store.save_df("intervention_features", df)
    logger.info("Saved %d rows", len(df))

    dose_response = _dose_response_analysis(df)
    store.save_json("dose_response", dose_response)
    _log_dose_response(dose_response)

    hypothesis_supported = _verdict(dose_response)
    store.save_json("verdict", {"hypothesis_supported": hypothesis_supported})

    logger.info("Phase 3 complete. Hypothesis supported: %s", hypothesis_supported)
    return {
        "n_rows": len(df),
        "hypothesis_supported": hypothesis_supported,
        "dose_response": dose_response,
        "output_dir": str(store.phase_dir),
    }


def _dose_response_analysis(df: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}
    completeness_cols = [c for c in ["nli_entailment", "nli_contradiction", "self_consistency"] if c in df.columns]

    for (domain, layer), grp in df.groupby(["domain", "target_layer"]):
        key = f"{domain}__layer{int(layer)}"
        results[key] = {"domain": domain, "layer": int(layer), "signals": {}}

        for col in completeness_cols:
            y_by_alpha = {
                float(a): float(sub[col].fillna(0).mean())
                for a, sub in grp.groupby("noise_alpha")
            }
            x = np.array(sorted(y_by_alpha.keys()))
            y = np.array([y_by_alpha[a] for a in x])

            if len(np.unique(y)) < 2:
                continue
            r, p = stats.spearmanr(x, y)
            if not np.isfinite(r):
                continue

            results[key]["signals"][col] = {
                "spearman_r": float(r),
                "p_value": float(p),
                "supports_hypothesis": bool(r < 0 and p < 0.10),
                "per_alpha_means": {str(a): float(y_by_alpha[a]) for a in sorted(y_by_alpha)},
            }

    return results


def _log_dose_response(dose_response: dict) -> None:
    for key, val in sorted(dose_response.items()):
        for metric, sig in val.get("signals", {}).items():
            r = sig["spearman_r"]
            supported = "SUPPORTS" if sig["supports_hypothesis"] else "no support"
            logger.info("  %-35s  %-22s  r=%+.3f  %s", key, metric, r, supported)


def _verdict(dose_response: dict) -> bool:
    n_support = sum(
        1 for val in dose_response.values()
        for sig in val.get("signals", {}).values()
        if sig.get("supports_hypothesis")
    )
    n_total = sum(len(val.get("signals", {})) for val in dose_response.values())
    return n_total > 0 and (n_support / n_total) >= 0.5
