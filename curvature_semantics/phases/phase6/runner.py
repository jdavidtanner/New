"""Phase 6 orchestrator: build routing layer; evaluate on benchmark; compare baselines."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from curvature_semantics.core.config_manager import ExperimentConfig
from curvature_semantics.core.artifact_store import ArtifactStore
from curvature_semantics.core.dataset_loader import load_domain_examples
from curvature_semantics.core.logging_utils import get_logger
from curvature_semantics.core.model_loader import load_model_and_tokenizer
from curvature_semantics.core.hidden_state_extractor import HiddenStateExtractor
from curvature_semantics.curvature.curvature_aggregator import CurvatureAggregator
from curvature_semantics.semantics.completeness_aggregator import CompletenessAggregator
from curvature_semantics.phases.phase6.routing_policy import ThresholdPolicy
from curvature_semantics.phases.phase6.curvature_router import CurvatureAwareRouter
from curvature_semantics.phases.phase6.baselines.vanilla_baseline import VanillaBaseline
from curvature_semantics.phases.phase6.baselines.rag_only_baseline import RAGOnlyBaseline
from curvature_semantics.phases.phase6.baselines.calibrated_baseline import CalibratedBaseline
from curvature_semantics.phases.phase6.baselines.self_consistency_baseline import SelfConsistencyBaseline
from curvature_semantics.phases.phase6.evaluator import compare_systems

logger = get_logger(__name__)


def _apply_chat_template(tokenizer: Any, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _make_generate_fn(model: Any, tokenizer: Any, cfg: ExperimentConfig) -> Callable[[str], str]:
    def generate_fn(prompt: str) -> str:
        formatted = _apply_chat_template(tokenizer, prompt)
        inputs = tokenizer(
            formatted, return_tensors="pt",
            truncation=True, max_length=512, padding=False,
        )
        with torch.no_grad():
            gen_ids = model.generate(
                inputs["input_ids"].to(cfg.device),
                max_new_tokens=cfg.max_new_tokens,
                temperature=None, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generate_fn


def run(cfg: ExperimentConfig) -> dict[str, Any]:
    store = ArtifactStore.from_config(cfg)
    store.save_config(cfg.to_dict())

    model_spec = cfg.model
    model, tokenizer = load_model_and_tokenizer(
        model_spec.id, device=cfg.device, dtype=cfg.dtype,
    )
    extractor = HiddenStateExtractor(model, extract_logits=True)
    curv_agg = CurvatureAggregator.from_config(cfg)

    routing_cfg = cfg.raw.get("routing", {})
    policy = ThresholdPolicy(
        low=routing_cfg.get("thresholds", {}).get("low_curvature", 0.3),
        medium=routing_cfg.get("thresholds", {}).get("medium_curvature", 0.6),
        high=routing_cfg.get("thresholds", {}).get("high_curvature", 0.85),
    )
    router = CurvatureAwareRouter(policy=policy)

    eval_cfg = cfg.raw.get("evaluation", {})
    n_examples = eval_cfg.get("n_examples", 20)

    eval_domain = eval_cfg.get("domain", "medical_advice")
    examples = load_domain_examples(eval_domain, n=n_examples, seed=cfg.seed)
    gold_answers = [ex.get("answer", "") for ex in examples]

    # Build retrieval index from example prompts (context unavailable for synthetic data)
    documents = [ex.get("context", ex["prompt"]) for ex in examples]
    try:
        from curvature_semantics.phases.phase4.curvature_aware_retrieval import CurvatureAwareRetriever
        retriever = CurvatureAwareRetriever(
            documents=documents,
            retrieval_model=cfg.raw.get("retrieval", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            top_k=cfg.raw.get("retrieval", {}).get("top_k", 3),
        )
        def retrieve_fn(prompt: str) -> str:
            return " ".join(h["document"] for h in retriever.retrieve(prompt))
    except Exception:
        retrieve_fn = lambda p: ""

    generate_fn = _make_generate_fn(model, tokenizer, cfg)

    # ── Pass 1: extract hidden states + logit entropy for all examples ────────
    logger.info("Pass 1: extracting hidden states (%d examples)", n_examples)
    all_hs: list[np.ndarray] = []
    all_logits: list[np.ndarray | None] = []

    for ex in examples:
        formatted = _apply_chat_template(tokenizer, ex["prompt"])
        inputs = tokenizer(
            formatted, return_tensors="pt",
            truncation=True, max_length=512, padding=False,
        )
        bundle = extractor.extract(inputs["input_ids"], inputs.get("attention_mask"))
        all_hs.append(bundle.last_token_array()[:, 0, :])  # (n_layers, hidden_dim)
        all_logits.append(
            bundle.logits[0, -1, :].float().cpu().numpy() if bundle.logits is not None else None
        )

    # ── Threshold calibration from entropy distribution ───────────────────────
    if cfg.raw.get("routing", {}).get("calibrate_thresholds", False):
        entropies = []
        for lp in all_logits:
            if lp is not None:
                p = np.exp(lp - lp.max()); p /= p.sum()
                entropies.append(float(-np.sum(p * np.log(p + 1e-12))))
        if entropies:
            curv_vals = [min(e / 10.0, 1.0) for e in entropies]
            new_low = float(np.percentile(curv_vals, 40))
            new_med = float(np.percentile(curv_vals, 75))
            policy.low = new_low
            policy.medium = new_med
            logger.info(
                "Calibrated thresholds from entropy distribution: low=%.3f  medium=%.3f",
                new_low, new_med,
            )

    # ── Pass 2: batch curvature at mid-layer ──────────────────────────────────
    n_layers = all_hs[0].shape[0]
    mid = n_layers // 2
    batch_hs = np.stack([hs[mid] for hs in all_hs]).astype(np.float32)  # (n, hidden_dim)
    batch_curv = curv_agg.compute_layer(batch_hs, layer_idx=mid)
    logger.info(
        "Batch curvature (layer %d): traj_div=%.4f  ID=%.4f",
        mid,
        batch_curv.metrics.get("trajectory_divergence", 0.0),
        batch_curv.metrics.get("intrinsic_dimension", 0.0),
    )

    # ── Pass 3: route + run all systems per example ───────────────────────────
    logger.info("Pass 3: running all systems")
    system_results: dict[str, list[dict[str, Any]]] = {
        "curvature_router": [],
        "vanilla": [],
        "rag_only": [],
        "calibrated": [],
        "self_consistency": [],
    }

    baseline_cfg = cfg.raw.get("baselines", {})
    vanilla = VanillaBaseline()
    rag = RAGOnlyBaseline()
    calibrated = CalibratedBaseline()
    sc = SelfConsistencyBaseline(n_samples=cfg.num_generations)

    action_counts: dict[str, int] = {}

    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        logits_np = all_logits[i]

        # Routing signal: entropy-based per-example curvature proxy
        signal = router.estimate_signal(logits_np, curvature_bundle=None)

        # Curvature router
        result = router.route(prompt, signal, generate_fn, retrieve_fn)
        result["batch_trajectory_divergence"] = batch_curv.metrics.get("trajectory_divergence", 0.0)
        system_results["curvature_router"].append(result)
        action_counts[result["action"]] = action_counts.get(result["action"], 0) + 1

        if baseline_cfg.get("vanilla", True):
            system_results["vanilla"].append(vanilla.run(prompt, generate_fn))
        if baseline_cfg.get("rag_only", True):
            system_results["rag_only"].append(rag.run(prompt, generate_fn, retrieve_fn=retrieve_fn))
        if baseline_cfg.get("calibrated", True):
            system_results["calibrated"].append(calibrated.run(prompt, generate_fn, logits=logits_np))
        if baseline_cfg.get("self_consistency", True):
            system_results["self_consistency"].append(sc.run(prompt, generate_fn))

    logger.info("Router action distribution: %s", action_counts)

    # ── Evaluation ────────────────────────────────────────────────────────────
    active_systems = {k: v for k, v in system_results.items() if v}
    comparison_df = compare_systems(active_systems, gold_answers)
    store.save_df("system_comparison", comparison_df)
    store.save_json("system_comparison_dict", comparison_df.to_dict())

    store.save_json("routing_stats", {
        "action_counts": action_counts,
        "batch_curvature": batch_curv.metrics,
    })

    logger.info("Phase 6 complete.\n%s", comparison_df.to_string())
    router_metrics = (
        comparison_df.loc["curvature_router"].to_dict()
        if "curvature_router" in comparison_df.index else {}
    )
    return {
        "n_examples": n_examples,
        "action_counts": action_counts,
        "curvature_router_metrics": router_metrics,
        "comparison": comparison_df.to_dict(),
        "output_dir": str(store.phase_dir),
    }
