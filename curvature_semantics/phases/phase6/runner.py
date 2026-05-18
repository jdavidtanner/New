"""Phase 6 orchestrator: build routing layer; evaluate on benchmark; compare baselines."""

from __future__ import annotations

from typing import Any

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


def run(cfg: ExperimentConfig) -> dict[str, Any]:
    store = ArtifactStore.from_config(cfg)
    store.save_config(cfg.to_dict())

    model_spec = cfg.model
    model, tokenizer = load_model_and_tokenizer(
        model_spec.id, device=cfg.device, dtype=cfg.dtype,
    )
    extractor = HiddenStateExtractor(model, extract_logits=True)
    curv_agg = CurvatureAggregator.from_config(cfg)
    sem_agg = CompletenessAggregator.from_config(cfg)

    routing_cfg = cfg.raw.get("routing", {})
    policy = ThresholdPolicy(
        low=routing_cfg.get("thresholds", {}).get("low_curvature", 0.3),
        medium=routing_cfg.get("thresholds", {}).get("medium_curvature", 0.6),
        high=routing_cfg.get("thresholds", {}).get("high_curvature", 0.85),
    )
    router = CurvatureAwareRouter(policy=policy)

    eval_cfg = cfg.raw.get("evaluation", {})
    n_examples = eval_cfg.get("n_examples", 100)

    # Load evaluation data
    eval_domain = "historical_dates"
    examples = load_domain_examples(eval_domain, n=n_examples, seed=cfg.seed)
    gold_answers = [ex.get("answer", "") for ex in examples]

    # Build simple corpus for retrieval (from context fields)
    documents = [ex.get("context", ex["prompt"]) for ex in examples]
    try:
        from curvature_semantics.phases.phase4.curvature_aware_retrieval import CurvatureAwareRetriever
        retriever = CurvatureAwareRetriever(
            documents=documents,
            retrieval_model=cfg.raw.get("retrieval", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            top_k=cfg.raw.get("retrieval", {}).get("top_k", 3),
        )
        def retrieve_fn(prompt: str) -> str:
            hits = retriever.retrieve(prompt)
            return " ".join(h["document"] for h in hits)
    except Exception:
        retrieve_fn = lambda p: ""

    def generate_fn(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            gen_ids = model.generate(
                inputs["input_ids"].to(cfg.device),
                max_new_tokens=cfg.max_new_tokens,
                temperature=None, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Run each system
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

    for ex in examples:
        prompt = ex["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
        bundle = extractor.extract(inputs["input_ids"], inputs.get("attention_mask"))
        hs_array = bundle.last_token_array()
        mid = hs_array.shape[0] // 2
        hs = hs_array[mid, :, :][np.newaxis, :]

        curv_bundle = curv_agg.compute_layer(hs, layer_idx=mid)

        logits_np = None
        if bundle.logits is not None:
            logits_np = bundle.logits[0, -1, :].float().numpy()

        signal = router.estimate_signal(logits_np, curvature_bundle=curv_bundle)

        # Curvature router
        result = router.route(prompt, signal, generate_fn, retrieve_fn)
        system_results["curvature_router"].append(result)

        # Baselines
        if baseline_cfg.get("vanilla", True):
            system_results["vanilla"].append(vanilla.run(prompt, generate_fn))
        if baseline_cfg.get("rag_only", True):
            system_results["rag_only"].append(rag.run(prompt, generate_fn, retrieve_fn=retrieve_fn))
        if baseline_cfg.get("calibrated", True):
            system_results["calibrated"].append(calibrated.run(prompt, generate_fn, logits=logits_np))
        if baseline_cfg.get("self_consistency", True):
            system_results["self_consistency"].append(sc.run(prompt, generate_fn))

    # Evaluate
    active_systems = {k: v for k, v in system_results.items() if v}
    comparison_df = compare_systems(active_systems, gold_answers)
    store.save_df("system_comparison", comparison_df)
    store.save_json("system_comparison", comparison_df.to_dict())

    logger.info("Phase 6 complete.\n%s", comparison_df.to_string())
    curvature_router_metrics = comparison_df.loc["curvature_router"].to_dict() if "curvature_router" in comparison_df.index else {}
    return {
        "n_examples": n_examples,
        "curvature_router_metrics": curvature_router_metrics,
        "comparison": comparison_df.to_dict(),
        "output_dir": str(store.phase_dir),
    }
