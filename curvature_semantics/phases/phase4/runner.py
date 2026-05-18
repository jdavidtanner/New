"""Phase 4 orchestrator: applies each intervention type; measures curvature+completeness delta."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from curvature_semantics.core.artifact_store import ArtifactStore
from curvature_semantics.core.config_manager import ExperimentConfig
from curvature_semantics.core.dataset_loader import load_domain_examples
from curvature_semantics.core.hidden_state_extractor import HiddenStateExtractor
from curvature_semantics.core.logging_utils import get_logger
from curvature_semantics.core.model_loader import load_model_and_tokenizer
from curvature_semantics.curvature.curvature_aggregator import CurvatureAggregator
from curvature_semantics.phases.phase4.bridge_concept_adder import build_bridge_variants
from curvature_semantics.phases.phase4.contradiction_remover import build_decontradicted_variants
from curvature_semantics.phases.phase4.curvature_aware_retrieval import CurvatureAwareRetriever
from curvature_semantics.phases.phase4.evidence_inserter import build_evidence_variants
from curvature_semantics.semantics.completeness_aggregator import CompletenessAggregator
from curvature_semantics.semantics.nli_scorer import NLIScorer
from curvature_semantics.visualization.phase_space import plot_intervention_delta

logger = get_logger(__name__)


def _score_examples(
    examples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    extractor: HiddenStateExtractor,
    curv_agg: CurvatureAggregator,
    sem_agg: CompletenessAggregator,
    cfg: ExperimentConfig,
    intervention_label: str = "none",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hidden_by_example: list[np.ndarray] = []
    row_metadata: list[dict[str, Any]] = []

    for ex in examples:
        prompt = ex["prompt"]
        context = ex.get("context", "")
        reference = ex.get("answer", "")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
        bundle = extractor.extract(inputs["input_ids"], inputs.get("attention_mask"))
        hidden_by_example.append(bundle.last_token_array()[:, 0, :])

        with torch.no_grad():
            gen_ids = model.generate(
                inputs["input_ids"].to(next(model.parameters()).device),
                max_new_tokens=cfg.max_new_tokens,
                temperature=None,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        sem = sem_agg.score(prompt, response, context=context, reference=reference, domain=ex.get("domain", ""))
        row_metadata.append({
            "domain": ex.get("domain", ""),
            "intervention": intervention_label,
            **sem.metrics,
        })

    if not hidden_by_example:
        return rows

    stacked = np.stack(hidden_by_example, axis=1)  # (n_layers, n_examples, hidden_dim)
    mid_layer = stacked.shape[0] // 2
    curv = curv_agg.compute_layer(stacked[mid_layer], layer_idx=mid_layer)
    for metadata in row_metadata:
        rows.append({**metadata, **curv.metrics})
    return rows


def run(cfg: ExperimentConfig) -> dict[str, Any]:
    store = ArtifactStore.from_config(cfg)
    store.save_config(cfg.to_dict())

    model_spec = cfg.model
    model, tokenizer = load_model_and_tokenizer(
        model_spec.id,
        device=cfg.device,
        dtype=cfg.dtype,
        load_in_8bit=model_spec.load_in_8bit,
        load_in_4bit=model_spec.load_in_4bit,
    )
    extractor = HiddenStateExtractor(model, extract_logits=False)
    curv_agg = CurvatureAggregator.from_config(cfg)
    sem_agg = CompletenessAggregator.from_config(cfg)
    nli = NLIScorer()

    intervention_cfg = cfg.raw.get("interventions", {})
    domains = cfg.raw.get("domains", ["speculative_medicine"])
    n_prompts = cfg.raw.get("prompts_per_domain", 20)

    all_rows: list[dict[str, Any]] = []

    for domain in domains:
        examples = load_domain_examples(domain, n=n_prompts, seed=cfg.seed)
        base_rows = _score_examples(examples, model, tokenizer, extractor, curv_agg, sem_agg, cfg, "baseline")
        all_rows.extend(base_rows)

        if intervention_cfg.get("evidence_insertion", {}).get("enabled", True):
            num_sent = intervention_cfg["evidence_insertion"].get("num_sentences", 3)
            _, aug = build_evidence_variants(examples, num_sentences=num_sent)
            rows = _score_examples(aug, model, tokenizer, extractor, curv_agg, sem_agg, cfg, "evidence_insertion")
            all_rows.extend(rows)

        if intervention_cfg.get("bridge_concept", {}).get("enabled", True):
            max_c = intervention_cfg["bridge_concept"].get("max_concepts", 5)
            _, aug = build_bridge_variants(examples, max_concepts=max_c)
            rows = _score_examples(aug, model, tokenizer, extractor, curv_agg, sem_agg, cfg, "bridge_concept")
            all_rows.extend(rows)

        if intervention_cfg.get("contradiction_removal", {}).get("enabled", True):
            thresh = intervention_cfg["contradiction_removal"].get("nli_threshold", 0.7)
            _, aug = build_decontradicted_variants(examples, nli, threshold=thresh)
            rows = _score_examples(aug, model, tokenizer, extractor, curv_agg, sem_agg, cfg, "contradiction_removal")
            all_rows.extend(rows)

        if intervention_cfg.get("curvature_aware_retrieval", {}).get("enabled", True):
            documents = [ex["context"] for ex in examples if ex.get("context")]
            if documents:
                retriever = CurvatureAwareRetriever(
                    documents=documents,
                    retrieval_model=intervention_cfg["curvature_aware_retrieval"].get("retrieval_model", "sentence-transformers/all-MiniLM-L6-v2"),
                    curvature_weight=intervention_cfg["curvature_aware_retrieval"].get("curvature_weight", 0.4),
                    top_k=intervention_cfg["curvature_aware_retrieval"].get("top_k", 3),
                )
                aug_examples = []
                for ex in examples:
                    retrieved = retriever.retrieve(ex["prompt"])
                    extra_context = " ".join(r["document"] for r in retrieved)
                    aug_ex = dict(ex)
                    aug_ex["context"] = extra_context
                    aug_ex["intervention"] = "curvature_aware_retrieval"
                    aug_examples.append(aug_ex)
                rows = _score_examples(aug_examples, model, tokenizer, extractor, curv_agg, sem_agg, cfg, "curvature_aware_retrieval")
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    store.save_df("intervention_features", df)

    baseline_df = df[df["intervention"] == "baseline"]
    summary: dict[str, Any] = {}
    for intervention in df["intervention"].unique():
        if intervention == "baseline":
            continue
        inter_df = df[df["intervention"] == intervention]
        delta_curv = inter_df["trajectory_divergence"].mean() - baseline_df["trajectory_divergence"].mean() if "trajectory_divergence" in df.columns else None
        delta_comp = inter_df["nli_entailment"].mean() - baseline_df["nli_entailment"].mean() if "nli_entailment" in df.columns else None
        summary[intervention] = {
            "delta_curvature": float(delta_curv) if delta_curv is not None else None,
            "delta_completeness": float(delta_comp) if delta_comp is not None else None,
        }

    store.save_json("intervention_summary", summary)

    figures_dir = store.path("figures")
    for intervention in df["intervention"].unique():
        if intervention == "baseline":
            continue
        plot_intervention_delta(
            baseline_df, df[df["intervention"] == intervention],
            output_path=figures_dir / f"delta_{intervention}",
        )

    logger.info("Phase 4 complete. Interventions: %s", list(summary.keys()))
    return {"n_rows": len(all_rows), "intervention_summary": summary, "output_dir": str(store.phase_dir)}
