"""Phase 4 orchestrator: applies each intervention type; measures curvature+completeness delta."""

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
from curvature_semantics.semantics.nli_scorer import NLIScorer
from curvature_semantics.phases.phase4.evidence_inserter import build_evidence_variants
from curvature_semantics.phases.phase4.bridge_concept_adder import build_bridge_variants
from curvature_semantics.phases.phase4.contradiction_remover import build_decontradicted_variants
from curvature_semantics.phases.phase4.curvature_aware_retrieval import CurvatureAwareRetriever
from curvature_semantics.visualization.phase_space import plot_intervention_delta

logger = get_logger(__name__)


def _apply_chat_template(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _collect_batch(
    examples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    extractor: HiddenStateExtractor,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, list[str]]:
    """One forward pass per example; returns (n_examples, n_layers, hidden_dim) and responses."""
    hs_list: list[np.ndarray] = []
    responses: list[str] = []
    for ex in examples:
        formatted = _apply_chat_template(tokenizer, ex["prompt"])
        inputs = tokenizer(
            formatted, return_tensors="pt",
            truncation=True, max_length=512, padding=False,
        )
        bundle = extractor.extract(inputs["input_ids"], inputs.get("attention_mask"))
        # last_token_array() → (n_layers, batch=1, hidden_dim); take batch dim 0
        hs_list.append(bundle.last_token_array()[:, 0, :])

        with torch.no_grad():
            gen_ids = model.generate(
                inputs["input_ids"].to(cfg.device),
                max_new_tokens=cfg.max_new_tokens,
                temperature=None, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        responses.append(
            tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        )

    # (n_examples, n_layers, hidden_dim)
    batch_hs = np.stack(hs_list).astype(np.float32)
    return batch_hs, responses


def _score_batch(
    examples: list[dict[str, Any]],
    responses: list[str],
    batch_hs: np.ndarray,
    curv_agg: CurvatureAggregator,
    sem_agg: CompletenessAggregator,
    intervention_label: str,
) -> list[dict[str, Any]]:
    """Batch-compute curvature at mid-layer, then per-example semantic scoring."""
    n_layers = batch_hs.shape[1]
    mid_layer = n_layers // 2
    # (n_examples, hidden_dim) at mid-layer
    curv = curv_agg.compute_layer(batch_hs[:, mid_layer, :], layer_idx=mid_layer)

    rows: list[dict[str, Any]] = []
    for ex, response in zip(examples, responses):
        sem = sem_agg.score(
            ex["prompt"], response,
            context=ex.get("context", ""),
            reference=ex.get("answer", ""),
            domain=ex.get("domain", ""),
        )
        rows.append({
            "domain": ex.get("domain", ""),
            "intervention": intervention_label,
            **curv.metrics,
            **sem.metrics,
        })
    return rows


def run(cfg: ExperimentConfig) -> dict[str, Any]:
    store = ArtifactStore.from_config(cfg)
    store.save_config(cfg.to_dict())

    model_spec = cfg.model
    model, tokenizer = load_model_and_tokenizer(
        model_spec.id, device=cfg.device, dtype=cfg.dtype,
    )
    extractor = HiddenStateExtractor(model, extract_logits=False)
    curv_agg = CurvatureAggregator.from_config(cfg)
    sem_agg = CompletenessAggregator.from_config(cfg)
    nli = NLIScorer()

    intervention_cfg = cfg.raw.get("interventions", {})
    domains = cfg.raw.get("domains", ["medical_advice"])
    n_prompts = cfg.raw.get("prompts_per_domain", 12)

    all_rows: list[dict[str, Any]] = []

    for domain in domains:
        logger.info("=== Domain: %s ===", domain)
        examples = load_domain_examples(domain, n=n_prompts, seed=cfg.seed)
        for ex in examples:
            ex["domain"] = domain

        # ── baseline ──────────────────────────────────────────────────────────
        logger.info("  intervention=baseline")
        batch_hs, responses = _collect_batch(examples, model, tokenizer, extractor, cfg)
        rows = _score_batch(examples, responses, batch_hs, curv_agg, sem_agg, "baseline")
        all_rows.extend(rows)
        _log_summary("baseline", rows)

        # ── evidence insertion ────────────────────────────────────────────────
        if intervention_cfg.get("evidence_insertion", {}).get("enabled", True):
            num_sent = intervention_cfg["evidence_insertion"].get("num_sentences", 3)
            logger.info("  intervention=evidence_insertion (num_sentences=%d)", num_sent)
            _, aug = build_evidence_variants(examples, num_sentences=num_sent)
            batch_hs, responses = _collect_batch(aug, model, tokenizer, extractor, cfg)
            rows = _score_batch(aug, responses, batch_hs, curv_agg, sem_agg, "evidence_insertion")
            all_rows.extend(rows)
            _log_summary("evidence_insertion", rows)

        # ── bridge concept ────────────────────────────────────────────────────
        if intervention_cfg.get("bridge_concept", {}).get("enabled", True):
            max_c = intervention_cfg["bridge_concept"].get("max_concepts", 5)
            logger.info("  intervention=bridge_concept (max_concepts=%d)", max_c)
            _, aug = build_bridge_variants(examples, max_concepts=max_c)
            batch_hs, responses = _collect_batch(aug, model, tokenizer, extractor, cfg)
            rows = _score_batch(aug, responses, batch_hs, curv_agg, sem_agg, "bridge_concept")
            all_rows.extend(rows)
            _log_summary("bridge_concept", rows)

        # ── contradiction removal ─────────────────────────────────────────────
        if intervention_cfg.get("contradiction_removal", {}).get("enabled", True):
            thresh = intervention_cfg["contradiction_removal"].get("nli_threshold", 0.7)
            logger.info("  intervention=contradiction_removal (threshold=%.2f)", thresh)
            _, aug = build_decontradicted_variants(examples, nli, threshold=thresh)
            batch_hs, responses = _collect_batch(aug, model, tokenizer, extractor, cfg)
            rows = _score_batch(aug, responses, batch_hs, curv_agg, sem_agg, "contradiction_removal")
            all_rows.extend(rows)
            _log_summary("contradiction_removal", rows)

        # ── curvature-aware retrieval ─────────────────────────────────────────
        if intervention_cfg.get("curvature_aware_retrieval", {}).get("enabled", True):
            documents = [ex.get("context", "") for ex in examples if ex.get("context")]
            if documents:
                retriever_cfg = intervention_cfg["curvature_aware_retrieval"]
                retriever = CurvatureAwareRetriever(
                    documents=documents,
                    retrieval_model=retriever_cfg.get("retrieval_model", "sentence-transformers/all-MiniLM-L6-v2"),
                    curvature_weight=retriever_cfg.get("curvature_weight", 0.4),
                    top_k=retriever_cfg.get("top_k", 3),
                )
                aug = []
                for ex in examples:
                    retrieved = retriever.retrieve(ex["prompt"])
                    aug_ex = dict(ex)
                    aug_ex["context"] = " ".join(r["document"] for r in retrieved)
                    aug.append(aug_ex)
                logger.info("  intervention=curvature_aware_retrieval (%d docs indexed)", len(documents))
                batch_hs, responses = _collect_batch(aug, model, tokenizer, extractor, cfg)
                rows = _score_batch(aug, responses, batch_hs, curv_agg, sem_agg, "curvature_aware_retrieval")
                all_rows.extend(rows)
                _log_summary("curvature_aware_retrieval", rows)
            else:
                logger.info("  intervention=curvature_aware_retrieval skipped (no context in examples)")

    df = pd.DataFrame(all_rows)
    store.save_df("intervention_features", df)
    logger.info("Saved %d rows", len(df))

    summary = _compute_summary(df)
    store.save_json("intervention_summary", summary)
    _log_intervention_summary(summary)

    figures_dir = store.path("figures")
    baseline_df = df[df["intervention"] == "baseline"]
    for intervention in df["intervention"].unique():
        if intervention == "baseline":
            continue
        plot_intervention_delta(
            baseline_df, df[df["intervention"] == intervention],
            output_path=figures_dir / f"delta_{intervention}",
        )

    logger.info("Phase 4 complete. Interventions tested: %s", list(summary.keys()))
    return {"n_rows": len(all_rows), "intervention_summary": summary, "output_dir": str(store.phase_dir)}


def _compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    baseline_df = df[df["intervention"] == "baseline"]
    summary: dict[str, Any] = {}
    for intervention in df["intervention"].unique():
        if intervention == "baseline":
            continue
        inter_df = df[df["intervention"] == intervention]
        delta_curv = (
            float(inter_df["trajectory_divergence"].mean() - baseline_df["trajectory_divergence"].mean())
            if "trajectory_divergence" in df.columns else None
        )
        delta_ent = (
            float(inter_df["nli_entailment"].mean() - baseline_df["nli_entailment"].mean())
            if "nli_entailment" in df.columns else None
        )
        delta_con = (
            float(inter_df["nli_contradiction"].mean() - baseline_df["nli_contradiction"].mean())
            if "nli_contradiction" in df.columns else None
        )
        summary[intervention] = {
            "delta_curvature": delta_curv,
            "delta_completeness_entailment": delta_ent,
            "delta_completeness_contradiction": delta_con,
            "hypothesis_consistent": (
                delta_curv is not None and delta_ent is not None
                and delta_curv < 0 and delta_ent > 0
            ),
        }
    return summary


def _log_summary(label: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ent = float(np.mean([r.get("nli_entailment", 0) for r in rows]))
    con = float(np.mean([r.get("nli_contradiction", 0) for r in rows]))
    curv = float(np.mean([r.get("trajectory_divergence", 0) for r in rows]))
    logger.info(
        "    %-28s  traj_div=%.3f  entail=%.3f  contra=%.3f",
        label, curv, ent, con,
    )


def _log_intervention_summary(summary: dict[str, Any]) -> None:
    for intervention, vals in summary.items():
        consistent = "CONSISTENT" if vals.get("hypothesis_consistent") else "inconsistent"
        logger.info(
            "  %-30s  Δcurv=%+.3f  Δentail=%+.3f  %s",
            intervention,
            vals.get("delta_curvature") or 0.0,
            vals.get("delta_completeness_entailment") or 0.0,
            consistent,
        )
