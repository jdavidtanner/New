"""Runs all enabled semantic metrics; returns SemanticCompletenessBundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from curvature_semantics.semantics.base import SemanticMetric
from curvature_semantics.semantics.nli_scorer import NLIScorer
from curvature_semantics.semantics.factuality_scorer import FactualityScorer
from curvature_semantics.semantics.self_consistency import SelfConsistency
from curvature_semantics.semantics.calibration import CalibrationError
from curvature_semantics.semantics.ood_uncertainty import OODUncertainty
from curvature_semantics.semantics.abstention_accuracy import AbstentionAccuracy
from curvature_semantics.semantics.evidence_utilization import EvidenceUtilization
from curvature_semantics.semantics.answer_completeness import AnswerCompleteness
from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)

_METRIC_REGISTRY: dict[str, type[SemanticMetric]] = {
    "nli_entailment": NLIScorer,
    "nli_contradiction": NLIScorer,
    "factuality": FactualityScorer,
    "self_consistency": SelfConsistency,
    "calibration_error": CalibrationError,
    "ood_uncertainty": OODUncertainty,
    "abstention_accuracy": AbstentionAccuracy,
    "evidence_utilization": EvidenceUtilization,
    "answer_completeness": AnswerCompleteness,
}


@dataclass
class SemanticCompletenessBundle:
    """All semantic completeness metric values for a single (prompt, response) pair."""
    prompt: str
    response: str
    domain: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def primary(self) -> float:
        """Return the primary completeness signal (NLI entailment)."""
        return self.metrics.get("nli_entailment", 0.0)

    def to_flat_dict(self) -> dict[str, Any]:
        d = {"domain": self.domain}
        d.update(self.metrics)
        d.update(self.metadata)
        return d


class CompletenessAggregator:
    """Runs all enabled semantic metrics and returns SemanticCompletenessBundle."""

    def __init__(self, metric_names: list[str] | None = None, cfg: Any = None):
        metric_names = metric_names or list(_METRIC_REGISTRY.keys())
        nli_model = (cfg.raw.get("nli_model", "cross-encoder/nli-deberta-v3-base") if cfg else "cross-encoder/nli-deberta-v3-base")
        self.metrics: list[SemanticMetric] = []
        seen: set[type] = set()
        for name in metric_names:
            cls = _METRIC_REGISTRY.get(name)
            if cls is None:
                logger.warning("Unknown semantic metric: %s — skipping", name)
                continue
            if cls not in seen:
                if cls in (NLIScorer, FactualityScorer):
                    self.metrics.append(cls(model_name=nli_model))
                else:
                    self.metrics.append(cls())
                seen.add(cls)

    @classmethod
    def from_config(cls, cfg: Any) -> "CompletenessAggregator":
        metric_names = cfg.raw.get("semantic_metrics", list(_METRIC_REGISTRY.keys()))
        return cls(metric_names=metric_names, cfg=cfg)

    def score(
        self,
        prompt: str,
        response: str,
        context: str = "",
        reference: str = "",
        domain: str = "",
        **kwargs: Any,
    ) -> SemanticCompletenessBundle:
        bundle = SemanticCompletenessBundle(prompt=prompt, response=response, domain=domain)
        for metric in self.metrics:
            try:
                result = metric.score(prompt, response, context=context, reference=reference, **kwargs)
                bundle.metrics.update(result)
            except Exception as exc:
                logger.warning("Metric %s failed: %s", metric.name, exc)
        return bundle
