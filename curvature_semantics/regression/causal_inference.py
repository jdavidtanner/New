"""Causal inference: DoWhy causal graph for curvature -> completeness path."""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)

# DAG: curvature -> semantic_completeness; confounders: entropy, model_size
_DEFAULT_GRAPH = """
digraph {
    trajectory_divergence -> nli_entailment;
    entropy -> nli_entailment;
    entropy -> trajectory_divergence;
    model_params_billions -> nli_entailment;
    model_params_billions -> trajectory_divergence;
}
"""


class CausalAnalysis:
    """Wraps DoWhy for estimating causal effect of curvature on completeness."""

    def __init__(
        self,
        treatment: str = "trajectory_divergence",
        outcome: str = "nli_entailment",
        graph: str = _DEFAULT_GRAPH,
    ):
        self.treatment = treatment
        self.outcome = outcome
        self.graph = graph
        self._estimate = None

    def fit(self, df: pd.DataFrame) -> "CausalAnalysis":
        try:
            import dowhy
            from dowhy import CausalModel
        except ImportError:
            logger.warning("dowhy not installed; skipping causal analysis")
            return self

        if self.treatment not in df.columns or self.outcome not in df.columns:
            logger.warning("Treatment or outcome not in DataFrame")
            return self

        try:
            model = CausalModel(
                data=df.fillna(df.mean(numeric_only=True)),
                treatment=self.treatment,
                outcome=self.outcome,
                graph=self.graph,
            )
            identified = model.identify_effect(proceed_when_unidentifiable=True)
            self._estimate = model.estimate_effect(
                identified,
                method_name="backdoor.linear_regression",
            )
            logger.info("Causal estimate: %.4f", self._estimate.value)
        except Exception as exc:
            logger.warning("Causal estimation failed: %s", exc)
        return self

    def summary(self) -> dict[str, Any]:
        if self._estimate is None:
            return {}
        return {
            "causal_estimate": float(self._estimate.value),
            "estimand_type": str(getattr(self._estimate, "estimand_type", "")),
        }
