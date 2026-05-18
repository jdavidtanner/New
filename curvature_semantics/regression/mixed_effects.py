"""Mixed-effects model with model/layer as random intercepts."""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)


class MixedEffectsRegression:
    """LMM: SemanticCompleteness ~ fixed(curvature) + (1|model) + (1|layer)."""

    def __init__(
        self,
        target: str = "nli_entailment",
        fixed: list[str] | None = None,
        group_col: str = "model_alias",
    ):
        self.target = target
        self.fixed = fixed or ["trajectory_divergence", "intrinsic_dimension", "entropy"]
        self.group_col = group_col
        self._result = None

    def fit(self, df: pd.DataFrame) -> "MixedEffectsRegression":
        try:
            import statsmodels.formula.api as smf
        except ImportError:
            logger.warning("statsmodels not installed; skipping mixed effects model")
            return self

        available = [f for f in self.fixed if f in df.columns]
        if not available or self.target not in df.columns:
            logger.warning("Insufficient columns for mixed effects model")
            return self
        if self.group_col not in df.columns:
            df = df.copy()
            df[self.group_col] = "default"

        formula = f"{self.target} ~ " + " + ".join(available)
        try:
            model = smf.mixedlm(formula, df.fillna(0.0), groups=df[self.group_col])
            self._result = model.fit(reml=False)
            logger.info("MixedLM log-likelihood=%.4f", self._result.llf)
        except Exception as exc:
            logger.warning("Mixed effects fitting failed: %s", exc)
        return self

    def summary(self) -> dict[str, Any]:
        if self._result is None:
            return {}
        return {
            "log_likelihood": float(self._result.llf),
            "aic": float(self._result.aic),
            "bic": float(self._result.bic),
            "fixed_effects": self._result.fe_params.to_dict(),
            "random_effects_variance": float(self._result.cov_re.values[0, 0]) if self._result.cov_re.size > 0 else 0.0,
        }
