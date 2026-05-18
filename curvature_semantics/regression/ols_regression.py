"""OLS regression: SemanticCompleteness ~ K_E + Entropy + Confidence + ... """

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)


class OLSRegression:
    """Wraps statsmodels OLS with coefficient tables and model summary."""

    def __init__(self, target: str = "nli_entailment", predictors: list[str] | None = None):
        self.target = target
        self.predictors = predictors or [
            "trajectory_divergence",
            "intrinsic_dimension",
            "neighborhood_distortion",
            "entropy",
            "confidence",
            "model_params_billions",
        ]
        self._result = None

    def fit(self, df: pd.DataFrame) -> "OLSRegression":
        try:
            import statsmodels.api as sm
        except ImportError:
            logger.warning("statsmodels not installed; using sklearn OLS fallback")
            return self._fit_sklearn(df)

        available = [p for p in self.predictors if p in df.columns]
        if self.target not in df.columns:
            raise ValueError(f"Target '{self.target}' not in DataFrame")
        X = df[available].fillna(df[available].mean())
        y = df[self.target].fillna(0.0)
        X = sm.add_constant(X)
        self._result = sm.OLS(y, X).fit()
        logger.info("OLS R²=%.4f  AIC=%.2f", self._result.rsquared, self._result.aic)
        return self

    def _fit_sklearn(self, df: pd.DataFrame) -> "OLSRegression":
        from sklearn.linear_model import LinearRegression
        available = [p for p in self.predictors if p in df.columns]
        X = df[available].fillna(0.0).values
        y = df[self.target].fillna(0.0).values if self.target in df.columns else np.zeros(len(df))
        model = LinearRegression().fit(X, y)
        self._sklearn_model = model
        self._sklearn_features = available
        r2 = model.score(X, y)
        logger.info("sklearn OLS R²=%.4f", r2)
        return self

    def summary(self) -> dict[str, Any]:
        if self._result is not None:
            return {
                "r_squared": float(self._result.rsquared),
                "adj_r_squared": float(self._result.rsquared_adj),
                "aic": float(self._result.aic),
                "bic": float(self._result.bic),
                "coefficients": self._result.params.to_dict(),
                "p_values": self._result.pvalues.to_dict(),
                "conf_int": self._result.conf_int().to_dict(),
            }
        if hasattr(self, "_sklearn_model"):
            return {
                "coefficients": dict(zip(self._sklearn_features, self._sklearn_model.coef_.tolist())),
                "intercept": float(self._sklearn_model.intercept_),
            }
        return {}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._result is not None:
            import statsmodels.api as sm
            available = [p for p in self.predictors if p in df.columns]
            X = sm.add_constant(df[available].fillna(0.0))
            return self._result.predict(X).values
        if hasattr(self, "_sklearn_model"):
            X = df[self._sklearn_features].fillna(0.0).values
            return self._sklearn_model.predict(X)
        return np.zeros(len(df))
