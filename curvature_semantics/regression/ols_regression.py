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

    def fit(self, df: pd.DataFrame) -> OLSRegression:
        try:
            import statsmodels.api as sm
        except ImportError:
            logger.warning("statsmodels not installed; using dependency-free NumPy OLS fallback")
            return self._fit_numpy(df)

        available = [p for p in self.predictors if p in df.columns]
        if self.target not in df.columns:
            raise ValueError(f"Target '{self.target}' not in DataFrame")
        X = df[available].fillna(df[available].mean())
        y = df[self.target].fillna(0.0)
        X = sm.add_constant(X)
        self._result = sm.OLS(y, X).fit()
        logger.info("OLS R²=%.4f  AIC=%.2f", self._result.rsquared, self._result.aic)
        return self

    def _fit_numpy(self, df: pd.DataFrame) -> OLSRegression:
        available = [p for p in self.predictors if p in df.columns]
        if self.target not in df.columns:
            raise ValueError(f"Target '{self.target}' not in DataFrame")
        X_df = df[available].fillna(df[available].mean() if available else 0.0)
        X = X_df.to_numpy(dtype=float)
        y = df[self.target].fillna(0.0).to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(df)), X])
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        y_hat = design @ coef
        ss_res = float(np.square(y - y_hat).sum())
        ss_tot = float(np.square(y - y.mean()).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        self._numpy_intercept = float(coef[0])
        self._numpy_coef = coef[1:]
        self._numpy_features = available
        self._numpy_r2 = float(max(0.0, min(1.0, r2)))
        logger.info("NumPy OLS R²=%.4f", self._numpy_r2)
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
        if hasattr(self, "_numpy_coef"):
            return {
                "r_squared": self._numpy_r2,
                "coefficients": dict(zip(self._numpy_features, self._numpy_coef.tolist())),
                "intercept": self._numpy_intercept,
            }
        return {}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._result is not None:
            import statsmodels.api as sm
            available = [p for p in self.predictors if p in df.columns]
            X = sm.add_constant(df[available].fillna(0.0))
            return self._result.predict(X).values
        if hasattr(self, "_numpy_coef"):
            X = df[self._numpy_features].fillna(0.0).to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(df)), X])
            coef = np.concatenate([[self._numpy_intercept], self._numpy_coef])
            return design @ coef
        return np.zeros(len(df))
