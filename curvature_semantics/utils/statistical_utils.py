"""Bootstrap CIs, effect size, partial correlation, FDR correction."""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point_estimate, lower_ci, upper_ci)."""
    rng = np.random.default_rng(seed)
    point = float(statistic(data))
    samples = np.array([statistic(rng.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return point, float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two groups."""
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def eta_squared(groups: list[np.ndarray]) -> float:
    """Eta-squared effect size for one-way ANOVA."""
    all_data = np.concatenate(groups)
    grand_mean = all_data.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((all_data - grand_mean) ** 2).sum()
    return float(ss_between / max(ss_total, 1e-10))


def partial_correlation(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> tuple[float, float]:
    """Partial correlation of x and y controlling for columns in controls.

    Returns (r, p_value).
    """
    def residuals(target: np.ndarray) -> np.ndarray:
        X = np.column_stack([np.ones(len(controls)), controls])
        beta, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
        return target - X @ beta

    r_x = residuals(x)
    r_y = residuals(y)
    if r_x.std() < 1e-10 or r_y.std() < 1e-10:
        return 0.0, 1.0
    r, p = stats.pearsonr(r_x, r_y)
    return float(r), float(p)


def fdr_correct(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Returns (reject, adjusted_p_values).
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = alpha * np.arange(1, n + 1) / n
    reject_sorted = sorted_p <= thresholds
    # Find last rejection
    last = np.where(reject_sorted)[0]
    if len(last) == 0:
        reject_sorted[:] = False
    else:
        reject_sorted[: last[-1] + 1] = True
    adjusted = np.minimum(sorted_p * n / np.arange(1, n + 1), 1.0)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    reject = np.empty(n, dtype=bool)
    reject[sorted_idx] = reject_sorted
    adj_p = np.empty(n)
    adj_p[sorted_idx] = adjusted
    return reject, adj_p
