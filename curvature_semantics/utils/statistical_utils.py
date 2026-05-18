"""Bootstrap CIs, effect size, partial correlation, and FDR correction."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def _normal_two_sided_p(z: float) -> float:
    """Approximate a two-sided p-value from a standard-normal z score."""
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Dependency-free Pearson correlation with an approximate p-value."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0, 1.0
    r = float(np.corrcoef(x, y)[0, 1])
    r = max(-1.0, min(1.0, r))
    if n < 4 or abs(r) >= 1.0:
        p = 0.0 if abs(r) >= 1.0 else 1.0
    else:
        z = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(n - 3)
        p = _normal_two_sided_p(z)
    return r, p


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks for a 1D array, handling ties."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Dependency-free Spearman correlation with an approximate p-value."""
    return pearsonr(_rankdata(np.asarray(x, dtype=float)), _rankdata(np.asarray(y, dtype=float)))


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
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
    """Partial correlation of x and y controlling for columns in controls."""

    def residuals(target: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(len(controls)), controls])
        beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        return target - design @ beta

    r_x = residuals(np.asarray(x, dtype=float))
    r_y = residuals(np.asarray(y, dtype=float))
    return pearsonr(r_x, r_y)


def fdr_correct(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Returns (reject, adjusted_p_values).
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = alpha * np.arange(1, n + 1) / n
    reject_sorted = sorted_p <= thresholds
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
