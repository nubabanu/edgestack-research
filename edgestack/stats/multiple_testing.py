"""Global multiple-testing correction and discovery gauntlets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edgestack.stats._types import BoolArray, FloatArray, IntArray


def _pvalues(p_values: FloatArray | list[float]) -> FloatArray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and in [0, 1]")
    return values


@dataclass(frozen=True, slots=True)
class MultipleTestingResult:
    """Adjusted p-values and rejection decisions in original order."""

    adjusted_p_values: FloatArray
    reject: BoolArray
    alpha: float
    method: str
    critical_p_value: float | None


def benjamini_hochberg(
    p_values: FloatArray | list[float], *, q: float = 0.05
) -> MultipleTestingResult:
    """Control the Benjamini-Hochberg false-discovery rate globally."""

    values = _pvalues(p_values)
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly between zero and one")
    m = values.size
    if m == 0:
        return MultipleTestingResult(
            np.array([], dtype=float), np.array([], dtype=bool), q, "BH", None
        )
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.arange(1, m + 1, dtype=float)
    raw_adjusted = sorted_values * m / ranks
    sorted_adjusted = np.minimum.accumulate(raw_adjusted[::-1])[::-1].clip(0.0, 1.0)
    adjusted = np.empty_like(sorted_adjusted)
    adjusted[order] = sorted_adjusted
    thresholds = ranks / m * q
    passing = np.flatnonzero(sorted_values <= thresholds)
    critical = float(sorted_values[passing[-1]]) if passing.size else None
    reject = adjusted <= q
    return MultipleTestingResult(adjusted, reject, q, "BH", critical)


def bonferroni(
    p_values: FloatArray | list[float], *, alpha: float = 0.05
) -> MultipleTestingResult:
    """Apply strong family-wise Bonferroni error control."""

    values = _pvalues(p_values)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    m = values.size
    adjusted = np.minimum(values * max(m, 1), 1.0)
    threshold = alpha / m if m else None
    reject = adjusted <= alpha
    return MultipleTestingResult(adjusted, reject, alpha, "BONFERRONI", threshold)


@dataclass(frozen=True, slots=True)
class DiscoveryGauntlet:
    """Vectorized hard-gate decisions for all real and placebo trials."""

    minimum_sample: BoolArray
    directed_positive: BoolArray
    t_gate: BoolArray
    fdr_gate: BoolArray
    dsr_gate: BoolArray
    survivors: BoolArray
    adjusted_p_values: FloatArray


def discovery_gauntlet(
    *,
    sample_sizes: IntArray,
    directed_means: FloatArray,
    t_statistics: FloatArray,
    p_values: FloatArray,
    dsr_probabilities: FloatArray,
    minimum_observations: int = 100,
    t_threshold: float = 3.0,
    fdr_q: float = 0.05,
    dsr_probability: float = 0.95,
) -> DiscoveryGauntlet:
    """Apply the preregistered discovery filters without dropping trials.

    Ineligible/underpowered trials stay in the global family as p=1 so their
    registration cannot silently shrink the multiple-testing denominator.
    """

    sizes = np.asarray(sample_sizes)
    means = np.asarray(directed_means, dtype=float)
    t_stats = np.asarray(t_statistics, dtype=float)
    pvals = np.asarray(p_values, dtype=float)
    dsr = np.asarray(dsr_probabilities, dtype=float)
    if (
        not (sizes.shape == means.shape == t_stats.shape == pvals.shape == dsr.shape)
        or sizes.ndim != 1
    ):
        raise ValueError("all gauntlet arrays must have the same one-dimensional shape")
    minimum = sizes >= minimum_observations
    positive = means > 0.0
    t_gate = t_stats > t_threshold
    eligible_p = np.where(minimum & positive & np.isfinite(pvals), pvals, 1.0)
    fdr = benjamini_hochberg(eligible_p, q=fdr_q)
    dsr_gate = dsr > dsr_probability
    survivors = minimum & positive & t_gate & fdr.reject & dsr_gate
    return DiscoveryGauntlet(
        minimum,
        positive,
        t_gate,
        fdr.reject,
        dsr_gate,
        survivors,
        fdr.adjusted_p_values,
    )


fdr_bh = benjamini_hochberg
harvey_liu_zhu_gate = discovery_gauntlet
