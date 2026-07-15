"""Global multiple-testing correction and discovery gauntlets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from edgestack.stats._types import BoolArray, FloatArray, IntArray
from edgestack.stats.bootstrap import stationary_bootstrap_indices
from edgestack.stats.tests import hac_mean_test


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


def romano_wolf_stepdown(
    strategy_returns: FloatArray,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    average_block_length: float = 10.0,
    seed: int = 0,
) -> MultipleTestingResult:
    """Apply a studentized Romano-Wolf max-t stepdown procedure.

    Columns are hypotheses and rows are aligned date observations.  The same
    stationary-bootstrap dates are used for every column, preserving the
    dependence that makes closely related trading rules a single effective
    search family.  Returns are recentered under the joint null and each
    column is studentized by its HAC standard error.
    """

    values = np.asarray(strategy_returns, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("strategy_returns must be (dates, hypotheses)")
    if np.any(~np.isfinite(values)):
        raise ValueError("Romano-Wolf inputs must be finite and aligned")
    if values.shape[0] < 2:
        raise ValueError("Romano-Wolf requires at least two dates")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    n_dates, n_hypotheses = values.shape
    if n_hypotheses == 0:
        return MultipleTestingResult(
            np.array([], dtype=float),
            np.array([], dtype=bool),
            alpha,
            "ROMANO_WOLF_STATIONARY_STEPDOWN",
            None,
        )

    tests = [hac_mean_test(values[:, column]) for column in range(n_hypotheses)]
    observed = np.asarray([test.t_stat for test in tests], dtype=float)
    standard_errors = np.asarray([test.standard_error for test in tests], dtype=float)
    means = values.mean(axis=0)
    valid = np.isfinite(standard_errors) & (standard_errors > 0.0)
    observed = np.where(
        valid,
        observed,
        np.where(means > 0.0, math.inf, np.where(means < 0.0, -math.inf, 0.0)),
    )
    draws = stationary_bootstrap_indices(
        n_dates,
        n_bootstrap,
        average_block_length=average_block_length,
        seed=seed,
    )
    centered = values - means
    bootstrap_means = centered[draws].mean(axis=1)
    bootstrap_t = np.divide(
        bootstrap_means,
        standard_errors,
        out=np.zeros_like(bootstrap_means),
        where=valid,
    )
    order = np.argsort(-observed, kind="stable")
    ordered_bootstrap = bootstrap_t[:, order]
    max_remaining = np.maximum.accumulate(ordered_bootstrap[:, ::-1], axis=1)[:, ::-1]
    raw_ordered = (
        1.0 + np.count_nonzero(max_remaining >= observed[order][None, :], axis=0)
    ) / (n_bootstrap + 1.0)
    adjusted_ordered = np.maximum.accumulate(raw_ordered).clip(0.0, 1.0)
    adjusted = np.empty(n_hypotheses, dtype=float)
    adjusted[order] = adjusted_ordered
    reject = adjusted <= alpha
    critical = float(np.max(adjusted[reject])) if np.any(reject) else None
    return MultipleTestingResult(
        adjusted,
        reject,
        alpha,
        "ROMANO_WOLF_STATIONARY_STEPDOWN",
        critical,
    )


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
    t_threshold: float | FloatArray = 3.0,
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
    thresholds = np.asarray(t_threshold, dtype=float)
    if thresholds.ndim == 0:
        thresholds = np.full(t_stats.shape, float(thresholds))
    if thresholds.shape != t_stats.shape or np.any(~np.isfinite(thresholds)):
        raise ValueError("t_threshold must be finite and scalar or aligned")
    t_gate = t_stats > thresholds
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
