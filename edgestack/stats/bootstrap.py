"""Deterministic block and stationary bootstrap confidence intervals."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from edgestack.stats._types import FloatArray, IntArray
from edgestack.stats.tests import (
    annualized_sharpe,
    clean_returns,
    hac_lag,
    newey_west_long_run_variance,
)

Statistic = Literal["mean", "sharpe"] | Callable[[FloatArray], float]


def stationary_bootstrap_indices(
    n_observations: int,
    n_resamples: int,
    *,
    average_block_length: float = 10.0,
    seed: int = 0,
) -> IntArray:
    """Draw Politis-Romano stationary-bootstrap index paths.

    The same returned matrix can be shared across every strategy in a family,
    preserving their cross-sectional dependence while reducing compute cost.
    """

    if n_observations < 1 or n_resamples < 1:
        raise ValueError("n_observations and n_resamples must be positive")
    if average_block_length < 1.0:
        raise ValueError("average_block_length must be at least one")
    rng = np.random.default_rng(seed)
    indices = np.empty((n_resamples, n_observations), dtype=np.int64)
    indices[:, 0] = rng.integers(0, n_observations, size=n_resamples)
    restart_probability = 1.0 / average_block_length
    for column in range(1, n_observations):
        restart = rng.random(n_resamples) < restart_probability
        continuation = (indices[:, column - 1] + 1) % n_observations
        fresh = rng.integers(0, n_observations, size=n_resamples)
        indices[:, column] = np.where(restart, fresh, continuation)
    return indices


def moving_block_bootstrap_indices(
    n_observations: int,
    n_resamples: int,
    *,
    block_length: int = 10,
    seed: int = 0,
) -> IntArray:
    """Draw circular moving-block bootstrap index paths."""

    if n_observations < 1 or n_resamples < 1 or block_length < 1:
        raise ValueError("sizes and block_length must be positive")
    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(n_observations / block_length)
    starts = rng.integers(0, n_observations, size=(n_resamples, blocks_needed))
    offsets = np.arange(block_length, dtype=np.int64)
    paths = (starts[..., None] + offsets) % n_observations
    return paths.reshape(n_resamples, -1)[:, :n_observations]


def _statistic_function(
    statistic: Statistic, periods_per_year: float
) -> Callable[[FloatArray], float]:
    if statistic == "mean":
        return lambda sample: float(np.mean(sample))
    if statistic == "sharpe":
        return lambda sample: annualized_sharpe(
            sample, periods_per_year=periods_per_year
        )
    if callable(statistic):
        return statistic
    raise ValueError("statistic must be 'mean', 'sharpe', or callable")


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Percentile stationary-bootstrap estimate and confidence interval."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    average_block_length: float


@dataclass(frozen=True, slots=True)
class StudentizedSharpeCI:
    """Studentized stationary-bootstrap Sharpe (or Sharpe-difference) CI."""

    estimate: float
    lower: float
    upper: float
    standard_error: float
    p_value_two_sided: float
    confidence: float
    n_resamples: int
    average_block_length: float
    benchmark_included: bool
    method: str = "LEDOIT_WOLF_STUDENTIZED_STATIONARY"


def stationary_bootstrap_ci(
    values: FloatArray | list[float],
    *,
    statistic: Statistic = "mean",
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    average_block_length: float = 10.0,
    periods_per_year: float = 252.0,
    seed: int = 0,
    indices: IntArray | None = None,
) -> BootstrapCI:
    """Estimate a percentile CI using the stationary bootstrap."""

    sample = clean_returns(values)
    if sample.size < 2:
        raise ValueError("at least two finite observations are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    draws = indices
    if draws is None:
        draws = stationary_bootstrap_indices(
            sample.size,
            n_resamples,
            average_block_length=average_block_length,
            seed=seed,
        )
    else:
        draws = np.asarray(draws, dtype=np.int64)
        if draws.ndim != 2 or draws.shape[1] != sample.size:
            raise ValueError("shared indices must have shape (resamples, observations)")
        if np.any((draws < 0) | (draws >= sample.size)):
            raise ValueError("bootstrap index out of bounds")
        n_resamples = draws.shape[0]
    function = _statistic_function(statistic, periods_per_year)
    estimates = np.fromiter(
        (function(sample[path]) for path in draws), dtype=float, count=len(draws)
    )
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        raise ValueError("bootstrap statistic was non-finite for every resample")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return BootstrapCI(
        estimate=function(sample),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        n_resamples=len(draws),
        average_block_length=average_block_length,
    )


def studentized_sharpe_ci(
    values: FloatArray | list[float],
    *,
    benchmark: FloatArray | list[float] | None = None,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    average_block_length: float = 10.0,
    periods_per_year: float = 252.0,
    holding_period: int = 1,
    seed: int = 0,
) -> StudentizedSharpeCI:
    """Construct a studentized time-series bootstrap Sharpe interval.

    When ``benchmark`` is supplied the estimand is the strategy Sharpe minus
    the benchmark Sharpe and the same bootstrap dates are used for both.  The
    studentizer is a Newey-West long-run standard error of the Sharpe influence
    function, making the interval robust to non-normality and serial dependence
    in the sense advocated by Ledoit and Wolf (2008).
    """

    strategy = np.asarray(values, dtype=float)
    if strategy.ndim != 1:
        raise ValueError("values must be one-dimensional")
    reference: np.ndarray[Any, np.dtype[np.float64]] | None = None
    finite = np.isfinite(strategy)
    if benchmark is not None:
        reference = np.asarray(benchmark, dtype=float)
        if reference.ndim != 1 or reference.shape != strategy.shape:
            raise ValueError("benchmark must be one-dimensional and aligned")
        finite &= np.isfinite(reference)
        reference = reference[finite]
    strategy = strategy[finite]
    if strategy.size < 3:
        raise ValueError("at least three aligned finite observations are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if n_resamples < 1 or average_block_length < 1.0:
        raise ValueError("resamples and average_block_length must be positive")
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")

    estimate, influence = _sharpe_influence(
        strategy[None, :], periods_per_year=periods_per_year
    )
    if reference is not None:
        reference_estimate, reference_influence = _sharpe_influence(
            reference[None, :], periods_per_year=periods_per_year
        )
        estimate = estimate - reference_estimate
        influence = influence - reference_influence
    estimate_value = float(estimate[0])
    selected_lag = hac_lag(strategy.size, holding_period=holding_period)
    long_run_variance = newey_west_long_run_variance(influence[0], selected_lag)
    standard_error = math.sqrt(long_run_variance / strategy.size)
    if not math.isfinite(standard_error) or standard_error <= 0.0:
        raise ValueError("Sharpe influence function has no positive long-run variance")

    studentized = np.empty(n_resamples, dtype=float)
    canonical_batch = 32
    for start in range(0, n_resamples, canonical_batch):
        stop = min(start + canonical_batch, n_resamples)
        sequence = np.random.SeedSequence([seed, start])
        batch_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
        indices = stationary_bootstrap_indices(
            strategy.size,
            stop - start,
            average_block_length=average_block_length,
            seed=batch_seed,
        )
        bootstrap_estimate, bootstrap_influence = _sharpe_influence(
            strategy[indices], periods_per_year=periods_per_year
        )
        if reference is not None:
            reference_estimate, reference_influence = _sharpe_influence(
                reference[indices], periods_per_year=periods_per_year
            )
            bootstrap_estimate -= reference_estimate
            bootstrap_influence -= reference_influence
        bootstrap_se = _row_hac_standard_errors(bootstrap_influence, selected_lag)
        studentized[start:stop] = np.divide(
            bootstrap_estimate - estimate_value,
            bootstrap_se,
            out=np.full(stop - start, np.nan),
            where=bootstrap_se > 0.0,
        )
    finite_studentized = studentized[np.isfinite(studentized)]
    if not finite_studentized.size:
        raise ValueError("studentized Sharpe statistic was non-finite in every draw")
    alpha = (1.0 - confidence) / 2.0
    lower_quantile, upper_quantile = np.quantile(
        finite_studentized, [alpha, 1.0 - alpha]
    )
    lower = estimate_value - float(upper_quantile) * standard_error
    upper = estimate_value - float(lower_quantile) * standard_error
    observed_t = estimate_value / standard_error
    p_value = (
        1.0 + np.count_nonzero(np.abs(finite_studentized) >= abs(observed_t))
    ) / (finite_studentized.size + 1.0)
    return StudentizedSharpeCI(
        estimate_value,
        lower,
        upper,
        standard_error,
        float(p_value),
        confidence,
        n_resamples,
        average_block_length,
        reference is not None,
    )


def _sharpe_influence(
    samples: np.ndarray[Any, np.dtype[np.float64]],
    *,
    periods_per_year: float,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """Return row-wise Sharpe estimates and asymptotic influence values."""

    matrix = np.asarray(samples, dtype=float)
    means = matrix.mean(axis=1)
    centered = matrix - means[:, None]
    population_variance = np.mean(centered**2, axis=1)
    population_deviation = np.sqrt(population_variance)
    sample_deviation = matrix.std(axis=1, ddof=1)
    scale = math.sqrt(periods_per_year)
    estimates = np.divide(
        means * scale,
        sample_deviation,
        out=np.full(matrix.shape[0], np.nan),
        where=sample_deviation > 0.0,
    )
    influence = np.divide(
        centered,
        population_deviation[:, None],
        out=np.full_like(centered, np.nan),
        where=population_deviation[:, None] > 0.0,
    )
    second_term = np.divide(
        means[:, None] * (centered**2 - population_variance[:, None]),
        2.0 * population_deviation[:, None] ** 3,
        out=np.full_like(centered, np.nan),
        where=population_deviation[:, None] > 0.0,
    )
    return estimates, scale * (influence - second_term)


def _row_hac_standard_errors(
    influence: np.ndarray[Any, np.dtype[np.float64]], lags: int
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Vectorized Bartlett HAC standard errors for row-wise influence paths."""

    values = np.asarray(influence, dtype=float)
    centered = values - values.mean(axis=1, keepdims=True)
    n_observations = values.shape[1]
    long_run = np.mean(centered**2, axis=1)
    for lag in range(1, lags + 1):
        covariance = np.mean(centered[:, lag:] * centered[:, :-lag], axis=1)
        long_run += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    return np.asarray(
        np.sqrt(np.maximum(long_run, 0.0) / n_observations), dtype=np.float64
    )


bootstrap_confidence_interval = stationary_bootstrap_ci
