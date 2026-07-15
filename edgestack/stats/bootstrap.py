"""Deterministic block and stationary bootstrap confidence intervals."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from edgestack.stats._types import FloatArray, IntArray
from edgestack.stats.tests import annualized_sharpe, clean_returns

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


bootstrap_confidence_interval = stationary_bootstrap_ci
