"""Probabilistic and Deflated Sharpe Ratios.

Implements Bailey and Lopez de Prado (2014). Input Sharpe ratios must use the
same periodicity as the per-observation skew/kurtosis and sample count; callers
should normally pass an unannualized Sharpe.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

from edgestack.stats._types import FloatArray


def sharpe_standard_error(
    sharpe: float,
    n_observations: int,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Estimate the standard error of a sample Sharpe ratio."""

    if n_observations < 2:
        return math.nan
    variance_term = 1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
    if variance_term <= 0.0:
        return math.nan
    return math.sqrt(variance_term / (n_observations - 1.0))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    *,
    benchmark_sharpe: float = 0.0,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Return ``P(true Sharpe > benchmark Sharpe)`` under PSR asymptotics."""

    standard_error = sharpe_standard_error(
        observed_sharpe,
        n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
    )
    if not math.isfinite(standard_error) or standard_error == 0.0:
        if observed_sharpe == benchmark_sharpe:
            return 0.5
        return float(observed_sharpe > benchmark_sharpe)
    return float(norm.cdf((observed_sharpe - benchmark_sharpe) / standard_error))


def expected_maximum_sharpe(
    *,
    n_trials: int,
    trial_sharpe_std: float,
    trial_sharpe_mean: float = 0.0,
) -> float:
    """Approximate the expected maximum Sharpe over independent trials."""

    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    if trial_sharpe_std < 0.0:
        raise ValueError("trial_sharpe_std cannot be negative")
    if n_trials == 1 or trial_sharpe_std == 0.0:
        return trial_sharpe_mean
    euler_mascheroni = 0.5772156649015329
    first = norm.ppf(1.0 - 1.0 / n_trials)
    second = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(
        trial_sharpe_mean
        + trial_sharpe_std
        * ((1.0 - euler_mascheroni) * first + euler_mascheroni * second)
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_observations: int,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    trial_sharpes: FloatArray | list[float] | None = None,
    trial_sharpe_std: float | None = None,
    trial_sharpe_mean: float = 0.0,
) -> float:
    """Return PSR against the expected best Sharpe from multiple trials."""

    if trial_sharpes is not None:
        trials = np.asarray(trial_sharpes, dtype=float)
        trials = trials[np.isfinite(trials)]
        if trials.size > 1:
            trial_sharpe_std = float(trials.std(ddof=1))
            trial_sharpe_mean = float(trials.mean())
        elif trials.size == 1:
            trial_sharpe_std = 0.0
            trial_sharpe_mean = float(trials[0])
        else:
            raise ValueError("trial_sharpes contains no finite observations")
    if trial_sharpe_std is None:
        # Under iid Gaussian null returns, standard error of the unannualized
        # Sharpe is approximately 1/sqrt(T-1).
        if n_observations < 2:
            return math.nan
        trial_sharpe_std = 1.0 / math.sqrt(n_observations - 1.0)
    benchmark = expected_maximum_sharpe(
        n_trials=n_trials,
        trial_sharpe_std=trial_sharpe_std,
        trial_sharpe_mean=trial_sharpe_mean,
    )
    return probabilistic_sharpe_ratio(
        observed_sharpe,
        benchmark_sharpe=benchmark,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
    )


probabilistic_sharpe = probabilistic_sharpe_ratio
deflated_sharpe = deflated_sharpe_ratio
