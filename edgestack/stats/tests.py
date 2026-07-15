"""Core return-distribution and Newey-West statistics.

The HAC estimator follows Newey and West (1987) with Bartlett weights. Return
streams must already be aggregated to one portfolio observation per date; this
module intentionally does not pretend stacked stock-date rows are independent.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal, cast

import numpy as np
from scipy import stats as scipy_stats  # type: ignore[import-untyped]

from edgestack.stats._types import FloatArray


def clean_returns(returns: FloatArray | list[float]) -> FloatArray:
    """Return a finite, one-dimensional float64 view."""

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    return cast(FloatArray, values[np.isfinite(values)])


def automatic_newey_west_lag(n_observations: int) -> int:
    """Return the common Andrews/Newey-West automatic lag rule."""

    if n_observations < 1:
        raise ValueError("n_observations must be positive")
    return cast(int, math.floor(4.0 * (n_observations / 100.0) ** (2.0 / 9.0)))


def hac_lag(
    n_observations: int, *, holding_period: int = 1, requested: int | None = None
) -> int:
    """Choose a lag that accounts for overlapping holding-period returns."""

    if holding_period < 1:
        raise ValueError("holding_period must be positive")
    base = automatic_newey_west_lag(n_observations) if requested is None else requested
    if base < 0:
        raise ValueError("requested lag cannot be negative")
    return min(n_observations - 1, max(base, holding_period - 1))


@dataclass(frozen=True, slots=True)
class HACTestResult:
    """HAC inference for a sample mean."""

    n_observations: int
    mean: float
    standard_error: float
    t_stat: float
    p_value: float
    lags: int
    alternative: str

    def as_dict(self) -> dict[str, int | float | str]:
        """Return a serialization-friendly representation."""

        return asdict(self)


def newey_west_long_run_variance(values: FloatArray, lags: int) -> float:
    """Estimate long-run variance with Bartlett-kernel autocovariances."""

    observations = clean_returns(values)
    n = observations.size
    if n < 2:
        return math.nan
    if not 0 <= lags < n:
        raise ValueError("lags must be between zero and n-1")
    centered = observations - observations.mean()
    variance = float(np.dot(centered, centered) / n)
    for lag in range(1, lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * covariance
    # Finite samples can yield a tiny negative estimate from cancellation.
    return max(variance, 0.0)


def hac_mean_test(
    returns: FloatArray | list[float],
    *,
    null_mean: float = 0.0,
    holding_period: int = 1,
    lags: int | None = None,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> HACTestResult:
    """Test a return mean using a Bartlett Newey-West covariance estimate."""

    values = clean_returns(returns)
    n = values.size
    if n < 2:
        return HACTestResult(
            n,
            float(values.mean()) if n else math.nan,
            math.nan,
            math.nan,
            math.nan,
            0,
            alternative,
        )
    selected_lag = hac_lag(n, holding_period=holding_period, requested=lags)
    long_run_variance = newey_west_long_run_variance(values, selected_lag)
    standard_error = math.sqrt(long_run_variance / n)
    difference = float(values.mean() - null_mean)
    if standard_error == 0.0:
        t_stat = math.copysign(math.inf, difference) if difference != 0.0 else 0.0
    else:
        t_stat = difference / standard_error
    if alternative == "two-sided":
        p_value = float(2.0 * scipy_stats.t.sf(abs(t_stat), df=n - 1))
    elif alternative == "greater":
        p_value = float(scipy_stats.t.sf(t_stat, df=n - 1))
    elif alternative == "less":
        p_value = float(scipy_stats.t.cdf(t_stat, df=n - 1))
    else:
        raise ValueError("invalid alternative")
    return HACTestResult(
        n,
        float(values.mean()),
        standard_error,
        float(t_stat),
        p_value,
        selected_lag,
        alternative,
    )


def newey_west_tstat(
    returns: FloatArray | list[float],
    *,
    holding_period: int = 1,
    lags: int | None = None,
) -> float:
    """Return only the two-sided HAC mean t-statistic."""

    return hac_mean_test(returns, holding_period=holding_period, lags=lags).t_stat


def annualized_sharpe(
    returns: FloatArray | list[float], *, periods_per_year: float = 252.0
) -> float:
    """Compute arithmetic annualized Sharpe with zero risk-free rate."""

    values = clean_returns(returns)
    if values.size < 2 or periods_per_year <= 0:
        return math.nan
    volatility = float(values.std(ddof=1))
    if volatility == 0.0:
        return (
            math.copysign(math.inf, float(values.mean())) if values.mean() != 0 else 0.0
        )
    return float(values.mean() / volatility * math.sqrt(periods_per_year))


def hit_rate(returns: FloatArray | list[float]) -> float:
    """Fraction of finite observations strictly above zero."""

    values = clean_returns(returns)
    return float(np.mean(values > 0.0)) if values.size else math.nan


@dataclass(frozen=True, slots=True)
class ReturnStatistics:
    """Standard descriptive and inferential statistics for a return stream."""

    n_observations: int
    mean: float
    standard_deviation: float
    annualized_sharpe: float
    hac_t_stat: float
    hac_p_value: float
    hit_rate: float
    skewness: float
    kurtosis: float
    minimum_sample_pass: bool
    hac_lags: int

    def as_dict(self) -> dict[str, int | float | bool]:
        """Return a serialization-friendly representation."""

        return asdict(self)


def summarize_returns(
    returns: FloatArray | list[float],
    *,
    holding_period: int = 1,
    periods_per_year: float = 252.0,
    minimum_observations: int = 100,
) -> ReturnStatistics:
    """Compute the common EdgeStack discovery statistics."""

    values = clean_returns(returns)
    test = hac_mean_test(values, holding_period=holding_period)
    if values.size >= 3:
        skewness = float(scipy_stats.skew(values, bias=False))
    else:
        skewness = math.nan
    if values.size >= 4:
        # Pearson kurtosis (normal=3) is the convention used in PSR/DSR.
        kurtosis = float(scipy_stats.kurtosis(values, fisher=False, bias=False))
    else:
        kurtosis = math.nan
    return ReturnStatistics(
        n_observations=int(values.size),
        mean=float(values.mean()) if values.size else math.nan,
        standard_deviation=float(values.std(ddof=1)) if values.size >= 2 else math.nan,
        annualized_sharpe=annualized_sharpe(values, periods_per_year=periods_per_year),
        hac_t_stat=test.t_stat,
        hac_p_value=test.p_value,
        hit_rate=hit_rate(values),
        skewness=skewness,
        kurtosis=kurtosis,
        minimum_sample_pass=bool(values.size >= minimum_observations),
        hac_lags=test.lags,
    )
