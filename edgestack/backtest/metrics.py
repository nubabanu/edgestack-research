"""Portfolio and benchmark-relative performance metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from edgestack.stats._types import FloatArray
from edgestack.stats.tests import annualized_sharpe, clean_returns


def equity_curve(
    returns: FloatArray | list[float], *, initial: float = 1.0
) -> FloatArray:
    """Compound simple returns into an equity curve including initial value."""

    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if initial <= 0.0 or np.any(values[np.isfinite(values)] <= -1.0):
        raise ValueError("initial must be positive and returns greater than -100%")
    safe = np.where(np.isfinite(values), values, 0.0)
    return np.concatenate(([initial], initial * np.cumprod(1.0 + safe)))


def cagr(
    returns: FloatArray | list[float], *, periods_per_year: float = 252.0
) -> float:
    """Compute compound annual growth rate."""

    values = clean_returns(returns)
    if values.size == 0 or periods_per_year <= 0.0 or np.any(values <= -1.0):
        return math.nan
    total = float(np.prod(1.0 + values))
    return float(total ** (periods_per_year / values.size) - 1.0)


def annualized_volatility(
    returns: FloatArray | list[float], *, periods_per_year: float = 252.0
) -> float:
    """Compute sample volatility at the requested annualization."""

    values = clean_returns(returns)
    return (
        float(values.std(ddof=1) * math.sqrt(periods_per_year))
        if values.size >= 2
        else math.nan
    )


def sortino_ratio(
    returns: FloatArray | list[float], *, periods_per_year: float = 252.0
) -> float:
    """Compute annualized Sortino using root-mean-square downside return."""

    values = clean_returns(returns)
    if values.size == 0:
        return math.nan
    downside = np.minimum(values, 0.0)
    deviation = float(np.sqrt(np.mean(downside**2)))
    if deviation == 0.0:
        return math.inf if values.mean() > 0.0 else 0.0
    return float(values.mean() / deviation * math.sqrt(periods_per_year))


def max_drawdown(returns: FloatArray | list[float]) -> float:
    """Return the most negative peak-to-trough equity drawdown."""

    curve = equity_curve(np.asarray(returns, dtype=float))
    peaks = np.maximum.accumulate(curve)
    return float(np.min(curve / peaks - 1.0))


def turnover(positions: FloatArray) -> float:
    """Mean one-way portfolio turnover from target weights."""

    weights = np.asarray(positions, dtype=float)
    if weights.ndim == 1:
        weights = weights[:, None]
    if weights.ndim != 2 or len(weights) < 2:
        return 0.0
    return float(np.mean(np.sum(np.abs(np.diff(weights, axis=0)), axis=1) / 2.0))


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Absolute and optional benchmark-relative metrics."""

    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    exposure: float
    turnover: float
    beta: float | None = None
    alpha_annualized: float | None = None
    information_ratio: float | None = None
    correlation: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        """Return a serialization-friendly representation."""

        return asdict(self)


def performance_metrics(
    returns: FloatArray | list[float],
    *,
    positions: FloatArray | None = None,
    benchmark: FloatArray | list[float] | None = None,
    periods_per_year: float = 252.0,
) -> PerformanceMetrics:
    """Compute comprehensive performance statistics on aligned returns."""

    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    finite = np.isfinite(values)
    clean = values[finite]
    total = float(np.prod(1.0 + clean) - 1.0) if clean.size else math.nan
    exposure_value = (
        float(np.nanmean(np.abs(np.asarray(positions, dtype=float))))
        if positions is not None
        else 1.0
    )
    turnover_value = (
        turnover(np.asarray(positions, dtype=float)) if positions is not None else 0.0
    )
    beta = alpha = information = correlation = None
    if benchmark is not None:
        reference = np.asarray(benchmark, dtype=float)
        if reference.shape != values.shape:
            raise ValueError("benchmark must be aligned with returns")
        aligned = np.isfinite(values) & np.isfinite(reference)
        strategy_sample = values[aligned]
        benchmark_sample = reference[aligned]
        if strategy_sample.size >= 2:
            benchmark_variance = float(np.var(benchmark_sample, ddof=1))
            covariance = float(np.cov(strategy_sample, benchmark_sample, ddof=1)[0, 1])
            beta = (
                covariance / benchmark_variance
                if benchmark_variance > 0.0
                else math.nan
            )
            alpha = float(
                (strategy_sample.mean() - beta * benchmark_sample.mean())
                * periods_per_year
            )
            active = strategy_sample - benchmark_sample
            information = annualized_sharpe(active, periods_per_year=periods_per_year)
            correlation = float(np.corrcoef(strategy_sample, benchmark_sample)[0, 1])
    return PerformanceMetrics(
        total_return=total,
        cagr=cagr(clean, periods_per_year=periods_per_year),
        annualized_volatility=annualized_volatility(
            clean, periods_per_year=periods_per_year
        ),
        sharpe=annualized_sharpe(clean, periods_per_year=periods_per_year),
        sortino=sortino_ratio(clean, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(clean) if clean.size else math.nan,
        exposure=exposure_value,
        turnover=turnover_value,
        beta=beta,
        alpha_annualized=alpha,
        information_ratio=information,
        correlation=correlation,
    )
