"""Chunked causal NumPy sweep engine.

Signals are target portfolio weights at a decision timestamp. They are shifted
by at least one bar before multiplication by realized returns. This convention
is intentionally centralized so discovery code cannot accidentally execute on
the signal bar.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import numpy as np

from edgestack.backtest.costs import CostModel
from edgestack.backtest.metrics import PerformanceMetrics, performance_metrics
from edgestack.models import HypothesisSpec, Session, ensure_fill_after_signal
from edgestack.stats._types import BoolArray, FloatArray
from edgestack.stats.tests import ReturnStatistics, summarize_returns


@dataclass(frozen=True, slots=True)
class EligibleExecution:
    """The first tradable bar and first return earned by a close-derived signal.

    ``fill_index`` identifies the session containing the executable open or
    close.  ``return_index`` identifies the row on which the first realized
    return is recorded.  They are equal for an intraday open-to-close trade.
    For close-to-close and overnight trades the fill is a close, so the first
    return is necessarily recorded on the following session.
    """

    fill_index: int
    return_index: int
    fill_time: datetime


def next_eligible_execution(
    signal_available_at: datetime,
    market_opens: Sequence[datetime],
    market_closes: Sequence[datetime],
    *,
    session: Session,
) -> EligibleExecution:
    """Resolve the first strictly later fill and its first realizable return.

    A daily close-derived signal can enter an intraday convention at the next
    open.  A close-to-close or overnight convention requires a close fill; the
    return ending at that close (and the overnight ending at the next open)
    started before the fill and is therefore ineligible.
    """

    if len(market_opens) != len(market_closes):
        raise ValueError("market open and close timelines must be aligned")
    candidates = market_opens if session is Session.INTRADAY else market_closes
    try:
        fill_index = next(
            index
            for index, candidate in enumerate(candidates)
            if candidate > signal_available_at
        )
    except StopIteration as error:
        raise ValueError("no eligible fill exists after signal availability") from error
    fill_time = candidates[fill_index]
    ensure_fill_after_signal(signal_available_at, fill_time)
    return_index = fill_index if session is Session.INTRADAY else fill_index + 1
    if return_index >= len(candidates):
        raise ValueError("eligible fill has no subsequent realizable return")
    return EligibleExecution(fill_index, return_index, fill_time)


def close_derived_execution_lag(session: Session) -> int:
    """Return the regular-session lag implied by close-derived information.

    Campaign preparation proves each close observation is available before the
    next market open.  Under that invariant the exact timeline resolved by
    :func:`next_eligible_execution` is one row for intraday and two rows for
    close-to-close/overnight returns.
    """

    return 1 if session is Session.INTRADAY else 2


def overlapping_cohort_targets(
    entries: FloatArray, *, holding_period: int
) -> FloatArray:
    """Average the currently active fixed-horizon entry cohorts.

    Cohort ``t`` remains active for exactly ``holding_period`` return intervals.
    The warm-up averages only cohorts that have actually been declared, which
    keeps gross exposure comparable to the mature portfolio without inventing
    pre-sample signals.
    """

    values = np.asarray(entries, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError("entries must be one- or two-dimensional")
    if holding_period < 1:
        raise ValueError("holding_period must be positive")
    finite = np.nan_to_num(values, nan=0.0)
    output = np.empty_like(finite, dtype=float)
    for index in range(len(finite)):
        start = max(0, index - holding_period + 1)
        output[index] = finite[start : index + 1].mean(axis=0)
    return cast(FloatArray, output)


def lag_positions(signal: FloatArray, *, execution_lag: int = 1) -> FloatArray:
    """Shift a signal into executable positions, filling the prefix with zero."""

    values = np.asarray(signal, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError("signal must be one- or two-dimensional")
    if execution_lag < 1:
        raise ValueError("execution_lag must be at least one bar")
    lagged = np.zeros_like(values, dtype=float)
    if execution_lag < len(values):
        lagged[execution_lag:] = values[:-execution_lag]
    return lagged


def aggregate_cross_sectional_returns(
    weights: FloatArray, asset_returns: FloatArray
) -> FloatArray:
    """Aggregate one portfolio observation per date before inference."""

    positions = np.asarray(weights, dtype=float)
    returns = np.asarray(asset_returns, dtype=float)
    if positions.ndim != 2 or positions.shape != returns.shape:
        raise ValueError(
            "weights and asset_returns must have identical (date, asset) shape"
        )
    contributions = np.where(np.isfinite(returns), positions * returns, 0.0)
    return cast(FloatArray, contributions.sum(axis=1))


def vectorized_backtest(
    signal: FloatArray,
    returns: FloatArray,
    *,
    execution_lag: int = 1,
    cost_model: CostModel | None = None,
    asset_type: str | Sequence[str] = "equity",
    order_dollars: float = 10_000.0,
    adv_dollars: float | FloatArray = 100_000_000.0,
    cost_multiplier: float = 1.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return gross, net, and lagged-position streams.

    A one-dimensional signal can be paired with a one-dimensional return. For a
    cross-section both arrays must be date-by-asset and are aggregated to dates.
    Missing asset returns contribute zero only for zero/finite weights; a date
    with all asset returns missing is reported as ``NaN``.
    """

    raw_signal = np.asarray(signal, dtype=float)
    asset_returns = np.asarray(returns, dtype=float)
    if raw_signal.shape != asset_returns.shape:
        raise ValueError("signal and returns must have identical shape")
    positions = lag_positions(
        np.nan_to_num(raw_signal, nan=0.0), execution_lag=execution_lag
    )
    if positions.ndim == 1:
        gross = positions * asset_returns
        all_missing = ~np.isfinite(asset_returns)
        gross = np.where(all_missing, np.nan, gross)
        cost_positions = positions[:, None]
    else:
        gross = aggregate_cross_sectional_returns(positions, asset_returns)
        all_missing = ~np.isfinite(asset_returns).any(axis=1)
        gross[all_missing] = np.nan
        cost_positions = positions
    model = cost_model or CostModel()
    costs = model.portfolio_costs(
        cost_positions,
        asset_type=(
            ("etf" if asset_type.lower() == "etf" else "equity")
            if isinstance(asset_type, str)
            else asset_type
        ),
        order_dollars=order_dollars,
        adv_dollars=adv_dollars,
        multiplier=cost_multiplier,
    )
    net = gross - costs
    net[~np.isfinite(gross)] = np.nan
    return gross, net, positions


def event_window_signal(mask: BoolArray, *, direction: int = 1) -> FloatArray:
    """Convert a Boolean event mask into directed target positions."""

    events = np.asarray(mask)
    if events.ndim != 1:
        raise ValueError("event mask must be one-dimensional")
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    return events.astype(float) * direction


def matched_random_entry_signal(
    signal: FloatArray, *, seed: int, preserve_count: bool = True
) -> FloatArray:
    """Generate a random-entry benchmark with matched exposure count."""

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    rng = np.random.default_rng(seed)
    if preserve_count:
        return values[rng.permutation(len(values))]
    probability = float(np.mean(values != 0.0))
    signs = np.sign(values[values != 0.0])
    sign = int(np.sign(signs.mean())) if signs.size else 1
    return (rng.random(len(values)) < probability).astype(float) * sign


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """One net-of-cost hypothesis result and its causal streams."""

    hypothesis_id: str
    gross_returns: FloatArray
    net_returns: FloatArray
    positions: FloatArray
    return_statistics: ReturnStatistics
    performance: PerformanceMetrics


@dataclass(frozen=True, slots=True)
class SweepData:
    """Aligned returns and precomputed signals for a deterministic sweep."""

    returns: FloatArray
    signals: Mapping[str, FloatArray]
    benchmark_returns: FloatArray | None = None
    adv_dollars: float | FloatArray = 100_000_000.0
    asset_type: str = "equity"


class SweepEngine:
    """Memory-bounded independent NumPy sweep implementation."""

    def __init__(self, *, chunk_size: int = 256, execution_lag: int = 1) -> None:
        if chunk_size < 1 or execution_lag < 1:
            raise ValueError("chunk_size and execution_lag must be positive")
        self.chunk_size = chunk_size
        self.execution_lag = execution_lag

    def run(
        self,
        specs: Iterable[HypothesisSpec],
        data: SweepData,
        costs: CostModel | None = None,
    ) -> list[BacktestResult]:
        """Evaluate specs in bounded batches; missing signal IDs are errors."""

        declarations = list(specs)
        output: list[BacktestResult] = []
        model = costs or CostModel()
        for start in range(0, len(declarations), self.chunk_size):
            for spec in declarations[start : start + self.chunk_size]:
                try:
                    signal = data.signals[spec.hypothesis_id]
                except KeyError as error:
                    raise KeyError(
                        f"missing signal for {spec.hypothesis_id}"
                    ) from error
                gross, net, positions = vectorized_backtest(
                    signal,
                    data.returns,
                    execution_lag=self.execution_lag,
                    cost_model=model,
                    asset_type=data.asset_type,
                    adv_dollars=data.adv_dollars,
                )
                holding = (
                    int(spec.holding_period)
                    if isinstance(spec.holding_period, int)
                    else 1
                )
                stats = summarize_returns(net, holding_period=holding)
                metrics = performance_metrics(
                    net,
                    positions=positions,
                    benchmark=data.benchmark_returns,
                )
                output.append(
                    BacktestResult(
                        spec.hypothesis_id, gross, net, positions, stats, metrics
                    )
                )
        return output


def chunked_signal_sweep(
    signals: FloatArray,
    returns: FloatArray,
    *,
    chunk_size: int = 256,
    execution_lag: int = 1,
) -> FloatArray:
    """Fast date-by-rule sweep for scalar market-return event rules."""

    matrix = np.asarray(signals, dtype=float)
    market = np.asarray(returns, dtype=float)
    if matrix.ndim != 2 or market.ndim != 1 or matrix.shape[0] != market.size:
        raise ValueError("signals must be (dates, rules), returns must be (dates,)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    result = np.empty_like(matrix)
    for start in range(0, matrix.shape[1], chunk_size):
        stop = min(start + chunk_size, matrix.shape[1])
        positions = lag_positions(matrix[:, start:stop], execution_lag=execution_lag)
        result[:, start:stop] = positions * market[:, None]
    return result
