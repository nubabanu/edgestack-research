"""Real Zipline finalist confirmation over canonical adjusted daily bars.

The adapter intentionally uses Zipline's asset database, data portal, blotter,
order API, and performance loop.  It does not turn a tested return stream into
synthetic prices.  Canonical adjusted OHLCV matrices are supplied directly, so
split/dividend effects already embedded by the provider are represented once
and no second corporate-action adjustment is applied.

Daily Zipline orders fill at a session close.  Consequently, close-to-close and
overnight conventions can reproduce their exact close fill timestamps.  An
intraday next-open finalist still executes through Zipline, but its timestamp
comparison must fail closed because daily bars cannot prove an opening-auction
fill.  Suitable minute/auction data would be required to promote it.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from edgestack.backtest.confirm import ConfirmationResult
from edgestack.backtest.costs import CostModel
from edgestack.backtest.engine import BacktestResult
from edgestack.models import HypothesisSpec, Session


@dataclass(frozen=True, slots=True)
class ZiplineCanonicalData:
    """Canonical bar, target, and frozen-cost inputs for one finalist."""

    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    target_positions: np.ndarray[Any, np.dtype[np.float64]]
    asset_returns: np.ndarray[Any, np.dtype[np.float64]]
    cost_positions: np.ndarray[Any, np.dtype[np.float64]]
    market_open: tuple[pd.Timestamp, ...]
    market_close: tuple[pd.Timestamp, ...]
    adv_dollars: float | np.ndarray[Any, np.dtype[np.float64]]
    asset_type: str | tuple[str, ...]


class _CompatibleInMemoryDailyBarReader:
    """Construct Zipline's reader with two 3.1.1 compatibility repairs.

    ``zipline-reloaded==3.1.1`` sets ``_frames`` while two reader methods access
    ``frames``, and its ``last_available_dt`` assumes an older subscriptable
    calendar.  The tiny runtime subclass keeps all storage/array loading in the
    upstream reader and only repairs those accessors.
    """

    @staticmethod
    def build(
        frames: dict[str, pd.DataFrame],
        calendar: Any,
        currency_codes: pd.Series,
    ) -> Any:
        """Return an upstream in-memory reader with stable accessors."""

        from zipline.data.in_memory_daily_bars import (  # type: ignore[import-untyped]
            InMemoryDailyBarReader,
        )

        class Reader(InMemoryDailyBarReader):  # type: ignore[misc]
            @property
            def frames(self) -> dict[str, pd.DataFrame]:
                return cast(dict[str, pd.DataFrame], self._frames)

            @property
            def last_available_dt(self) -> pd.Timestamp:
                return self._sessions[-1]  # type: ignore[no-any-return]

            def get_value(self, sid: Any, dt: Any, field: str) -> float:
                key = getattr(sid, "sid", sid)
                return float(self._frames[field].loc[dt, key])

            def get_last_traded_dt(self, asset: Any, dt: Any) -> pd.Timestamp:
                key = getattr(asset, "sid", asset)
                values = self._frames["close"].loc[:dt, key]
                return pd.Timestamp(values.last_valid_index())

        return Reader(frames, calendar, currency_codes)


def confirm_with_zipline(
    spec: HypothesisSpec,
    data: ZiplineCanonicalData,
    vector_result: BacktestResult,
    cost_model: CostModel,
    *,
    tolerance_bps_per_trade: float = 1.0,
    capital_base: float = 1_000_000_000.0,
) -> ConfirmationResult:
    """Execute target changes in Zipline and compare fills and net mean.

    A single pre-sample calendar session is used only as an order-submission
    warm-up.  Its price is a duplicate of the first canonical close, no position
    is held during it, and it is excluded from every return comparison.  This
    lets a target whose first eligible fill is the first canonical close be sent
    through Zipline's normal next-daily-bar fill path.
    """

    _validate(data, vector_result, tolerance_bps_per_trade, capital_base)
    # Imports remain local so the core package and ordinary tests do not require
    # the optional confirmation extra.
    import exchange_calendars  # type: ignore[import-untyped]
    from sqlalchemy import create_engine
    from zipline.algorithm import TradingAlgorithm  # type: ignore[import-untyped]
    from zipline.assets import (  # type: ignore[import-untyped]
        AssetDBWriter,
        AssetFinder,
    )
    from zipline.data.data_portal import DataPortal  # type: ignore[import-untyped]
    from zipline.finance import commission, slippage  # type: ignore[import-untyped]
    from zipline.finance.trading import (  # type: ignore[import-untyped]
        SimulationParameters,
    )

    dates = pd.DatetimeIndex(data.dates).tz_localize(None).normalize()
    calendar = exchange_calendars.get_calendar(
        "XNYS",
        start=dates[0] - pd.DateOffset(years=1),
        end=dates[-1] + pd.DateOffset(years=1),
        side="right",
    )
    expected_sessions = calendar.sessions_in_range(dates[0], dates[-1])
    if not dates.equals(expected_sessions):
        raise ValueError(
            "canonical sessions do not exactly match Zipline's pinned XNYS calendar"
        )
    padding = pd.Timestamp(calendar.previous_session(dates[0]))
    frame_dates = pd.DatetimeIndex([padding, *dates])
    sids = tuple(range(len(data.symbols)))
    frames = {
        field: _with_padding(getattr(data, field), frame_dates, sids)
        for field in ("open", "high", "low", "close", "volume")
    }

    engine = create_engine("sqlite://")
    equities = pd.DataFrame(
        {
            "symbol": list(data.symbols),
            "asset_name": list(data.symbols),
            "start_date": padding,
            "end_date": dates[-1],
            "first_traded": padding,
            "auto_close_date": dates[-1] + pd.Timedelta(days=10),
            "exchange": "XNYS",
        },
        index=sids,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        AssetDBWriter(engine).write(equities=equities)
    finder = AssetFinder(engine)
    assets = tuple(finder.retrieve_all(sids))
    reader = _CompatibleInMemoryDailyBarReader.build(
        frames,
        calendar,
        pd.Series("USD", index=pd.Index(sids, dtype=int)),
    )
    portal = DataPortal(
        finder,
        calendar,
        padding,
        equity_daily_reader=reader,
        last_available_session=dates[-1],
    )
    simulation = SimulationParameters(
        padding,
        dates[-1],
        calendar,
        capital_base=capital_base,
        emission_rate="daily",
        data_frequency="daily",
    )
    targets = np.asarray(data.target_positions, dtype=float)

    def initialize(context: Any) -> None:
        context.assets = assets
        context.row = 0
        context.previous_target = np.zeros(targets.shape[1], dtype=float)
        context.set_commission(us_equities=commission.NoCommission())
        context.set_slippage(us_equities=slippage.NoSlippage())

    def handle_data(context: Any, bar_data: Any) -> None:
        # Frame row 0 is the warm-up session. Its order fills at canonical row 0
        # and establishes target_positions[1] for the following return interval.
        target_row = context.row + 1
        if target_row < len(targets):
            desired = np.nan_to_num(targets[target_row], nan=0.0)
            changed = ~np.isclose(
                desired,
                context.previous_target,
                rtol=0.0,
                atol=1e-12,
            )
            for column in np.flatnonzero(changed):
                asset = context.assets[int(column)]
                if not bar_data.can_trade(asset):
                    continue
                context.order_target_percent(asset, float(desired[column]))
            context.previous_target = desired
        context.row += 1

    benchmark = pd.Series(0.0, index=frame_dates.tz_localize("UTC"))
    algorithm = TradingAlgorithm(
        sim_params=simulation,
        data_portal=portal,
        asset_finder=finder,
        initialize=initialize,
        handle_data=handle_data,
        trading_calendar=calendar,
        benchmark_returns=benchmark,
    )
    try:
        performance = algorithm.run()
    finally:
        finder.close()
        engine.dispose()

    expected_times = _expected_fill_times(data, spec.session)
    actual_times = _transaction_times(performance)
    timestamps_match = actual_times == expected_times
    transaction_count = len(actual_times)
    expected_count = len(expected_times)
    actual_positions = _zipline_return_positions(
        performance,
        dates,
        data.close,
        sids,
    )
    actual_gross = _portfolio_returns(actual_positions, data.asset_returns)
    costs = _frozen_costs(data, cost_model)
    actual_net = actual_gross - costs
    actual_net[~np.isfinite(actual_gross)] = np.nan
    net_mean = (
        float(np.nanmean(actual_net)) if np.isfinite(actual_net).any() else math.nan
    )
    vector_mean = float(np.nanmean(vector_result.net_returns))
    denominator = max(transaction_count, expected_count, 1)
    difference = (
        abs(net_mean - vector_mean) * 10_000.0 * len(dates) / denominator
        if math.isfinite(net_mean) and math.isfinite(vector_mean)
        else math.inf
    )
    convention_supported = spec.session is not Session.INTRADAY
    count_match = transaction_count == expected_count
    passed = bool(
        convention_supported
        and count_match
        and timestamps_match
        and difference <= tolerance_bps_per_trade
    )
    reasons: list[str] = []
    if not convention_supported:
        reasons.append("daily Zipline bars cannot verify a next-open intraday fill")
    if not count_match:
        reasons.append(
            f"transaction count differs ({transaction_count} vs {expected_count})"
        )
    if not timestamps_match:
        reasons.append("transaction timestamps differ")
    if difference > tolerance_bps_per_trade:
        reasons.append(f"net mean differs by {difference:.6g} bps per transaction")
    return ConfirmationResult(
        spec.hypothesis_id,
        transaction_count,
        expected_count,
        net_mean,
        vector_mean,
        difference,
        timestamps_match,
        passed,
        backend="zipline-reloaded-3.1.1-in-memory-adjusted-ohlcv",
        reason="agreement within tolerance" if passed else "; ".join(reasons),
    )


def _validate(
    data: ZiplineCanonicalData,
    vector_result: BacktestResult,
    tolerance: float,
    capital: float,
) -> None:
    dates = pd.DatetimeIndex(data.dates)
    if dates.empty or not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError("Zipline confirmation dates must be non-empty, sorted, unique")
    if tolerance < 0.0 or not math.isfinite(capital) or capital <= 0.0:
        raise ValueError("confirmation tolerance/capital is invalid")
    columns = list(data.symbols)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Zipline confirmation symbols must be unique and non-empty")
    for name in ("open", "high", "low", "close", "volume"):
        frame = getattr(data, name)
        if not frame.index.equals(dates) or list(map(str, frame.columns)) != columns:
            raise ValueError(f"canonical {name} is not aligned to dates/symbols")
    target = np.asarray(data.target_positions, dtype=float)
    returns = np.asarray(data.asset_returns, dtype=float)
    if target.shape != returns.shape or target.shape != (len(dates), len(columns)):
        raise ValueError("Zipline targets and asset returns must be date-by-asset")
    if bool(np.any(np.abs(np.nan_to_num(target[0], nan=0.0)) > 1e-12)):
        raise ValueError("a causal finalist cannot hold a first-session position")
    if len(data.market_open) != len(dates) or len(data.market_close) != len(dates):
        raise ValueError("market timestamps are not aligned to canonical sessions")
    if len(vector_result.net_returns) != len(dates):
        raise ValueError("vector result is not aligned to canonical sessions")


def _with_padding(
    frame: pd.DataFrame,
    frame_dates: pd.DatetimeIndex,
    sids: tuple[int, ...],
) -> pd.DataFrame:
    values = frame.astype(float).copy()
    values.index = frame_dates[1:]
    values.columns = list(sids)
    padding = values.iloc[[0]].copy()
    padding.index = frame_dates[:1]
    return pd.concat((padding, values), axis=0)


def _expected_fill_times(
    data: ZiplineCanonicalData, session: Session
) -> list[pd.Timestamp]:
    targets = np.asarray(data.target_positions, dtype=float)
    previous = np.vstack((np.zeros((1, targets.shape[1])), targets[:-1]))
    changed = ~np.isclose(targets, previous, rtol=0.0, atol=1e-12, equal_nan=True)
    output: list[pd.Timestamp] = []
    for row, _column in np.argwhere(changed):
        if row == 0:
            raise ValueError("a first-session target has no causal fill timestamp")
        timestamp = (
            data.market_open[row]
            if session is Session.INTRADAY
            else data.market_close[row - 1]
        )
        output.append(_utc(timestamp))
    return sorted(output)


def _transaction_times(performance: pd.DataFrame) -> list[pd.Timestamp]:
    output: list[pd.Timestamp] = []
    for transactions in performance["transactions"]:
        for transaction in transactions:
            output.append(_utc(transaction["dt"]))
    return sorted(output)


def _zipline_return_positions(
    performance: pd.DataFrame,
    dates: pd.DatetimeIndex,
    close: pd.DataFrame,
    sids: tuple[int, ...],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    by_session: dict[pd.Timestamp, tuple[list[dict[str, Any]], float]] = {}
    for raw_timestamp, performance_row in performance.iterrows():
        timestamp: Any = raw_timestamp
        session = pd.Timestamp(timestamp).tz_localize(None).normalize()
        by_session[session] = (
            performance_row["positions"],
            float(performance_row["portfolio_value"]),
        )
    post_close = np.zeros((len(dates), len(sids)), dtype=float)
    sid_to_column = {sid: column for column, sid in enumerate(sids)}
    for row_number, session in enumerate(dates):
        positions, portfolio_value = by_session[session]
        if portfolio_value == 0.0:
            raise ValueError("Zipline produced a zero portfolio value")
        for position in positions:
            sid = int(getattr(position["sid"], "sid", position["sid"]))
            column = sid_to_column[sid]
            post_close[row_number, column] = (
                float(position["amount"])
                * float(cast(Any, close.iloc[row_number, column]))
                / portfolio_value
            )
    aligned = np.zeros_like(post_close)
    aligned[1:] = post_close[:-1]
    return aligned


def _portfolio_returns(
    positions: np.ndarray[Any, np.dtype[np.float64]],
    returns: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    finite = np.isfinite(returns)
    gross = np.asarray(
        np.sum(np.where(finite, positions * returns, 0.0), axis=1), dtype=float
    )
    gross[~finite.any(axis=1)] = np.nan
    return gross


def _frozen_costs(
    data: ZiplineCanonicalData, cost_model: CostModel
) -> np.ndarray[Any, np.dtype[np.float64]]:
    positions = np.asarray(data.cost_positions, dtype=float)
    if positions.ndim == 1:
        positions = positions[:, None]
    return cost_model.portfolio_costs(
        positions,
        asset_type=data.asset_type,
        adv_dollars=data.adv_dollars,
    )


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
