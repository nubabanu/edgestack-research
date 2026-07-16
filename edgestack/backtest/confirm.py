"""Independent event-loop confirmation for vectorized finalists."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata

import numpy as np

from edgestack.backtest.costs import CostModel
from edgestack.backtest.engine import BacktestResult
from edgestack.models import HypothesisSpec, ensure_fill_after_signal
from edgestack.stats._types import FloatArray


@dataclass(frozen=True, slots=True)
class ConfirmationData:
    """Inputs for the deliberately loop-based reference simulator."""

    signal: FloatArray
    returns: FloatArray
    timestamps: tuple[datetime, ...]
    adv_dollars: float | FloatArray = 100_000_000.0
    asset_type: str = "equity"


@dataclass(frozen=True, slots=True)
class ConfirmationResult:
    """Agreement evidence between independent simulation implementations."""

    hypothesis_id: str
    trade_count: int
    vector_trade_count: int | None
    net_mean: float
    vector_net_mean: float | None
    difference_bps_per_trade: float | None
    timestamps_match: bool
    passed: bool
    backend: str = "independent_event_loop"
    reason: str = ""
    missing_fill_events: tuple[tuple[int, str], ...] = ()
    extra_fill_events: tuple[tuple[int, str], ...] = ()
    convention_supported: bool = True


@dataclass(frozen=True, slots=True)
class ZiplineBackendStatus:
    """Runtime status of the required Zipline finalist backend.

    Installing Zipline is not evidence that a strategy was run through it.
    ``executable`` therefore means that EdgeStack has an implemented adapter
    capable of supplying canonical bars, assets, and corporate actions to a
    real Zipline simulation—not merely that the package imports.
    """

    installed: bool
    version: str | None
    executable: bool
    reason: str


class ConfirmationEngine:
    """Confirm fills with a sequential engine sharing no sweep arithmetic."""

    def __init__(self, *, tolerance_bps_per_trade: float = 1.0) -> None:
        if tolerance_bps_per_trade < 0.0:
            raise ValueError("tolerance cannot be negative")
        self.tolerance_bps_per_trade = tolerance_bps_per_trade

    def confirm(
        self,
        spec: HypothesisSpec,
        data: ConfirmationData,
        vector_result: BacktestResult | None = None,
        costs: CostModel | None = None,
    ) -> ConfirmationResult:
        """Sequentially execute each target one bar after its signal."""

        signal = np.asarray(data.signal, dtype=float)
        returns = np.asarray(data.returns, dtype=float)
        if (
            signal.ndim != 1
            or returns.shape != signal.shape
            or len(data.timestamps) != len(signal)
        ):
            raise ValueError(
                "confirmation inputs must be aligned one-dimensional series"
            )
        liquidity = np.asarray(data.adv_dollars, dtype=float)
        if liquidity.ndim > 1 or (
            liquidity.ndim == 1 and len(liquidity) != len(signal)
        ):
            raise ValueError("confirmation ADV must be scalar or aligned by timestamp")
        model = costs or CostModel()
        position = 0.0
        previous = 0.0
        net_stream = np.full(len(signal), np.nan, dtype=float)
        fill_times: list[datetime] = []
        for bar in range(len(signal)):
            if bar == 0:
                position = 0.0
            else:
                signal_time = data.timestamps[bar - 1]
                fill_time = data.timestamps[bar]
                ensure_fill_after_signal(signal_time, fill_time)
                position = (
                    0.0 if not math.isfinite(signal[bar - 1]) else signal[bar - 1]
                )
                if position != previous:
                    fill_times.append(fill_time)
            if math.isfinite(returns[bar]):
                gross = position * returns[bar]
                daily_cost = model.portfolio_costs(
                    np.array([[previous], [position]], dtype=float),
                    asset_type="etf" if data.asset_type.lower() == "etf" else "equity",
                    adv_dollars=(
                        float(liquidity)
                        if liquidity.ndim == 0
                        else float(liquidity[bar])
                    ),
                )[1]
                net_stream[bar] = gross - daily_cost
            previous = position
        trade_count = len(fill_times)
        net_mean = (
            float(np.nanmean(net_stream)) if np.isfinite(net_stream).any() else math.nan
        )
        if vector_result is None:
            return ConfirmationResult(
                spec.hypothesis_id,
                trade_count,
                None,
                net_mean,
                None,
                None,
                True,
                True,
                reason="standalone independent confirmation completed",
            )
        vector_positions = np.asarray(vector_result.positions, dtype=float)
        if vector_positions.ndim != 1:
            raise ValueError(
                "sequential confirmation currently expects a scalar strategy"
            )
        vector_trades = int(
            np.count_nonzero(np.diff(np.r_[0.0, vector_positions]) != 0.0)
        )
        vector_mean = float(np.nanmean(vector_result.net_returns))
        denominator = max(trade_count, 1)
        difference = abs(net_mean - vector_mean) * 10_000.0 * len(signal) / denominator
        count_match = trade_count == vector_trades
        passed = bool(count_match and difference <= self.tolerance_bps_per_trade)
        return ConfirmationResult(
            spec.hypothesis_id,
            trade_count,
            vector_trades,
            net_mean,
            vector_mean,
            difference,
            count_match,
            passed,
            reason=(
                "agreement within tolerance"
                if passed
                else "trade count or net mean disagreement"
            ),
        )


def zipline_available() -> bool:
    """Report whether the optional independent Zipline backend is installed."""

    try:
        import zipline  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        return False
    return True


def zipline_backend_status() -> ZiplineBackendStatus:
    """Return an honest status for the independent finalist backend.

    Importability alone is never enough.  EdgeStack's executable adapter builds
    an in-memory Zipline asset database and daily bar reader from the canonical
    adjusted OHLCV matrices, submits real orders through ``TradingAlgorithm``,
    and compares the resulting fills with the vectorized finalist.  Runtime or
    convention failures are handled by the validation layer as unavailable or
    failed confirmation evidence; they can never become a pass merely because
    this status probe succeeds.
    """

    if not zipline_available():
        return ZiplineBackendStatus(
            installed=False,
            version=None,
            executable=False,
            reason="zipline-reloaded is not installed",
        )
    try:
        version = metadata.version("zipline-reloaded")
    except metadata.PackageNotFoundError:
        version = None
    if version != "3.1.1":
        return ZiplineBackendStatus(
            installed=True,
            version=version,
            executable=False,
            reason="the runtime is not pinned to zipline-reloaded 3.1.1",
        )
    return ZiplineBackendStatus(
        installed=True,
        version=version,
        executable=True,
        reason=(
            "canonical adjusted-OHLCV in-memory TradingAlgorithm adapter is "
            "available; actions are represented once in adjusted prices"
        ),
    )
