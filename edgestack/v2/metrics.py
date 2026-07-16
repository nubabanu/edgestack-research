"""Loss-first metrics with deterministic bootstrap uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LossMetrics:
    """Immutable downside evidence for completed cohorts."""

    loss_probability: float
    loss_probability_ci: tuple[float, float]
    expected_shortfall_95: float
    tenth_percentile_return: float
    trade_mae: float
    basket_mae: float
    maximum_drawdown_duration: int
    median_drawdown_duration: float
    drawdown_duration_p90: float
    maximum_losing_streak: int
    median_losing_streak: float
    losing_streak_p90: float


def loss_metrics(
    outcomes: FloatArray,
    *,
    path_returns: FloatArray | None = None,
    bootstrap_draws: int = 2_000,
    seed: int = 20250301,
) -> LossMetrics:
    """Compute downside metrics; expected shortfall is a positive loss amount."""

    values = _finite_1d(outcomes)
    if values.size == 0:
        raise ValueError("at least one finite completed outcome is required")
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    losing = values < 0
    probability = float(losing.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(bootstrap_draws, values.size))
    boot_probability = (values[indices] < 0).mean(axis=1)
    probability_ci = tuple(
        float(item) for item in np.quantile(boot_probability, [0.025, 0.975])
    )
    cutoff = float(np.quantile(values, 0.05))
    expected_shortfall = max(0.0, -float(values[values <= cutoff].mean()))
    path: FloatArray
    if path_returns is None:
        path = cast(FloatArray, values.reshape(-1, 1))
    else:
        path = cast(FloatArray, np.asarray(path_returns, dtype=np.float64))
        if path.ndim != 2 or path.shape[0] != values.size:
            raise ValueError("path_returns must have one row per completed outcome")
    trade_mae_values = _cohort_mae(path)
    basket_path = np.nanmean(path, axis=0)
    basket_mae = float(np.min(np.cumprod(1 + np.nan_to_num(basket_path, nan=0.0)) - 1))
    drawdown_durations = _drawdown_durations(np.nan_to_num(basket_path, nan=0.0))
    streaks = _losing_streaks(values)
    return LossMetrics(
        loss_probability=probability,
        loss_probability_ci=(probability_ci[0], probability_ci[1]),
        expected_shortfall_95=expected_shortfall,
        tenth_percentile_return=float(np.quantile(values, 0.10)),
        trade_mae=float(np.min(trade_mae_values)),
        basket_mae=basket_mae,
        maximum_drawdown_duration=max(drawdown_durations, default=0),
        median_drawdown_duration=_percentile(drawdown_durations, 50),
        drawdown_duration_p90=_percentile(drawdown_durations, 90),
        maximum_losing_streak=max(streaks, default=0),
        median_losing_streak=_percentile(streaks, 50),
        losing_streak_p90=_percentile(streaks, 90),
    )


def gap_adjusted_stop_return(
    *,
    entry_price: float,
    stop_price: float,
    first_tradable_price: float,
    direction: str,
    costs_bps: float,
) -> float:
    """Execute a breached stop at the worse first tradable price plus costs."""

    if min(entry_price, stop_price, first_tradable_price) <= 0 or costs_bps < 0:
        raise ValueError("prices must be positive and costs nonnegative")
    if direction == "LONG":
        fill = min(stop_price, first_tradable_price)
        gross = fill / entry_price - 1
    elif direction == "SHORT":
        fill = max(stop_price, first_tradable_price)
        gross = 1 - fill / entry_price
    else:
        raise ValueError("direction must be LONG or SHORT")
    return float(gross - costs_bps / 10_000)


def _cohort_mae(path: FloatArray) -> FloatArray:
    wealth = np.cumprod(1 + np.nan_to_num(path, nan=0.0), axis=1)
    return cast(FloatArray, np.min(wealth - 1, axis=1))


def _drawdown_durations(returns: FloatArray) -> list[int]:
    wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])
    underwater = np.r_[1.0, wealth] < peak
    return _run_lengths(underwater)


def _losing_streaks(values: FloatArray) -> list[int]:
    return _run_lengths(values < 0)


def _run_lengths(mask: NDArray[np.bool_]) -> list[int]:
    runs: list[int] = []
    current = 0
    for flagged in mask:
        if bool(flagged):
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _finite_1d(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("outcomes must be one-dimensional")
    return cast(FloatArray, array[np.isfinite(array)])


def _percentile(values: list[int], percentile: float) -> float:
    return 0.0 if not values else float(np.percentile(values, percentile))
