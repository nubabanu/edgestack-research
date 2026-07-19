"""Effective-spread estimation from daily OHLC.

Corwin-Schultz (2012) infers the full quoted spread from the fact that a
day's high is (almost always) a buyer-initiated trade and its low a
seller-initiated one, so high/low ratios embed the spread while two-day
ranges embed twice the variance. Abdi-Ranaldo (2017) is the fallback where
Corwin-Schultz is undefined. Both are noisy on daily data, so estimates are
aggregated to per-name monthly medians and consumers must FLOOR them at the
assumed baseline: measurement may only make research costs harder to beat,
never easier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_MONTH = "M"


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Daily Corwin-Schultz full-spread fraction from adjacent-day ranges.

    Negative alpha estimates (common in quiet or trending tapes) yield
    non-positive spreads and are reported as NaN rather than clipped to a
    fake zero-cost day.
    """

    high_values = pd.to_numeric(high, errors="coerce")
    low_values = pd.to_numeric(low, errors="coerce")
    valid = (high_values > 0) & (low_values > 0) & (high_values >= low_values)
    log_ratio = pd.Series(np.nan, index=high_values.index, dtype=float)
    log_ratio[valid] = np.log(
        high_values[valid].to_numpy() / low_values[valid].to_numpy()
    )
    beta = log_ratio.pow(2) + log_ratio.shift(1).pow(2)
    two_day_high = pd.concat([high_values, high_values.shift(1)], axis=1).max(axis=1)
    two_day_low = pd.concat([low_values, low_values.shift(1)], axis=1).min(axis=1)
    two_day_valid = (two_day_high > 0) & (two_day_low > 0)
    gamma = pd.Series(np.nan, index=high_values.index, dtype=float)
    gamma[two_day_valid] = (
        np.log(
            two_day_high[two_day_valid].to_numpy()
            / two_day_low[two_day_valid].to_numpy()
        )
        ** 2
    )
    denominator = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denominator - np.sqrt(
        gamma / denominator
    )
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.where(spread > 0.0).rename("cs_spread")


def abdi_ranaldo_spread(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Daily Abdi-Ranaldo full-spread fraction from close-to-midrange gaps."""

    high_values = pd.to_numeric(high, errors="coerce")
    low_values = pd.to_numeric(low, errors="coerce")
    close_values = pd.to_numeric(close, errors="coerce")
    valid = (
        (high_values > 0)
        & (low_values > 0)
        & (close_values > 0)
        & (high_values >= low_values)
    )
    log_close = pd.Series(np.nan, index=close_values.index, dtype=float)
    log_close[valid] = np.log(close_values[valid].to_numpy())
    midrange = pd.Series(np.nan, index=close_values.index, dtype=float)
    midrange[valid] = (
        np.log(high_values[valid].to_numpy()) + np.log(low_values[valid].to_numpy())
    ) / 2.0
    squared = 4.0 * (log_close - midrange) * (log_close - midrange.shift(-1))
    spread = np.sqrt(squared.where(squared > 0.0))
    return spread.rename("ar_spread")


def monthly_median_spread_bps(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    *,
    minimum_observations: int = 12,
) -> pd.DataFrame:
    """Per-name monthly median full-spread estimates in basis points.

    Corwin-Schultz is used where defined; Abdi-Ranaldo fills the remaining
    days. Months with fewer than ``minimum_observations`` defined daily
    estimates stay NaN so a thin month cannot masquerade as a measurement.
    """

    if not (high.columns.equals(low.columns) and high.columns.equals(close.columns)):
        raise ValueError("high, low, and close must share identical columns")
    if not (high.index.equals(low.index) and high.index.equals(close.index)):
        raise ValueError("high, low, and close must share an identical index")
    monthly: dict[str, pd.Series] = {}
    for column in high.columns:
        primary = corwin_schultz_spread(high[column], low[column])
        fallback = abdi_ranaldo_spread(high[column], low[column], close[column])
        daily = primary.fillna(fallback)
        grouped = daily.groupby(pd.PeriodIndex(daily.index, freq=_TRADING_MONTH))
        median = grouped.median()
        counts = grouped.count()
        median[counts < minimum_observations] = np.nan
        monthly[str(column)] = median * 10_000.0
    return pd.DataFrame(monthly)


def floored_spread_matrix(
    monthly_bps: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    baseline_bps: pd.Series | float,
) -> pd.DataFrame:
    """Expand monthly medians to sessions, floored at the assumed baseline.

    The PRIOR month's estimate prices each session (no same-month lookahead),
    and every missing or below-baseline value falls back to the baseline, so
    the result can only tighten the cost model.
    """

    lagged = monthly_bps.shift(1)
    periods = pd.PeriodIndex(sessions, freq=_TRADING_MONTH)
    expanded = lagged.reindex(periods)
    expanded.index = sessions
    if isinstance(baseline_bps, pd.Series):
        baseline = baseline_bps.reindex(monthly_bps.columns).astype(float)
        if baseline.isna().any():
            raise ValueError("baseline_bps must cover every column")
        floor = pd.DataFrame(
            np.broadcast_to(baseline.to_numpy(), expanded.shape),
            index=expanded.index,
            columns=expanded.columns,
        )
    else:
        floor = pd.DataFrame(
            float(baseline_bps), index=expanded.index, columns=expanded.columns
        )
    return expanded.where(expanded > floor, floor).fillna(floor)


__all__ = [
    "abdi_ranaldo_spread",
    "corwin_schultz_spread",
    "floored_spread_matrix",
    "monthly_median_spread_bps",
]
