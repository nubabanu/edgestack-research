"""Pure vectorized technical measurements used only as governed overlays.

These functions do not claim standalone alpha.  They expose execution and risk
measurements that must pass incremental interaction tests before activation.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def sma(values: pd.Series, window: int) -> pd.Series:
    """Simple moving average over trailing observations."""

    _positive_window(window)
    return values.astype(float).rolling(window, min_periods=window).mean().rename("sma")


def ema(values: pd.Series, span: int) -> pd.Series:
    """Causal exponential moving average."""

    _positive_window(span)
    return (
        values.astype(float)
        .ewm(span=span, adjust=False, min_periods=span)
        .mean()
        .rename("ema")
    )


def rsi(values: pd.Series, window: int = 14) -> pd.Series:
    """Wilder relative-strength index, including RSI(2)."""

    _positive_window(window)
    delta = values.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    relative = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + relative)
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return result.rename(f"rsi{window}")


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Maximum of intrabar range and gaps from the previous close."""

    previous = close.astype(float).shift(1)
    values = pd.concat(
        [
            high.astype(float) - low.astype(float),
            (high.astype(float) - previous).abs(),
            (low.astype(float) - previous).abs(),
        ],
        axis=1,
    )
    return values.max(axis=1).rename("true_range")


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Wilder average true range."""

    _positive_window(window)
    return (
        true_range(high, low, close)
        .ewm(alpha=1.0 / window, adjust=False, min_periods=window)
        .mean()
        .rename("atr")
    )


def realized_vol(
    returns: pd.Series, window: int = 21, annualization: int = 252
) -> pd.Series:
    """Trailing annualized standard deviation of simple returns."""

    _positive_window(window)
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    result = (
        returns.astype(float).rolling(window, min_periods=window).std(ddof=1)
        * np.sqrt(annualization)
    ).rename("realized_vol")
    return cast(pd.Series, result)


def vwap(
    price: pd.Series, volume: pd.Series, session: pd.Series | None = None
) -> pd.Series:
    """Cumulative true VWAP for intraday bars.

    ``price`` must be a trade price or intraday bar representative price.  Daily
    typical price is intentionally not accepted or inferred.
    """

    if not price.index.equals(volume.index):
        raise ValueError("price and volume indexes must match")
    if bool((volume.astype(float) < 0).any()):
        raise ValueError("volume cannot be negative")
    traded = price.astype(float) * volume.astype(float)
    if session is None:
        denominator = volume.astype(float).cumsum()
        return (traded.cumsum() / denominator.replace(0.0, np.nan)).rename("vwap")
    if not session.index.equals(price.index):
        raise ValueError("session index must match price index")
    numerator = traded.groupby(session, sort=False).cumsum()
    denominator = volume.astype(float).groupby(session, sort=False).cumsum()
    return (numerator / denominator.replace(0.0, np.nan)).rename("vwap")


def pct_from_52w_high(close: pd.Series, window: int = 252) -> pd.Series:
    """Fractional distance from the trailing 52-week high."""

    _positive_window(window)
    trailing_high = close.astype(float).rolling(window, min_periods=window).max()
    return (close.astype(float) / trailing_high - 1.0).rename("pct_from_52w_high")


def bollinger_pct_b(
    close: pd.Series, window: int = 20, deviations: float = 2.0
) -> pd.Series:
    """Position within trailing Bollinger bands, expressed as percent-B."""

    _positive_window(window)
    if deviations <= 0:
        raise ValueError("deviations must be positive")
    values = close.astype(float)
    middle = values.rolling(window, min_periods=window).mean()
    sigma = values.rolling(window, min_periods=window).std(ddof=0)
    lower = middle - deviations * sigma
    upper = middle + deviations * sigma
    return ((values - lower) / (upper - lower).replace(0.0, np.nan)).rename(
        "bollinger_pct_b"
    )


def _positive_window(window: int) -> None:
    if window <= 0:
        raise ValueError("window must be positive")
