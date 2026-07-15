"""Causal trend and VIX regime labels."""

from __future__ import annotations

from enum import StrEnum

import pandas as pd

from edgestack.entrytiming.indicators import sma


class TrendRegime(StrEnum):
    """Price relative to its trailing moving average."""

    UP = "UP"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


class VixRegime(StrEnum):
    """Predeclared VIX state."""

    LOW = "LOW"
    MID = "MID"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


def ma_regime(prices: pd.Series, window: int = 200) -> pd.Series:
    """Return UP/DOWN from price versus its trailing SMA."""

    average = sma(prices, window)
    result = pd.Series(TrendRegime.UNKNOWN, index=prices.index, dtype="object")
    known = average.notna() & prices.notna()
    result.loc[known & (prices >= average)] = TrendRegime.UP
    result.loc[known & (prices < average)] = TrendRegime.DOWN
    return result.rename("ma_regime")


def vix_regime(vix: pd.Series, low: float = 15.0, high: float = 25.0) -> pd.Series:
    """Classify VIX without forward filling values unavailable on the date."""

    if not low < high:
        raise ValueError("low must be below high")
    result = pd.Series(VixRegime.UNKNOWN, index=vix.index, dtype="object")
    result.loc[vix.notna() & (vix < low)] = VixRegime.LOW
    result.loc[vix.notna() & (vix >= low) & (vix < high)] = VixRegime.MID
    result.loc[vix.notna() & (vix >= high)] = VixRegime.HIGH
    return result.rename("vix_regime")
