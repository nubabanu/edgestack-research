"""Cross-sectional anomaly features computed without future information."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _prices_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame indexed by session")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("price index must be sorted and unique")
    finite = prices.to_numpy(dtype=float)
    if np.any(finite[np.isfinite(finite)] <= 0.0):
        raise ValueError("prices must be positive where finite")
    return prices.astype(float)


def momentum_12_1(
    prices: pd.DataFrame,
    *,
    lookback: int = 252,
    skip: int = 21,
    log: bool = False,
) -> pd.DataFrame:
    """Compute the 12-minus-1-month momentum signal.

    The value at session ``t`` is ``P[t-skip] / P[t-lookback] - 1``. Thus the
    latest month is excluded and the signal uses only observations known by
    ``t`` (Jegadeesh and Titman, 1993).
    """

    frame = _prices_frame(prices)
    if lookback <= skip or skip < 0:
        raise ValueError("lookback must be greater than non-negative skip")
    ratio = frame.shift(skip) / frame.shift(lookback)
    signal = (
        pd.DataFrame(np.log(ratio.to_numpy()), index=ratio.index, columns=ratio.columns)
        if log
        else ratio - 1.0
    )
    signal.attrs.update({"lookback": lookback, "skip": skip, "available_at": "close"})
    return signal


def short_term_reversal(
    prices: pd.DataFrame,
    *,
    lookback: int = 5,
    contrarian: bool = True,
) -> pd.DataFrame:
    """Compute prior-week return, optionally signed as a contrarian score."""

    frame = _prices_frame(prices)
    if lookback < 1:
        raise ValueError("lookback must be positive")
    past_return = frame / frame.shift(lookback) - 1.0
    result = -past_return if contrarian else past_return
    result.attrs.update(
        {"lookback": lookback, "contrarian": contrarian, "available_at": "close"}
    )
    return result


def realized_volatility(
    prices: pd.DataFrame,
    *,
    window: int = 252,
    annualization: float = 252.0,
    low_vol_score: bool = False,
) -> pd.DataFrame:
    """Compute trailing close-to-close realized volatility.

    ``low_vol_score=True`` negates volatility so that larger scores always mean
    stronger exposure to the named low-volatility characteristic.
    """

    frame = _prices_frame(prices)
    if window < 2 or annualization <= 0:
        raise ValueError("window must be >=2 and annualization positive")
    ratio = frame / frame.shift(1)
    returns = pd.DataFrame(
        np.log(ratio.to_numpy()), index=ratio.index, columns=ratio.columns
    )
    vol = (
        returns.rolling(window=window, min_periods=window)
        .std(ddof=1)
        .mul(float(np.sqrt(annualization)))
    )
    result = -vol if low_vol_score else vol
    result.attrs.update(
        {
            "window": window,
            "annualization": annualization,
            "available_at": "close",
        }
    )
    return result


def proximity_to_high(prices: pd.DataFrame, *, window: int = 252) -> pd.DataFrame:
    """Return price divided by its trailing high (George and Hwang, 2004)."""

    frame = _prices_frame(prices)
    if window < 2:
        raise ValueError("window must be >=2")
    trailing_high = frame.rolling(window=window, min_periods=window).max()
    result = frame / trailing_high
    result.attrs.update({"window": window, "available_at": "close"})
    return result


def standardized_unexpected_earnings(
    actual: pd.DataFrame,
    consensus: pd.DataFrame,
    *,
    scale: pd.DataFrame | None = None,
    min_scale: float = 1e-12,
) -> pd.DataFrame:
    """Compute SUE only when timestamp-aligned earnings inputs are supplied.

    Callers are responsible for joining each observation by its announcement
    ``available_at`` timestamp. Missing consensus data stays missing; it is never
    inferred from future estimates.
    """

    if not actual.index.equals(consensus.index) or not actual.columns.equals(
        consensus.columns
    ):
        raise ValueError("actual and consensus must be exactly aligned")
    surprise = actual.astype(float) - consensus.astype(float)
    denominator = (
        scale.astype(float) if scale is not None else consensus.astype(float).abs()
    )
    denominator = denominator.where(denominator.abs() >= min_scale)
    return surprise / denominator


def cross_sectional_percentile(
    signal: pd.DataFrame,
    *,
    ascending: bool = True,
    minimum_assets: int = 2,
) -> pd.DataFrame:
    """Rank a feature to deterministic [0, 1] cross-sectional percentiles."""

    if minimum_assets < 1:
        raise ValueError("minimum_assets must be positive")
    count = signal.notna().sum(axis=1)
    ranked = signal.rank(axis=1, pct=True, method="average", ascending=ascending)
    return ranked.where(count >= minimum_assets, axis=0)


def decile_weights(
    signal: pd.DataFrame,
    *,
    quantile: float = 0.1,
    long_short: bool = True,
) -> pd.DataFrame:
    """Construct equal-weight, dollar-neutral extreme-quantile portfolios."""

    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must be strictly between 0 and 0.5")
    ranks = cross_sectional_percentile(signal)
    long_mask = ranks > (1.0 - quantile)
    short_mask = ranks <= quantile
    long_count = long_mask.sum(axis=1).replace(0, np.nan)
    short_count = short_mask.sum(axis=1).replace(0, np.nan)
    long_weights = long_mask.div(long_count, axis=0).fillna(0.0)
    if not long_short:
        return long_weights
    short_weights = short_mask.div(short_count, axis=0).fillna(0.0)
    return long_weights - short_weights


@dataclass(frozen=True, slots=True)
class CrossSectionalFeatureSet:
    """Registered canonical cross-sectional features."""

    momentum: pd.DataFrame
    reversal: pd.DataFrame
    low_volatility: pd.DataFrame
    high_proximity: pd.DataFrame


def canonical_features(prices: pd.DataFrame) -> CrossSectionalFeatureSet:
    """Compute the four always-available canonical signal families."""

    return CrossSectionalFeatureSet(
        momentum=momentum_12_1(prices),
        reversal=short_term_reversal(prices),
        low_volatility=realized_volatility(prices, low_vol_score=True),
        high_proximity=proximity_to_high(prices),
    )


low_volatility = realized_volatility
pct_from_52w_high = proximity_to_high
