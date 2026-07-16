"""Causal, cross-sectional reversal signals for the opt-in research protocol.

The functions in this module never choose a winning variant.  They produce the
complete preregistered signal family so breadth and neutralization choices can
be evaluated as separate, multiplicity-adjusted trials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ReversalVariant = Literal["raw", "sector_neutral", "market_sector_residual"]


@dataclass(frozen=True, slots=True)
class ReversalSignalSet:
    """Aligned reversal scores and the exposures used to construct them."""

    raw: pd.DataFrame
    sector_neutral: pd.DataFrame
    market_sector_residual: pd.DataFrame
    market_beta: pd.DataFrame
    sector_beta: pd.DataFrame
    residual_volatility: pd.DataFrame
    sector_returns: pd.DataFrame
    market_returns: pd.Series
    point_in_time_universe: bool

    def signal(self, variant: ReversalVariant) -> pd.DataFrame:
        """Return one declared variant without dynamically searching names."""

        return {
            "raw": self.raw,
            "sector_neutral": self.sector_neutral,
            "market_sector_residual": self.market_sector_residual,
        }[variant]


def _membership_mask(
    values: pd.DataFrame, membership: pd.DataFrame | None
) -> tuple[pd.DataFrame, bool]:
    if membership is None:
        return values.notna(), False
    aligned = membership.reindex(index=values.index, columns=values.columns)
    if aligned.isna().any(axis=None):
        raise ValueError("point-in-time membership must cover every date and symbol")
    return aligned.astype(bool) & values.notna(), True


def leave_one_out_sector_returns(
    asset_returns: pd.DataFrame,
    sector_by_symbol: dict[str, str],
    *,
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return each asset's contemporaneous equal-weight sector excluding itself.

    Excluding the target asset prevents a large stock-specific shock from
    mechanically contaminating the benchmark used to call that same shock
    idiosyncratic.  A sector needs at least two eligible assets on a date.
    """

    values = asset_returns.astype(float)
    eligible, _ = _membership_mask(values, membership)
    output = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    groups: dict[str, list[object]] = {}
    for member in values.columns:
        sector = sector_by_symbol.get(str(member))
        if sector and sector != "ETF":
            groups.setdefault(sector, []).append(member)
    for columns in groups.values():
        group_values = values.loc[:, columns].where(eligible.loc[:, columns])
        finite = group_values.notna()
        group_sum = group_values.fillna(0.0).sum(axis=1)
        group_count = finite.sum(axis=1)
        for peer in columns:
            numerator = group_sum - group_values[peer].fillna(0.0)
            denominator = group_count - finite[peer].astype(int)
            benchmark = numerator.div(denominator.where(denominator > 0))
            output.loc[:, peer] = benchmark.where(eligible[peer])
    return output


def _compound(
    values: pd.DataFrame | pd.Series, window: int
) -> pd.DataFrame | pd.Series:
    return (1.0 + values).rolling(window, min_periods=window).apply(
        np.prod, raw=True
    ) - 1.0


def _rolling_two_factor_betas(
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    sector_returns: pd.DataFrame,
    *,
    window: int,
    min_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate causal rolling market/sector betas from data ending at ``t-1``."""

    market = market_returns.reindex(asset_returns.index).astype(float)
    market_beta = pd.DataFrame(
        np.nan, index=asset_returns.index, columns=asset_returns.columns
    )
    sector_beta = market_beta.copy()

    def lagged_mean(values: pd.Series) -> pd.Series:
        return values.rolling(window, min_periods=min_observations).mean().shift(1)

    def lagged_cov(left: pd.Series, right: pd.Series) -> pd.Series:
        valid = left.notna() & right.notna()
        x = left.where(valid)
        y = right.where(valid)
        return lagged_mean(x * y) - lagged_mean(x) * lagged_mean(y)

    for column in asset_returns.columns:
        asset = asset_returns[column].astype(float)
        sector = sector_returns[column].astype(float)
        var_market = lagged_cov(market, market)
        var_sector = lagged_cov(sector, sector)
        cov_market_sector = lagged_cov(market, sector)
        cov_asset_market = lagged_cov(asset, market)
        cov_asset_sector = lagged_cov(asset, sector)
        determinant = var_market * var_sector - cov_market_sector.pow(2)
        stable = determinant.abs() > np.finfo(float).eps
        market_coefficient = (
            cov_asset_market * var_sector - cov_asset_sector * cov_market_sector
        ).div(determinant.where(stable))
        sector_coefficient = (
            cov_asset_sector * var_market - cov_asset_market * cov_market_sector
        ).div(determinant.where(stable))
        market_beta.loc[:, column] = market_coefficient
        sector_beta.loc[:, column] = sector_coefficient
    return market_beta, sector_beta


def reversal_signal_set(
    adjusted_close: pd.DataFrame,
    sector_by_symbol: dict[str, str],
    *,
    market_returns: pd.Series | None = None,
    membership: pd.DataFrame | None = None,
    lookback: int = 5,
    beta_window: int = 252,
    beta_min_observations: int = 126,
    residual_vol_window: int = 20,
    epsilon: float = 1e-12,
) -> ReversalSignalSet:
    """Build raw, sector-neutral, and volatility-normalized residual reversal.

    Betas and idiosyncratic volatility visible on date ``t`` are estimated only
    from daily returns through ``t-1``.  The current five-session move can then
    be scored after the close without allowing that shock to refit its own risk
    model.
    """

    if lookback < 1:
        raise ValueError("lookback must be positive")
    if not 2 <= beta_min_observations <= beta_window:
        raise ValueError("beta_min_observations must be in [2, beta_window]")
    if residual_vol_window < 2 or epsilon <= 0.0:
        raise ValueError("residual volatility window and epsilon must be positive")
    close = adjusted_close.astype(float).sort_index()
    if close.empty or close.columns.has_duplicates or close.index.has_duplicates:
        raise ValueError("adjusted_close must be a non-empty unique date/symbol panel")
    eligible, point_in_time = _membership_mask(close, membership)
    daily = close.pct_change(fill_method=None).where(eligible)
    sectors = leave_one_out_sector_returns(
        daily, sector_by_symbol, membership=membership
    )
    if market_returns is None:
        market = daily.where(eligible).mean(axis=1, skipna=True)
    else:
        market = market_returns.reindex(close.index).astype(float)
    market.name = "market_return"
    market_beta, sector_beta = _rolling_two_factor_betas(
        daily,
        market,
        sectors,
        window=beta_window,
        min_observations=beta_min_observations,
    )
    five_day = close.div(close.shift(lookback)) - 1.0
    sector_five = _compound(sectors, lookback)
    if not isinstance(sector_five, pd.DataFrame):  # pragma: no cover - type narrowing
        raise TypeError("sector compounding did not return a DataFrame")
    market_five = _compound(market, lookback)
    if not isinstance(market_five, pd.Series):  # pragma: no cover - type narrowing
        raise TypeError("market compounding did not return a Series")
    residual_daily = daily - market_beta.mul(market, axis=0) - sector_beta.mul(sectors)
    residual_volatility = (
        residual_daily.rolling(residual_vol_window, min_periods=residual_vol_window)
        .std(ddof=1)
        .shift(1)
    )
    residual_five = (
        five_day - market_beta.mul(market_five, axis=0) - sector_beta.mul(sector_five)
    )
    raw = (-five_day).where(eligible)
    sector_neutral = (-(five_day - sector_five)).where(eligible)
    scale = residual_volatility * float(np.sqrt(lookback))
    residual = (-residual_five.div(scale.where(scale.abs() > epsilon))).where(eligible)
    metadata = {
        "lookback": lookback,
        "available_at": "close",
        "beta_information_end": "t-1",
        "point_in_time_universe": point_in_time,
    }
    raw.attrs.update(metadata | {"variant": "raw"})
    sector_neutral.attrs.update(metadata | {"variant": "sector_neutral"})
    residual.attrs.update(metadata | {"variant": "market_sector_residual"})
    return ReversalSignalSet(
        raw,
        sector_neutral,
        residual,
        market_beta,
        sector_beta,
        residual_volatility,
        sectors,
        market,
        point_in_time,
    )
