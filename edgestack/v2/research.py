"""Dedicated monthly/yearly loss-aware cohort research."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from edgestack.v2.metrics import LossMetrics, loss_metrics
from edgestack.v2.veto import VetoSpec, declared_vetoes


class Horizon(StrEnum):
    """V2 models are independent; the five-day edge is never extrapolated."""

    MONTHLY = "MONTHLY_21"
    YEARLY = "YEARLY_252"

    @property
    def sessions(self) -> int:
        return 21 if self is Horizon.MONTHLY else 252


class SignalFamily(StrEnum):
    """Compact preregistered family set."""

    MOMENTUM_252_SKIP_21 = "MOMENTUM_252_SKIP_21"
    REVERSAL_5 = "REVERSAL_5"
    LOW_VOLATILITY_252 = "LOW_VOLATILITY_252"
    HIGH_PRICE_PROXIMITY_252 = "HIGH_PRICE_PROXIMITY_252"
    EQUAL_RANK_COMPOSITE = "EQUAL_RANK_COMPOSITE"


class PortfolioForm(StrEnum):
    """Top/bottom-five portfolio construction."""

    LONG_ONLY = "LONG_ONLY"
    SHORT_ONLY = "SHORT_ONLY"
    MARKET_NEUTRAL = "MARKET_NEUTRAL"


@dataclass(frozen=True, slots=True)
class TrialSpec:
    """A declared test counted before data inspection."""

    horizon: Horizon
    family: SignalFamily
    direction: str
    portfolio_form: PortfolioForm
    veto: VetoSpec
    leverage: float

    @property
    def trial_id(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, default=str, separators=(",", ":")
        )
        return "v2-" + hashlib.sha256(payload.encode()).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Historical diagnostic result; never a replacement holdout."""

    spec: TrialSpec
    cohort_returns: tuple[float, ...]
    daily_returns: tuple[float, ...]
    loss: LossMetrics
    net_mean: float
    bankrupt: bool
    historical_diagnostic: bool = True
    forward_promotion_required: bool = True


def declared_trials() -> tuple[TrialSpec, ...]:
    """Enumerate every horizon/family/form/veto/leverage trial exactly once."""

    direction = {
        PortfolioForm.LONG_ONLY: "LONG",
        PortfolioForm.SHORT_ONLY: "SHORT",
        PortfolioForm.MARKET_NEUTRAL: "LONG_SHORT",
    }
    return tuple(
        TrialSpec(horizon, family, direction[form], form, veto, leverage)
        for horizon in Horizon
        for family in SignalFamily
        for form in PortfolioForm
        for veto in declared_vetoes()
        for leverage in (1.0, 1.5, 2.0)
    )


def signal_scores(close: pd.DataFrame) -> dict[SignalFamily, pd.DataFrame]:
    """Compute the five causal cross-sectional score panels."""

    prices = close.astype(float).sort_index()
    returns = prices.pct_change(fill_method=None)
    momentum = prices.shift(21).div(prices.shift(252)).sub(1)
    reversal = prices.div(prices.shift(5)).sub(1).mul(-1)
    low_vol = returns.rolling(252, min_periods=252).std().mul(-1)
    proximity = prices.div(prices.rolling(252, min_periods=252).max())
    ranks = [
        panel.rank(axis=1, pct=True)
        for panel in (momentum, reversal, low_vol, proximity)
    ]
    composite = sum(ranks[1:], ranks[0].copy()) / 4
    return {
        SignalFamily.MOMENTUM_252_SKIP_21: momentum,
        SignalFamily.REVERSAL_5: reversal,
        SignalFamily.LOW_VOLATILITY_252: low_vol,
        SignalFamily.HIGH_PRICE_PROXIMITY_252: proximity,
        SignalFamily.EQUAL_RANK_COMPOSITE: composite,
    }


def run_trial(
    spec: TrialSpec,
    close: pd.DataFrame,
    *,
    annual_sofr: pd.Series | float,
    one_way_cost_bps: float = 5.0,
    veto_mask: pd.DataFrame | None = None,
) -> TrialResult:
    """Run next-session, fixed-horizon top/bottom-five cohorts after costs."""

    if spec.veto.kind.value != "NONE" and veto_mask is None:
        raise ValueError("a preregistered event/gap veto requires a causal veto_mask")
    scores = signal_scores(close)[spec.family]
    daily_asset_returns = close.astype(float).pct_change(fill_method=None)
    date_index = pd.DatetimeIndex(scores.index)
    month_ends = scores.groupby(date_index.to_period("M")).tail(1).index
    cohort_paths: list[NDArray[np.float64]] = []
    for signal_date in month_ends:
        signal_position = close.index.get_loc(signal_date)
        if not isinstance(signal_position, int):
            raise ValueError("close index must be unique")
        start = signal_position + 1
        end = start + spec.horizon.sessions
        if start >= len(close) or end > len(close):
            continue
        row = scores.loc[signal_date].copy()
        if veto_mask is not None:
            if signal_date not in veto_mask.index:
                raise ValueError("veto_mask is missing a signal date")
            flags = (
                veto_mask.loc[signal_date].reindex(row.index).fillna(True).astype(bool)
            )
            row = row.mask(flags)
        row = row.dropna().sort_values()
        if len(row) < 10:
            continue
        bottom = list(row.index[:5])
        top = list(row.index[-5:])
        window = daily_asset_returns.iloc[start:end]
        if spec.portfolio_form is PortfolioForm.LONG_ONLY:
            path = window[top].mean(axis=1).to_numpy(float)
        elif spec.portfolio_form is PortfolioForm.SHORT_ONLY:
            path = -window[bottom].mean(axis=1).to_numpy(float)
        else:
            path = 0.5 * (
                window[top].mean(axis=1).to_numpy(float)
                - window[bottom].mean(axis=1).to_numpy(float)
            )
        financing = _daily_financing(annual_sofr, window.index)
        leveraged = spec.leverage * np.nan_to_num(path, nan=0.0)
        leveraged -= np.maximum(spec.leverage - 1.0, 0.0) * financing
        leveraged[0] -= one_way_cost_bps * spec.leverage / 10_000
        leveraged[-1] -= one_way_cost_bps * spec.leverage / 10_000
        cohort_paths.append(leveraged)
    if not cohort_paths:
        raise ValueError("no eligible completed cohorts")
    paths = np.vstack(cohort_paths)
    cohort_returns = np.prod(1 + paths, axis=1) - 1
    bankrupt = bool(np.any(np.cumprod(1 + paths, axis=1) <= 0))
    daily = np.nanmean(paths, axis=0)
    losses = loss_metrics(cohort_returns, path_returns=paths)
    return TrialResult(
        spec=spec,
        cohort_returns=tuple(float(item) for item in cohort_returns),
        daily_returns=tuple(float(item) for item in daily),
        loss=losses,
        net_mean=float(np.mean(cohort_returns)),
        bankrupt=bankrupt,
    )


def _daily_financing(rate: pd.Series | float, index: pd.Index) -> NDArray[np.float64]:
    if isinstance(rate, pd.Series):
        aligned = rate.reindex(index).ffill()
        if bool(aligned.isna().any()):
            raise ValueError("SOFR is unavailable for at least one holding session")
        annual = aligned.to_numpy(float) + 0.03
    else:
        if rate < 0:
            raise ValueError("annual SOFR cannot be negative")
        annual = np.full(len(index), float(rate) + 0.03)
    return annual / 360.0
