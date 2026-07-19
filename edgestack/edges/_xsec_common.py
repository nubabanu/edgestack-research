"""Shared machinery for cross-sectional monthly-rebalance study families.

Generalizes the portfolio contract validated by the low-vol and PEAD
studies: a causal per-name feature frame, ranked at each month-end close,
equal-weight entry into the selected fraction with buy-and-hold drift inside
the month, per-name flip costs at each rebalance, and a flat month whenever
the minimum name count is not met. Family modules stay thin: they declare
feature builders and trial parameters; this module produces the daily
gross/net streams the shared gauntlet consumes.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import numpy as np
import pandas as pd

from edgestack.edges._study_common import flip_cost_fraction

FeatureBuilder = Callable[[Mapping[str, pd.DataFrame], Sequence[str]], pd.DataFrame]


@dataclass(frozen=True)
class XsecTrial:
    """One declared cross-sectional trial."""

    trial_id: str
    feature: FeatureBuilder  # must already be shifted so only PRIOR data informs it
    denominator: int  # decile -> 10, quintile -> 5
    minimum_names: int
    ascending: bool = False  # False: select the LARGEST feature values


def portfolio_stream(
    returns: pd.DataFrame,
    feature: pd.DataFrame,
    adv: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
    *,
    denominator: int,
    minimum_names: int,
    ascending: bool,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Daily gross return, per-session cost, and diagnostics for one trial."""

    sessions = returns.index
    gross = pd.Series(np.nan, index=sessions)
    cost = pd.Series(0.0, index=sessions)
    held: list[str] = []
    rebalances = 0
    name_counts: list[int] = []
    for start, end in itertools.pairwise(month_ends):
        row = feature.loc[start].dropna()
        eligible = row.index[returns.loc[start, row.index].notna()]
        count = len(eligible) // denominator
        if count >= minimum_names:
            ranked = row.loc[eligible]
            chosen = ranked.nsmallest(count) if ascending else ranked.nlargest(count)
            selected = list(chosen.index)
        else:
            selected = []
        entering = sorted(set(selected) - set(held))
        leaving = sorted(set(held) - set(selected))
        if entering or leaving:
            rebalances += 1
            for name, size in [(n, len(selected)) for n in entering] + [
                (n, max(len(held), 1)) for n in leaving
            ]:
                adv_value = (
                    float(adv.at[start, name]) if name in adv.columns else np.nan
                )
                cost.at[start] += flip_cost_fraction(
                    1.0 / max(size, 1),
                    adv_value if np.isfinite(adv_value) else 1e8,
                    is_etf=False,
                )
        held = selected
        if not held:
            continue
        name_counts.append(len(held))
        window = sessions[(sessions > start) & (sessions <= end)]
        member_returns = returns.loc[window, held].fillna(0.0)
        drift = (1.0 + member_returns).cumprod().shift(1).fillna(1.0)
        weights = drift.div(drift.sum(axis=1), axis=0)
        gross.loc[window] = (weights * member_returns).sum(axis=1)
    diagnostics = {
        "rebalances": rebalances,
        "average_names": float(np.mean(name_counts)) if name_counts else 0.0,
        "active_fraction": float(gross.notna().mean()),
    }
    return gross, cost, diagnostics


def build_xsec_streams(
    panel: Mapping[str, pd.DataFrame],
    trials: Sequence[XsecTrial],
    end_exclusive: date,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], pd.Series]:
    """Equity sub-panel, one stream per declared trial, SPY benchmark."""

    asset_types = cast(pd.Series, panel["asset_types"])
    equities = [symbol for symbol, kind in asset_types.items() if kind == "equity"]
    adjusted = panel["adjusted_close"].loc[
        panel["adjusted_close"].index < pd.Timestamp(end_exclusive)
    ]
    close = panel["close"].reindex(adjusted.index)
    volume = panel["volume"].reindex(adjusted.index)
    returns = adjusted.pct_change(fill_method=None)
    equity_returns = returns[equities]
    adv = (
        close[equities].mul(volume[equities]).rolling(20, min_periods=1).mean().shift(1)
    )
    month_ends = pd.DatetimeIndex(
        adjusted.groupby(pd.PeriodIndex(adjusted.index, freq="M")).tail(1).index
    )
    truncated_panel: dict[str, pd.DataFrame] = {
        "adjusted_close": adjusted,
        "close": close,
        "volume": volume,
    }
    gross_streams: dict[str, pd.Series] = {}
    net_streams: dict[str, pd.Series] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for trial in trials:
        feature = trial.feature(truncated_panel, equities)
        gross, cost, diagnostics = portfolio_stream(
            equity_returns,
            feature,
            adv,
            month_ends,
            denominator=trial.denominator,
            minimum_names=trial.minimum_names,
            ascending=trial.ascending,
        )
        net = gross.fillna(0.0) - cost
        net = net.where(gross.notna() | (cost > 0))
        gross_streams[trial.trial_id] = gross
        net_streams[trial.trial_id] = net
        definitions[trial.trial_id] = {
            "signal": trial.trial_id.split("|", 1)[-1],
            **diagnostics,
        }
    benchmark = returns["SPY"]
    return (
        pd.DataFrame(gross_streams),
        pd.DataFrame(net_streams),
        definitions,
        benchmark,
    )
