"""Preholdout evaluator for the preregistered low-volatility family.

Implements configs/lowvol-study-v1.yaml exactly: 3 real trials on the sealed
(survivorship-biased, stamped) equity panel — bottom-decile portfolios by
trailing 252-session volatility, 63-session volatility, and 252-session beta
versus SPY. Equal-weight entry at each month-end close with buy-and-hold
weight drift inside the month, versus zero-return cash; per-name flip costs
at each rebalance. The forward holdout window is never read.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from edgestack.disclaimer import DISCLAIMER
from edgestack.edges._study_common import evaluate_family, flip_cost_fraction
from edgestack.edges.overnight_study import _load_panel

_SIGNALS = ("vol_252_bottom_decile", "vol_63_bottom_decile", "beta_252_bottom_decile")
_MINIMUM_DECILE_NAMES = 30


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("low-vol study configuration must be a mapping")
    family = cast(Mapping[str, Any], payload["declared_family"])
    if int(family["real_trial_count"]) != 3:
        raise ValueError("declared trial count does not match the preregistration")
    return cast(dict[str, Any], payload)


def _signal_frame(
    returns: pd.DataFrame, spy_returns: pd.Series, signal: str
) -> pd.DataFrame:
    """Per-name ranking feature, shifted so only PRIOR sessions inform it."""

    if signal == "vol_252_bottom_decile":
        feature = returns.rolling(252, min_periods=252).std()
    elif signal == "vol_63_bottom_decile":
        feature = returns.rolling(63, min_periods=63).std()
    elif signal == "beta_252_bottom_decile":
        covariance = returns.rolling(252, min_periods=252).cov(spy_returns)
        variance = spy_returns.rolling(252, min_periods=252).var()
        feature = covariance.div(variance, axis=0)
    else:
        raise ValueError(f"unknown signal {signal}")
    return feature.shift(1)


def _portfolio_stream(
    returns: pd.DataFrame,
    feature: pd.DataFrame,
    adv: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Gross daily portfolio return, per-session cost, and trial diagnostics."""

    sessions = returns.index
    gross = pd.Series(np.nan, index=sessions)
    cost = pd.Series(0.0, index=sessions)
    held: list[str] = []
    rebalances = 0
    name_counts: list[int] = []
    for start, end in itertools.pairwise(month_ends):
        row = feature.loc[start].dropna()
        eligible = row.index[returns.loc[start, row.index].notna()]
        decile = len(eligible) // 10
        if decile >= _MINIMUM_DECILE_NAMES:
            selected = list(row.loc[eligible].nsmallest(decile).index)
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


def build_streams(
    config: Mapping[str, Any],
    panel: Mapping[str, pd.DataFrame],
    end_exclusive: date,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], pd.Series]:
    asset_types = cast(pd.Series, panel["asset_types"])
    equities = [symbol for symbol, kind in asset_types.items() if kind == "equity"]
    adjusted = panel["adjusted_close"].loc[
        panel["adjusted_close"].index < pd.Timestamp(end_exclusive)
    ]
    close = panel["close"].reindex(adjusted.index)
    volume = panel["volume"].reindex(adjusted.index)
    returns = adjusted.pct_change(fill_method=None)
    equity_returns = returns[equities]
    spy_returns = returns["SPY"]
    adv = (
        close[equities].mul(volume[equities]).rolling(20, min_periods=1).mean().shift(1)
    )
    month_ends = pd.DatetimeIndex(
        adjusted.groupby(pd.PeriodIndex(adjusted.index, freq="M")).tail(1).index
    )
    gross_streams: dict[str, pd.Series] = {}
    net_streams: dict[str, pd.Series] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for signal in _SIGNALS:
        feature = _signal_frame(equity_returns, spy_returns, signal)
        gross, cost, diagnostics = _portfolio_stream(
            equity_returns, feature, adv, month_ends
        )
        net = gross.fillna(0.0) - cost
        net = net.where(gross.notna() | (cost > 0))
        trial_id = f"lowvol|{signal}"
        gross_streams[trial_id] = gross
        net_streams[trial_id] = net
        definitions[trial_id] = {"signal": signal, **diagnostics}
    benchmark = spy_returns
    return (
        pd.DataFrame(gross_streams),
        pd.DataFrame(net_streams),
        definitions,
        benchmark,
    )


def run_preholdout(config_path: str | Path, *, root: str | Path = ".") -> Path:
    base = Path(root).resolve()
    config = _load_config(base / config_path)
    forward_start = date.fromisoformat(
        str(cast(Mapping[str, Any], config["holdout"])["start"])
    )
    panel = _load_panel(base)
    gross, net, definitions, benchmark = build_streams(config, panel, forward_start)

    def rebuild(end_exclusive: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        rebuilt_gross, rebuilt_net, _, _ = build_streams(config, panel, end_exclusive)
        return rebuilt_gross, rebuilt_net

    return evaluate_family(
        campaign_id=str(config["campaign_id"]),
        config_path=base / config_path,
        root=base,
        net=net,
        gross=gross,
        definitions=definitions,
        accounting_family_size=int(
            cast(Mapping[str, Any], config["declared_family"])["accounting_family_size"]
        ),
        forward_start=forward_start,
        rebuild=rebuild,
        benchmark=benchmark,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preholdout",))
    parser.add_argument("--config", default="configs/lowvol-study-v1.yaml")
    parser.add_argument("--root", default=".")
    arguments = parser.parse_args(argv)
    path = run_preholdout(arguments.config, root=arguments.root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(DISCLAIMER)
    print(
        json.dumps(
            {
                "preholdout_pass": payload["preholdout_pass"],
                "survivors": payload["survivors"],
                "family_tests": payload["family_tests"],
                "placebos": payload["placebos"],
                "result": str(path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
