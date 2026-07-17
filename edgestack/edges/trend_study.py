"""Preholdout evaluator for the preregistered trend/TSMOM family.

Implements configs/trend-study-v1.yaml exactly: 27 real trials (9 index and
sector ETFs x 3 monthly trend signals), long-only vs zero-return cash,
positions changed ONLY at month-end closes, costs charged per flip. The
forward holdout window is never read.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import yaml

from edgestack.disclaimer import DISCLAIMER
from edgestack.edges._study_common import evaluate_family, flip_cost_fraction
from edgestack.edges.overnight_study import _load_panel

_SIGNALS = ("tsmom_12_1", "above_sma200", "above_10m_sma")


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("trend study configuration must be a mapping")
    family = cast(Mapping[str, Any], payload["declared_family"])
    if int(family["real_trial_count"]) != 27:
        raise ValueError("declared trial count does not match the preregistration")
    return cast(dict[str, Any], payload)


def _monthly_signal(
    daily_adjusted: pd.Series, daily_close: pd.Series, signal: str
) -> pd.Series:
    """Boolean long/flat decision at each month-end close (causal)."""

    month_end = daily_adjusted.groupby(
        pd.PeriodIndex(daily_adjusted.index, freq="M")
    ).tail(1)
    if signal == "tsmom_12_1":
        decided = month_end.shift(1) / month_end.shift(12) > 1.0
    elif signal == "above_sma200":
        sma = daily_adjusted.rolling(200, min_periods=200).mean()
        decided = daily_adjusted.loc[month_end.index] > sma.loc[month_end.index]
    elif signal == "above_10m_sma":
        decided = month_end > month_end.rolling(10, min_periods=10).mean()
    else:
        raise ValueError(f"unknown signal {signal}")
    return decided.fillna(False).astype(bool)


def build_streams(
    config: Mapping[str, Any],
    panel: Mapping[str, pd.DataFrame],
    end_exclusive: date,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], pd.Series]:
    family = cast(Mapping[str, Any], config["declared_family"])
    instruments = [str(item) for item in family["instruments"]]
    adjusted = panel["adjusted_close"].loc[
        panel["adjusted_close"].index < pd.Timestamp(end_exclusive)
    ]
    close = panel["close"].reindex(adjusted.index)
    volume = panel["volume"].reindex(adjusted.index)
    asset_types = cast(pd.Series, panel["asset_types"])
    returns = adjusted.pct_change(fill_method=None)
    adv = close.mul(volume).rolling(20, min_periods=1).mean().shift(1)
    gross_streams: dict[str, pd.Series] = {}
    net_streams: dict[str, pd.Series] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for instrument in instruments:
        series = adjusted[instrument].dropna()
        instrument_returns = returns[instrument]
        for signal in _SIGNALS:
            decided = _monthly_signal(series, close[instrument], signal)
            # Entered at the month-end MOC; exposure starts the NEXT session.
            daily_position = (
                decided.reindex(adjusted.index).ffill().fillna(False).shift(
                    1, fill_value=False
                )
            )
            gross = instrument_returns.where(daily_position.astype(bool))
            flips = decided.astype(int).diff().abs().fillna(0.0)
            flip_sessions = flips[flips > 0].index
            cost = pd.Series(0.0, index=adjusted.index)
            for session in flip_sessions:
                adv_value = float(adv[instrument].get(session, np.nan))
                cost.at[session] = flip_cost_fraction(
                    1.0,
                    adv_value if np.isfinite(adv_value) else 1e8,
                    is_etf=asset_types.get(instrument) == "etf",
                )
            # A flip cost is paid whether entering or exiting; charge it on
            # the flip session against whatever exposure exists that day.
            net = gross.fillna(0.0) - cost
            net = net.where(gross.notna() | (cost > 0))
            trial_id = f"trend|{instrument}|{signal}"
            gross_streams[trial_id] = gross
            net_streams[trial_id] = net
            definitions[trial_id] = {
                "instrument": instrument,
                "signal": signal,
                "flips": int(len(flip_sessions)),
                "active_fraction": float(daily_position.mean()),
            }
    benchmark = returns["SPY"]
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
        rebuilt_gross, rebuilt_net, _, _ = build_streams(
            config, panel, end_exclusive
        )
        return rebuilt_gross, rebuilt_net

    return evaluate_family(
        campaign_id=str(config["campaign_id"]),
        config_path=base / config_path,
        root=base,
        net=net,
        gross=gross,
        definitions=definitions,
        accounting_family_size=int(
            cast(Mapping[str, Any], config["declared_family"])[
                "accounting_family_size"
            ]
        ),
        forward_start=forward_start,
        rebuild=rebuild,
        benchmark=benchmark,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preholdout",))
    parser.add_argument("--config", default="configs/trend-study-v1.yaml")
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
