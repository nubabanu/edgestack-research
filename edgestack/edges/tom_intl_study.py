"""Preholdout evaluator for the international turn-of-month family.

Implements configs/tom-intl-v1.yaml exactly: the FROZEN validated SPY rule
(SPY_TURN_OF_MONTH_LAST1_FIRST3 — entry MOC on the session immediately
before the final session of each month, exposure on the final session plus
the first three sessions of the next month, exit MOC on the third) applied,
unchanged, to 12 US-listed country ETFs. Costs are charged per fill at the
entry and exit sessions. The forward holdout window is never read.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from edgestack.data.intl_panel import load_panel
from edgestack.disclaimer import DISCLAIMER
from edgestack.edges._study_common import evaluate_family, flip_cost_fraction

_EXPOSURE_NEXT_MONTH_SESSIONS = 3


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("intl TOM configuration must be a mapping")
    family = cast(Mapping[str, Any], payload["declared_family"])
    if int(family["real_trial_count"]) != 12:
        raise ValueError("declared trial count does not match the preregistration")
    return cast(dict[str, Any], payload)


def tom_exposure_mask(sessions: pd.DatetimeIndex) -> pd.Series:
    """LAST1_FIRST3: final session of each month + first three of the next.

    The final month in the sample is excluded (its month-end exposure would
    extend past the data), which also keeps truncation causal.
    """

    mask = np.zeros(len(sessions), dtype=bool)
    periods = pd.PeriodIndex(sessions, freq="M")
    last_of_month = np.flatnonzero(periods[:-1] != periods[1:])
    for position in last_of_month:
        end = min(position + 1 + _EXPOSURE_NEXT_MONTH_SESSIONS, len(sessions))
        mask[position:end] = True
    return pd.Series(mask, index=sessions)


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
    returns = adjusted.pct_change(fill_method=None)
    adv = close.mul(volume).rolling(20, min_periods=1).mean().shift(1)
    mask = tom_exposure_mask(pd.DatetimeIndex(adjusted.index))
    gross_streams: dict[str, pd.Series] = {}
    net_streams: dict[str, pd.Series] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for instrument in instruments:
        available = adjusted[instrument].notna()
        held = mask & available & available.shift(1, fill_value=False)
        gross = returns[instrument].where(held)
        cost = pd.Series(0.0, index=adjusted.index)
        held_values = held.to_numpy()
        starts = np.flatnonzero(held_values & ~np.roll(held_values, 1))
        ends = np.flatnonzero(held_values & ~np.roll(held_values, -1))
        if len(held_values) and held_values[0]:
            starts = starts[starts != 0]  # no observable entry fill for day one
        events = 0
        for block_start, block_end in zip(starts, ends, strict=False):
            events += 1
            for fill_index in (block_start - 1, block_end):
                if fill_index < 0:
                    continue
                session = adjusted.index[fill_index]
                adv_value = float(adv[instrument].iloc[fill_index])
                cost.at[session] += flip_cost_fraction(
                    1.0,
                    adv_value if np.isfinite(adv_value) else 1e8,
                    is_etf=True,
                )
        net = gross.fillna(0.0) - cost
        net = net.where(gross.notna() | (cost > 0))
        trial_id = f"tomintl|{instrument}"
        gross_streams[trial_id] = gross
        net_streams[trial_id] = net
        definitions[trial_id] = {
            "instrument": instrument,
            "signal": "turn_of_month_last1_first3",
            "events": events,
            "active_fraction": float(held.mean()),
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
    panel = load_panel(base)
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
    parser.add_argument("--config", default="configs/tom-intl-v1.yaml")
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
