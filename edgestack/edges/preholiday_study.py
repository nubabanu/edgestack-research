"""Preholdout evaluator for the preregistered pre-holiday family.

Implements configs/preholiday-study-v1.yaml exactly: 3 real trials (SPY,
QQQ, IWM), long ONLY the session immediately preceding an exchange holiday
— buy MOC one session earlier, sell MOC on the pre-holiday close — with one
flip cost per fill. Holiday timing is pure calendar information published
years in advance, so the mask is causal by construction; the final session
of a sample has no observable successor and is never classified. The
forward holdout window is never read.
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
from pandas.tseries.offsets import BDay

from edgestack.disclaimer import DISCLAIMER
from edgestack.edges._study_common import evaluate_family, flip_cost_fraction
from edgestack.edges.overnight_study import _load_panel


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pre-holiday study configuration must be a mapping")
    family = cast(Mapping[str, Any], payload["declared_family"])
    if int(family["real_trial_count"]) != 3:
        raise ValueError("declared trial count does not match the preregistration")
    return cast(dict[str, Any], payload)


def pre_holiday_mask(sessions: pd.DatetimeIndex) -> pd.Series:
    """True where at least one WEEKDAY between t and the next session is closed.

    A plain weekend gap (Fri -> Mon) is not a holiday; a gap longer than the
    business-day calendar implies an exchange holiday. The last session has
    no observable successor and is conservatively False.
    """

    mask = pd.Series(False, index=sessions)
    expected = sessions[:-1] + BDay(1)
    mask.iloc[:-1] = sessions[1:] > expected
    return mask


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
    mask = pre_holiday_mask(pd.DatetimeIndex(adjusted.index))
    gross_streams: dict[str, pd.Series] = {}
    net_streams: dict[str, pd.Series] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for instrument in instruments:
        available = adjusted[instrument].notna()
        held = mask & available & available.shift(1, fill_value=False)
        gross = returns[instrument].where(held)
        cost = pd.Series(0.0, index=adjusted.index)
        is_etf = asset_types.get(instrument) == "etf"
        positions = np.flatnonzero(held.to_numpy())
        for position in positions:
            for fill_index in (position - 1, position):  # entry MOC, exit MOC
                if fill_index < 0:
                    continue
                session = adjusted.index[fill_index]
                adv_value = float(adv[instrument].iloc[fill_index])
                cost.at[session] += flip_cost_fraction(
                    1.0,
                    adv_value if np.isfinite(adv_value) else 1e8,
                    is_etf=is_etf,
                )
        net = gross.fillna(0.0) - cost
        net = net.where(gross.notna() | (cost > 0))
        trial_id = f"preholiday|{instrument}"
        gross_streams[trial_id] = gross
        net_streams[trial_id] = net
        definitions[trial_id] = {
            "instrument": instrument,
            "signal": "pre_holiday_session",
            "events": int(held.sum()),
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
    parser.add_argument("--config", default="configs/preholiday-study-v1.yaml")
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
