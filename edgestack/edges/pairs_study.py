"""Preholdout evaluator for the preregistered ETF pairs family.

Implements configs/pairs-study-v1.yaml exactly: distance-method pairs on the
nine sealed liquid ETFs — 252-session formation immediately preceding each
contiguous 126-session trading period, top-K pairs by formation SSD, entries
at the declared sigma thresholds, exits on the first spread zero-crossing or
the final trading session. Decisions use closes through the prior session;
fills are next-session MOC; costs are charged per leg per fill plus the
declared flat borrow fee on open short legs. The forward holdout window is
never read.
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

_FORMATION_SESSIONS = 252
_TRADING_SESSIONS = 126
_TRIALS: tuple[tuple[str, int, float], ...] = (
    ("pairs_top5_z20", 5, 2.0),
    ("pairs_top5_z15", 5, 1.5),
    ("pairs_top10_z20", 10, 2.0),
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pairs study configuration must be a mapping")
    family = cast(Mapping[str, Any], payload["declared_family"])
    if int(family["real_trial_count"]) != len(_TRIALS):
        raise ValueError("declared trial count does not match the preregistration")
    return cast(dict[str, Any], payload)


def _cycle_bounds(n_sessions: int) -> list[tuple[int, int, int]]:
    """(formation_start, trade_start, trade_end_exclusive) index triples."""

    bounds = []
    trade_start = _FORMATION_SESSIONS
    while trade_start < n_sessions:
        trade_end = min(trade_start + _TRADING_SESSIONS, n_sessions)
        bounds.append((trade_start - _FORMATION_SESSIONS, trade_start, trade_end))
        trade_start = trade_end
    return bounds


def _trade_cycle(
    normalized: pd.DataFrame,
    returns: pd.DataFrame,
    adv: pd.DataFrame,
    pairs: list[tuple[str, str, float]],
    trade_index: pd.DatetimeIndex,
    *,
    threshold: float,
    per_leg_weight: float,
    daily_borrow: float,
) -> tuple[pd.Series, pd.Series, int]:
    """Gross and cost streams for one cycle; returns round-trip count too."""

    gross = pd.Series(0.0, index=trade_index)
    open_mask = pd.Series(False, index=trade_index)
    cost = pd.Series(0.0, index=trade_index)
    round_trips = 0
    for leg_a, leg_b, sigma in pairs:
        spread = normalized[leg_a].loc[trade_index] - normalized[leg_b].loc[trade_index]
        state = 0
        pending: int | None = None
        for position, session in enumerate(trade_index):
            if state != 0:
                # Exposure during this session was set at the prior close.
                gross.at[session] += (
                    state
                    * (returns.at[session, leg_a] - returns.at[session, leg_b])
                    * per_leg_weight
                )
                cost.at[session] += daily_borrow * per_leg_weight
                open_mask.at[session] = True
            if pending is not None:
                if pending == 0:
                    round_trips += 1
                for leg in (leg_a, leg_b):
                    adv_value = float(adv.at[session, leg])
                    cost.at[session] += flip_cost_fraction(
                        per_leg_weight,
                        adv_value if np.isfinite(adv_value) else 1e8,
                        is_etf=True,
                    )
                state, pending = pending, None
            is_last = position == len(trade_index) - 1
            if is_last:
                if state != 0:
                    round_trips += 1
                    for leg in (leg_a, leg_b):
                        adv_value = float(adv.at[session, leg])
                        cost.at[session] += flip_cost_fraction(
                            per_leg_weight,
                            adv_value if np.isfinite(adv_value) else 1e8,
                            is_etf=True,
                        )
                    state = 0
                continue
            value = float(spread.at[session])
            if state == 0 and pending is None and sigma > 0.0:
                if value > threshold * sigma:
                    pending = -1  # short the leader (leg_a), long the laggard
                elif value < -threshold * sigma:
                    pending = 1
            elif (
                state != 0
                and pending is None
                and ((state == -1 and value <= 0.0) or (state == 1 and value >= 0.0))
            ):
                pending = 0
    return gross.where(open_mask | (cost > 0)), cost, round_trips


def build_streams(
    config: Mapping[str, Any],
    panel: Mapping[str, pd.DataFrame],
    end_exclusive: date,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], pd.Series]:
    family = cast(Mapping[str, Any], config["declared_family"])
    instruments = [str(item) for item in family["instruments"]]
    daily_borrow = float(family["short_borrow_fee_annual_bps"]) / 1e4 / 252.0
    adjusted_all = panel["adjusted_close"].loc[
        panel["adjusted_close"].index < pd.Timestamp(end_exclusive)
    ]
    adjusted = adjusted_all[instruments].dropna(how="any")
    close = panel["close"].reindex(adjusted.index)[instruments]
    volume = panel["volume"].reindex(adjusted.index)[instruments]
    returns = adjusted.pct_change(fill_method=None).fillna(0.0)
    adv = close.mul(volume).rolling(20, min_periods=1).mean().shift(1)
    sessions = adjusted.index
    bounds = _cycle_bounds(len(sessions))
    all_pairs = list(itertools.combinations(instruments, 2))
    gross_streams: dict[str, pd.Series] = {}
    net_streams: dict[str, pd.Series] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for trial_name, top_k, threshold in _TRIALS:
        gross = pd.Series(np.nan, index=sessions)
        cost = pd.Series(0.0, index=sessions)
        cycles = 0
        round_trips = 0
        per_leg_weight = 1.0 / top_k
        for formation_start, trade_start, trade_end in bounds:
            formation_index = sessions[formation_start:trade_start]
            trade_index = sessions[trade_start:trade_end]
            base = adjusted.iloc[formation_start]
            normalized = adjusted.div(base, axis=1)
            ranked: list[tuple[float, str, str, float]] = []
            for leg_a, leg_b in all_pairs:
                spread = (
                    normalized[leg_a].loc[formation_index]
                    - normalized[leg_b].loc[formation_index]
                )
                ranked.append(
                    (
                        float((spread**2).sum()),
                        leg_a,
                        leg_b,
                        float(spread.std(ddof=1)),
                    )
                )
            ranked.sort(key=lambda item: item[0])
            selected = [
                (leg_a, leg_b, sigma) for _, leg_a, leg_b, sigma in ranked[:top_k]
            ]
            cycles += 1
            cycle_gross, cycle_cost, cycle_trips = _trade_cycle(
                normalized,
                returns,
                adv,
                selected,
                trade_index,
                threshold=threshold,
                per_leg_weight=per_leg_weight,
                daily_borrow=daily_borrow,
            )
            gross.loc[trade_index] = cycle_gross
            cost.loc[trade_index] = cycle_cost
            round_trips += cycle_trips
        net = gross.fillna(0.0) - cost
        net = net.where(gross.notna() | (cost > 0))
        trial_id = f"pairs|{trial_name}"
        gross_streams[trial_id] = gross
        net_streams[trial_id] = net
        definitions[trial_id] = {
            "signal": trial_name,
            "cycles": cycles,
            "round_trips": round_trips,
            "active_fraction": float(gross.notna().mean()),
        }
    benchmark = adjusted_all["SPY"].pct_change(fill_method=None)
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
    parser.add_argument("--config", default="configs/pairs-study-v1.yaml")
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
