"""Preholdout evaluator for the preregistered VIX risk-premium family.

Implements configs/vix-study-v1.yaml exactly: 6 real trials — SPY exposure
gated by the EXPANDING causal percentile of FRED VIXCLS at thresholds
{0.70, 0.80, 0.90} in both declared responses (RISK_OFF holds below the
threshold, RISK_SEEK holds at/above it). Costs are charged per threshold
crossing. The forward holdout window is never read.
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

_FACTORS = (
    "artifacts/campaigns/full-stooq-literature-v2-20260715-001/data/factors.parquet"
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("vix study configuration must be a mapping")
    family = cast(Mapping[str, Any], payload["declared_family"])
    if int(family["real_trial_count"]) != 6:
        raise ValueError("declared trial count does not match the preregistration")
    return cast(dict[str, Any], payload)


def _expanding_percentile(series: pd.Series, minimum: int = 252) -> pd.Series:
    """Causal percentile of each value within its own expanding history."""

    values = series.to_numpy(dtype=float)
    output = np.full(values.size, np.nan)
    sorted_history: list[float] = []
    import bisect

    for index, value in enumerate(values):
        if np.isfinite(value):
            if len(sorted_history) >= minimum:
                # Mid-rank between ties: repeated index values (common for
                # VIX) rank in the middle of their equals, not at the top.
                low = bisect.bisect_left(sorted_history, value)
                high = bisect.bisect_right(sorted_history, value)
                output[index] = ((low + high) / 2.0) / (len(sorted_history) + 1)
            bisect.insort(sorted_history, value)
    return pd.Series(output, index=series.index)


def build_streams(
    config: Mapping[str, Any],
    panel: Mapping[str, pd.DataFrame],
    vix: pd.Series,
    end_exclusive: date,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], pd.Series]:
    family = cast(Mapping[str, Any], config["declared_family"])
    thresholds = [float(value) for value in family["thresholds"]]
    adjusted = panel["adjusted_close"].loc[
        panel["adjusted_close"].index < pd.Timestamp(end_exclusive)
    ]
    sessions = pd.DatetimeIndex(adjusted.index)
    spy_returns = adjusted["SPY"].pct_change(fill_method=None)
    close = panel["close"]["SPY"].reindex(sessions)
    volume = panel["volume"]["SPY"].reindex(sessions)
    adv = close.mul(volume).rolling(20, min_periods=1).mean().shift(1)
    percentile = _expanding_percentile(vix.reindex(sessions))
    gross_streams: dict[str, pd.Series] = {}
    net_streams: dict[str, pd.Series] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        below = percentile < threshold
        for response in ("RISK_OFF", "RISK_SEEK"):
            decided = below if response == "RISK_OFF" else ~below
            decided = decided & percentile.notna()
            # Decision at the close; exposure starts the next session.
            position = decided.shift(1, fill_value=False)
            gross = spy_returns.where(position.astype(bool))
            flips = decided.astype(int).diff().abs().fillna(0.0)
            flip_sessions = flips[flips > 0].index
            cost = pd.Series(0.0, index=sessions)
            for session in flip_sessions:
                adv_value = float(adv.get(session, np.nan))
                cost.at[session] = flip_cost_fraction(
                    1.0,
                    adv_value if np.isfinite(adv_value) else 1e8,
                    is_etf=True,
                )
            net = gross.fillna(0.0) - cost
            net = net.where(gross.notna() | (cost > 0))
            trial_id = f"vix|{threshold:.2f}|{response}"
            gross_streams[trial_id] = gross
            net_streams[trial_id] = net
            definitions[trial_id] = {
                "threshold": threshold,
                "response": response,
                "flips": int(len(flip_sessions)),
                "active_fraction": float(position.mean()),
            }
    return (
        pd.DataFrame(gross_streams),
        pd.DataFrame(net_streams),
        definitions,
        spy_returns,
    )


def run_preholdout(config_path: str | Path, *, root: str | Path = ".") -> Path:
    base = Path(root).resolve()
    config = _load_config(base / config_path)
    forward_start = date.fromisoformat(
        str(cast(Mapping[str, Any], config["holdout"])["start"])
    )
    panel = _load_panel(base)
    factors = pd.read_parquet(base / _FACTORS, columns=["session", "VIXCLS"])
    factors["session"] = pd.to_datetime(factors["session"])
    vix = factors.set_index("session")["VIXCLS"].astype(float).sort_index()
    gross, net, definitions, benchmark = build_streams(
        config, panel, vix, forward_start
    )

    def rebuild(end_exclusive: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        rebuilt_gross, rebuilt_net, _, _ = build_streams(
            config, panel, vix, end_exclusive
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
    parser.add_argument("--config", default="configs/vix-study-v1.yaml")
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
