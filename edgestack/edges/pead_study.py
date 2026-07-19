"""Preholdout evaluator for the preregistered PEAD family (v2, EDGAR-fed).

Implements configs/pead-study-v2.yaml exactly: 3 real trials on the sealed
(survivorship-biased, stamped) equity panel using the free SEC EDGAR earnings
feed. Standardized unexpected earnings follow the Bernard-Thomas seasonal
random walk — EPS this quarter minus EPS four quarters earlier, scaled by the
standard deviation of the trailing eight such differences — so no analyst
consensus is required. A SUE becomes usable only at the first session
STRICTLY AFTER its announcement availability moment (8-K Item 2.02 EDGAR
acceptance; XBRL filing date fallback), and portfolios form at month-end
closes exactly like the other cross-sectional studies. The forward holdout
window is never read.
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

from edgestack.data.edgar_earnings import load_events
from edgestack.disclaimer import DISCLAIMER
from edgestack.edges._study_common import evaluate_family, flip_cost_fraction
from edgestack.edges.overnight_study import _load_panel

_TRIALS: tuple[tuple[str, int, int, int], ...] = (
    # (name, staleness_sessions, selection_denominator, minimum_names)
    ("sue_top_decile_63d", 63, 10, 30),
    ("sue_top_quintile_63d", 63, 5, 30),
    ("sue_top_quintile_21d", 21, 5, 20),
)
_ANNOUNCEMENT_WINDOW_DAYS = 90
_MINIMUM_SUE_HISTORY = 6
# EPS-difference std below this is numerically degenerate (floating-point
# residue of a constant series); a 0/0 SUE must never be fabricated from it.
_MINIMUM_SCALE = 1e-6


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PEAD study configuration must be a mapping")
    family = cast(Mapping[str, Any], payload["declared_family"])
    if int(family["real_trial_count"]) != len(_TRIALS):
        raise ValueError("declared trial count does not match the preregistration")
    return cast(dict[str, Any], payload)


def _frame_key(frame: str) -> int | None:
    """CY2020Q1 -> comparable integer quarter index; None if not quarterly."""

    if not frame.startswith("CY") or "Q" not in frame:
        return None
    try:
        year, quarter = frame[2:].split("Q")
        return int(year) * 4 + int(quarter[0]) - 1
    except (ValueError, IndexError):
        return None


def compute_sue(eps: pd.DataFrame) -> pd.DataFrame:
    """Seasonal-random-walk SUE per (symbol, quarter) from frame-tagged EPS."""

    rows: list[dict[str, Any]] = []
    for symbol, group in eps.groupby("symbol"):
        by_key: dict[int, dict[str, Any]] = {}
        for record in group.to_dict("records"):
            key = _frame_key(str(record["frame"]))
            if key is not None and key not in by_key:
                by_key[key] = record
        for key, record in by_key.items():
            prior = by_key.get(key - 4)
            if prior is None:
                continue
            difference = float(record["value"]) - float(prior["value"])
            history = [
                float(by_key[k]["value"]) - float(by_key[k - 4]["value"])
                for k in range(key - 8, key)
                if k in by_key and (k - 4) in by_key
            ]
            if len(history) < _MINIMUM_SUE_HISTORY:
                continue
            scale = float(np.std(history, ddof=1))
            if not np.isfinite(scale) or scale <= _MINIMUM_SCALE:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "quarter_end": pd.Timestamp(record["end"]),
                    "filed": pd.Timestamp(record["filed"]),
                    "sue": difference / scale,
                }
            )
    return pd.DataFrame(rows)


def attach_availability(sue: pd.DataFrame, announcements: pd.DataFrame) -> pd.DataFrame:
    """Availability date per SUE: first 8-K 2.02 after quarter end, else filed.

    The returned ``available`` column is a normalized DATE; the usability rule
    (first session strictly after it) is what guarantees causality even for
    pre-open announcements.
    """

    events = announcements.copy()
    events["acceptance_ts"] = pd.to_datetime(events["acceptance"], utc=True)
    events["acceptance_date"] = (
        events["acceptance_ts"]
        .dt.tz_convert("America/New_York")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    merged_rows: list[dict[str, Any]] = []
    by_symbol = {symbol: group for symbol, group in events.groupby("symbol")}
    for record in sue.to_dict("records"):
        window_start = record["quarter_end"]
        window_end = window_start + pd.Timedelta(days=_ANNOUNCEMENT_WINDOW_DAYS)
        candidates = by_symbol.get(record["symbol"])
        available = pd.Timestamp(record["filed"])
        source = "XBRL_FILED_DATE"
        if candidates is not None:
            in_window = candidates.loc[
                (candidates["acceptance_date"] > window_start)
                & (candidates["acceptance_date"] <= window_end)
            ]
            if len(in_window):
                first = in_window["acceptance_date"].min()
                if first <= available:
                    available = first
                    source = "EDGAR_8K_202_ACCEPTANCE"
        merged_rows.append({**record, "available": available, "source": source})
    return pd.DataFrame(merged_rows)


def build_sue_panel(
    events: pd.DataFrame, sessions: pd.DatetimeIndex, symbols: Sequence[str]
) -> pd.DataFrame:
    """Sparse SUE frame: value lands on the first session AFTER availability."""

    frame = pd.DataFrame(np.nan, index=sessions, columns=list(symbols))
    positions = sessions.searchsorted(events["available"].to_numpy(), side="right")
    for row, position in zip(events.to_dict("records"), positions, strict=True):
        if position >= len(sessions) or row["symbol"] not in frame.columns:
            continue
        frame.iloc[position, frame.columns.get_loc(row["symbol"])] = row["sue"]
    return frame


def _portfolio_stream(
    returns: pd.DataFrame,
    feature: pd.DataFrame,
    adv: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
    *,
    denominator: int,
    minimum_names: int,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Top-fraction-by-SUE monthly portfolio; identical cost model to low-vol."""

    import itertools

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
            selected = list(row.loc[eligible].nlargest(count).index)
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
    events: pd.DataFrame,
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
    adv = (
        close[equities].mul(volume[equities]).rolling(20, min_periods=1).mean().shift(1)
    )
    month_ends = pd.DatetimeIndex(
        adjusted.groupby(pd.PeriodIndex(adjusted.index, freq="M")).tail(1).index
    )
    sparse = build_sue_panel(events, adjusted.index, equities)
    gross_streams: dict[str, pd.Series] = {}
    net_streams: dict[str, pd.Series] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for name, staleness, denominator, minimum_names in _TRIALS:
        feature = sparse.ffill(limit=staleness - 1)
        gross, cost, diagnostics = _portfolio_stream(
            equity_returns,
            feature,
            adv,
            month_ends,
            denominator=denominator,
            minimum_names=minimum_names,
        )
        net = gross.fillna(0.0) - cost
        net = net.where(gross.notna() | (cost > 0))
        trial_id = f"pead|{name}"
        gross_streams[trial_id] = gross
        net_streams[trial_id] = net
        definitions[trial_id] = {"signal": name, **diagnostics}
    benchmark = returns["SPY"]
    return (
        pd.DataFrame(gross_streams),
        pd.DataFrame(net_streams),
        definitions,
        benchmark,
    )


def _load_all_events(base: Path) -> pd.DataFrame:
    announcements, eps = load_events(base)
    if not len(eps):
        raise RuntimeError("no EDGAR EPS rows; run the earnings crawl first")
    sue = compute_sue(eps)
    if not len(sue):
        raise RuntimeError("no SUE observations could be computed")
    return attach_availability(sue, announcements)


def run_preholdout(config_path: str | Path, *, root: str | Path = ".") -> Path:
    base = Path(root).resolve()
    config = _load_config(base / config_path)
    forward_start = date.fromisoformat(
        str(cast(Mapping[str, Any], config["holdout"])["start"])
    )
    panel = _load_panel(base)
    events = _load_all_events(base)
    gross, net, definitions, benchmark = build_streams(
        config, panel, events, forward_start
    )

    def rebuild(end_exclusive: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        rebuilt_gross, rebuilt_net, _, _ = build_streams(
            config, panel, events, end_exclusive
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
            cast(Mapping[str, Any], config["declared_family"])["accounting_family_size"]
        ),
        forward_start=forward_start,
        rebuild=rebuild,
        benchmark=benchmark,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preholdout",))
    parser.add_argument("--config", default="configs/pead-study-v2.yaml")
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
