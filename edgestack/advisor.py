"""Per-instrument diagnostic timing and edge-alignment advisor.

Given one symbol's daily bars, this module reports, for the week / month /
year horizons: which calendar and state conditions historically helped
(tailwinds) or hurt (headwinds), each condition's dark side (its behavior in
the opposite trend regime, worst session, and in-condition drawdown),
multiplicity-controlled combination candidates, best/worst buy-and-sell
windows, and an "alignment" scan of upcoming sessions where several
non-redundant favorable conditions coincide.

Honesty contract:
- Everything here is DIAGNOSTIC evidence for one instrument, not a validated
  or promoted edge; nothing bypasses the campaign gauntlet.
- Daily bars support exactly two execution anchors — the opening and closing
  auctions. There is no intraday data, so no intra-hour timing is claimed.
- Every tested condition and combination counts toward one Bonferroni family;
  a combination is never a free improvement.
- News and macro feeds are not part of the free validated data layer and are
  reported as DATA_UNAVAILABLE rather than scraped.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from edgestack.data.calendars import NYSECalendar
from edgestack.disclaimer import DISCLAIMER
from edgestack.features.calendar_feats import calendar_features
from edgestack.stats.bootstrap import stationary_bootstrap_ci
from edgestack.stats.deflated_sharpe import deflated_sharpe_ratio
from edgestack.stats.tests import hac_mean_test

_WEEKDAY_NAMES = ("MON", "TUE", "WED", "THU", "FRI")
_MONTH_NAMES = (
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
)
# Combination pairs are only formed from components at least this strong on
# their own; weaker pairs would multiply the tested family without prospect.
_COMBINATION_COMPONENT_T = 1.5
_EXECUTION_GUIDANCE = {
    "supported_anchors": (
        "MOC (market-on-close; freeze the decision by 15:45 ET and never "
        "reselect from the closing print)",
        "OPENING_AUCTION (second choice; overnight gap risk is unmodeled "
        "beyond the gap checks below)",
    ),
    "intraday_hours": (
        "DATA_UNAVAILABLE: only daily bars exist, so no intra-hour timing "
        "claim is honest. Anchor to the auctions above."
    ),
    "what_to_look_for_before_entry": (
        "quote is fresh (cancel if older than the staleness limit)",
        "no gap beyond your preregistered gap fraction since the prior close",
        "pre-entry move smaller than 1 ATR against the setup (do not chase)",
        "no scheduled event (earnings/FOMC) inside the exclusion window",
        "limit-on-close reference: 0.25 x ATR from the decision price",
        "risk reference: 2 x ATR(14) stop, sized so the loss at the stop is "
        "a fixed small fraction of capital",
    ),
}


def _shrunk_mean(mean: float, t_stat: float) -> float:
    """Positive-part shrinkage: weak evidence contributes exactly zero."""

    if not (math.isfinite(mean) and math.isfinite(t_stat)) or abs(t_stat) <= 1.0:
        return 0.0
    return mean * (1.0 - 1.0 / (t_stat * t_stat))


def _max_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    wealth = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peaks - 1.0))


def _condition_stats(
    returns: pd.Series,
    mask: pd.Series,
    trend_up: pd.Series,
    *,
    name: str,
    kind: str,
) -> dict[str, Any]:
    """Two-sided conditional evidence for one condition, dark side included.

    ``regime_up_*``/``regime_down_*`` are reported for EVERY condition so a
    positive edge's losses in a down tape (its negative effects) and a
    negative condition's wins in an up tape (its positive effects) are always
    visible to the rating.
    """

    selected = returns[mask & returns.notna()]
    values = selected.to_numpy(dtype=float)
    test = hac_mean_test(values)
    up = returns[mask & trend_up & returns.notna()].to_numpy(dtype=float)
    down = returns[mask & ~trend_up & returns.notna()].to_numpy(dtype=float)
    if values.size >= 2:
        interval = stationary_bootstrap_ci(values, n_resamples=1_000, seed=42)
        ci_lower: float | None = interval.lower
        ci_upper: float | None = interval.upper
        std = float(values.std(ddof=1))
        periodic_sharpe = float(values.mean()) / std if std > 0.0 else math.nan
        centered = values - values.mean()
        skewness = float(np.mean(centered**3) / std**3) if std > 0.0 else 0.0
        kurtosis = float(np.mean(centered**4) / std**4) if std > 0.0 else 3.0
    else:
        ci_lower = ci_upper = None
        periodic_sharpe = math.nan
        skewness, kurtosis = 0.0, 3.0
    # Out-of-sample stability in the walk-forward spirit: an edge that only
    # existed in the first half of the sample, or whose recent third flipped
    # sign, is flagged as decayed (McLean-Pontiff post-publication decay).
    halves = np.array_split(values, 2) if values.size >= 4 else [values, values]
    thirds = np.array_split(values, 3) if values.size >= 6 else [values]
    first_mean = float(halves[0].mean()) if halves[0].size else math.nan
    second_mean = float(halves[1].mean()) if halves[1].size else math.nan
    recent_mean = float(thirds[-1].mean()) if thirds[-1].size else math.nan
    full_mean = float(values.mean()) if values.size else math.nan
    oos_sign_agreement = bool(
        math.isfinite(first_mean)
        and math.isfinite(second_mean)
        and np.sign(first_mean) == np.sign(second_mean) != 0
    )
    decayed = bool(
        math.isfinite(full_mean)
        and math.isfinite(recent_mean)
        and np.sign(full_mean) != np.sign(recent_mean)
    )
    return {
        "name": name,
        "kind": kind,
        "n": int(values.size),
        "hit_rate": float((values > 0).mean()) if values.size else None,
        "mean_daily": full_mean if values.size else None,
        "hac_t": test.t_stat if math.isfinite(test.t_stat) else None,
        "p_value": test.p_value if math.isfinite(test.p_value) else None,
        "shrunk_mean_daily": _shrunk_mean(
            full_mean if values.size else math.nan, test.t_stat
        ),
        "ci_lower_daily": ci_lower,
        "ci_upper_daily": ci_upper,
        "periodic_sharpe": (
            periodic_sharpe if math.isfinite(periodic_sharpe) else None
        ),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "first_half_mean": first_mean if math.isfinite(first_mean) else None,
        "second_half_mean": second_mean if math.isfinite(second_mean) else None,
        "recent_third_mean": recent_mean if math.isfinite(recent_mean) else None,
        "oos_sign_agreement": oos_sign_agreement,
        "decayed_in_recent_third": decayed,
        "worst_session": float(values.min()) if values.size else None,
        "max_drawdown_within_condition": _max_drawdown(values),
        "regime_up_mean": float(up.mean()) if up.size else None,
        "regime_up_n": int(up.size),
        "regime_down_mean": float(down.mean()) if down.size else None,
        "regime_down_n": int(down.size),
    }


def _classify(entry: dict[str, Any], family_size: int, alpha: float) -> None:
    """Attach multiplicity, selection-aware reliability, and the tier label.

    ``dsr_probability`` deflates each condition's Sharpe against the expected
    best of ``family_size`` searched trials (the gauntlet's own selection
    correction), and ``reliability_weighted_daily`` mirrors the promoted
    stack's confidence formula: reliability times shrunk magnitude. Weak,
    over-searched, or decayed conditions therefore contribute ~nothing.
    """

    p_value = entry.get("p_value")
    bonferroni = (
        min(1.0, p_value * family_size) if p_value is not None else None
    )
    entry["bonferroni_p"] = bonferroni
    periodic_sharpe = entry.get("periodic_sharpe")
    if periodic_sharpe is not None and entry["n"] >= 2:
        magnitude = deflated_sharpe_ratio(
            abs(periodic_sharpe),
            n_observations=entry["n"],
            n_trials=family_size,
            skewness=entry.get("skewness", 0.0),
            kurtosis=entry.get("kurtosis", 3.0),
        )
        entry["dsr_probability"] = magnitude if math.isfinite(magnitude) else 0.0
    else:
        entry["dsr_probability"] = 0.0
    entry["reliability_weighted_daily"] = (
        entry["shrunk_mean_daily"] * entry["dsr_probability"]
    )
    mean = entry.get("mean_daily")
    if mean is None or p_value is None:
        entry["classification"] = "INSUFFICIENT_DATA"
        return
    side = "TAILWIND" if mean > 0 else "HEADWIND"
    ci_lower = entry.get("ci_lower_daily")
    ci_upper = entry.get("ci_upper_daily")
    ci_confirmed = (
        (mean > 0 and ci_lower is not None and ci_lower > 0.0)
        or (mean < 0 and ci_upper is not None and ci_upper < 0.0)
    )
    if (
        bonferroni is not None
        and bonferroni <= alpha
        and ci_confirmed
        and entry.get("oos_sign_agreement")
        and not entry.get("decayed_in_recent_third")
    ):
        entry["classification"] = f"{side}_FAMILY_SIGNIFICANT_CI_AND_OOS_CONFIRMED"
    elif bonferroni is not None and bonferroni <= alpha:
        entry["classification"] = f"{side}_FAMILY_SIGNIFICANT"
    elif p_value <= alpha:
        entry["classification"] = f"{side}_RAW_ONLY_NOT_FAMILY_SIGNIFICANT"
    else:
        entry["classification"] = "NEUTRAL"


def _symbol_frame(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = {"symbol", "session", "adjusted_close"}
    if not required.issubset(bars.columns):
        raise ValueError("bars require symbol, session, and adjusted_close")
    selected = bars.loc[bars["symbol"].astype(str) == symbol].copy()
    if selected.empty:
        raise ValueError(f"no bars for symbol {symbol}")
    selected["session"] = pd.to_datetime(selected["session"])
    columns = ["adjusted_close"] + [
        column for column in ("open", "close") if column in selected.columns
    ]
    frame = selected.set_index("session")[columns].astype(float).sort_index()
    if frame.index.duplicated().any():
        raise ValueError(f"duplicate sessions for symbol {symbol}")
    return frame


def _anchor_assessment(frame: pd.DataFrame) -> dict[str, Any]:
    """Rate the two executable intraday anchors from the daily OHLC record.

    Daily bars contain exactly two intra-session prices — the opening and
    closing auctions — so the close-to-open (overnight) and open-to-close
    (intraday) splits are the ONLY hour-level evidence that exists. Anything
    finer (hourly, 15-minute) is DATA_UNAVAILABLE, never estimated.
    """

    if not {"open", "close"}.issubset(frame.columns):
        return {
            "status": "DATA_UNAVAILABLE",
            "reason": "bars lack raw open/close columns",
        }
    total = frame["adjusted_close"].pct_change(fill_method=None)
    intraday = frame["close"].div(frame["open"]) - 1.0
    overnight = (1.0 + total).div(1.0 + intraday) - 1.0
    legs: dict[str, Any] = {}
    for name, series in (("overnight", overnight), ("intraday", intraday)):
        values = series.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        test = hac_mean_test(values)
        legs[name] = {
            "n": int(values.size),
            "mean_daily": float(values.mean()) if values.size else None,
            "hit_rate": float((values > 0).mean()) if values.size else None,
            "hac_t": test.t_stat if math.isfinite(test.t_stat) else None,
            "shrunk_mean_daily": _shrunk_mean(
                float(values.mean()) if values.size else math.nan, test.t_stat
            ),
        }
    overnight_better = (
        legs["overnight"]["shrunk_mean_daily"] > legs["intraday"]["shrunk_mean_daily"]
    )
    return {
        "status": "TWO_ANCHORS_ONLY",
        "legs": legs,
        "best_buy_anchor": (
            "CLOSE_AUCTION (decide by 15:45 ET, fill ~16:00 ET; captures the "
            "overnight leg)"
            if overnight_better
            else "OPEN_AUCTION (~09:30 ET; captures the intraday leg)"
        ),
        "matching_sell_anchor": (
            "next OPEN_AUCTION (~09:30 ET)"
            if overnight_better
            else "same-day CLOSE_AUCTION (~16:00 ET)"
        ),
        "hourly_calendar": (
            "DATA_UNAVAILABLE beyond the two auction anchors: daily bars hold "
            "no hourly prices"
        ),
        "fifteen_minute_calendar": (
            "DATA_UNAVAILABLE: no intraday history exists in the free data "
            "layer; a 15-minute claim would be fabricated"
        ),
    }


_ANCHOR_WINDOWS = (
    ("OPEN_AUCTION", 9 * 60, 10 * 60),
    ("CLOSE_AUCTION", 15 * 60 + 30, 16 * 60),
)


def _classify_hour(hour_minute: str) -> tuple[str | None, str]:
    """Map a requested ET clock time onto an executable auction anchor."""

    try:
        hour, minute = (int(part) for part in hour_minute.split(":"))
        total = hour * 60 + minute
    except ValueError as error:
        raise ValueError("buy hour must be HH:MM (ET)") from error
    if not 0 <= total < 24 * 60:
        raise ValueError("buy hour must be a valid clock time")
    for name, start, stop in _ANCHOR_WINDOWS:
        if start <= total <= stop:
            return name, "REQUESTED_TIME_IS_AN_EXECUTABLE_ANCHOR"
    if total < 9 * 60 + 30:
        return None, "PREMARKET_UNRATED_NEAREST_ANCHOR_IS_THE_OPEN"
    if total >= 16 * 60:
        return None, "AFTER_HOURS_UNRATED_NEAREST_ANCHOR_IS_TOMORROWS_OPEN"
    return None, "MID_SESSION_UNRATED_ONLY_THE_AUCTIONS_ARE_MEASURED"


def _holidays(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Weekdays absent from the session index are exchange holidays."""

    if len(index) < 2:
        return pd.DatetimeIndex([])
    all_weekdays = pd.bdate_range(index.min(), index.max())
    return all_weekdays.difference(index)


def _condition_masks(
    calendar: pd.DataFrame,
) -> dict[str, tuple[str, pd.Series]]:
    """Deterministic calendar condition masks keyed by condition name."""

    masks: dict[str, tuple[str, pd.Series]] = {}
    for value, name in enumerate(_WEEKDAY_NAMES):
        masks[f"weekday={name}"] = ("week", calendar["weekday"].eq(value))
    masks["turn_of_month"] = ("month", calendar["turn_of_month"].astype(bool))
    masks["month_end_window"] = ("month", calendar["month_end_window"].astype(bool))
    masks["quarter_end_window"] = (
        "month",
        calendar["quarter_end_window"].astype(bool),
    )
    masks["opex_week"] = ("event", calendar["opex_week"].astype(bool))
    masks["pre_holiday"] = ("event", calendar["pre_holiday"].astype(bool))
    masks["post_holiday"] = ("event", calendar["post_holiday"].astype(bool))
    for value, name in enumerate(_MONTH_NAMES, start=1):
        masks[f"month={name}"] = ("year", calendar["month"].eq(value))
    return masks


def _state_masks(prices: pd.Series) -> dict[str, tuple[str, pd.Series, bool]]:
    """Causal instrument-state condition masks plus each state's CURRENT value.

    Every state is computed from data through the prior close only.
    """

    returns = prices.pct_change(fill_method=None)
    ma200 = prices.rolling(200, min_periods=200).mean()
    trend_up = (prices > ma200).shift(1, fill_value=False).astype(bool)
    momentum = (prices.shift(21) / prices.shift(252) - 1.0).shift(1)
    momentum_positive = momentum.gt(0.0).fillna(False)
    week_return = (prices / prices.shift(5) - 1.0).shift(1)
    oversold = week_return.lt(0.0).fillna(False)
    vol = returns.rolling(21, min_periods=21).std().shift(1)
    vol_terciles = vol.dropna().quantile([1.0 / 3.0, 2.0 / 3.0])
    low_vol = vol.le(vol_terciles.iloc[0]).fillna(False)
    high_vol = vol.gt(vol_terciles.iloc[1]).fillna(False)
    states = {
        "state:trend_above_ma200": ("state", trend_up, bool(trend_up.iloc[-1])),
        "state:trend_below_ma200": ("state", ~trend_up, not bool(trend_up.iloc[-1])),
        "state:momentum_12_1_positive": (
            "state",
            momentum_positive,
            bool(momentum_positive.iloc[-1]),
        ),
        "state:prior_week_negative": ("state", oversold, bool(oversold.iloc[-1])),
        "state:volatility_low_tercile": ("state", low_vol, bool(low_vol.iloc[-1])),
        "state:volatility_high_tercile": ("state", high_vol, bool(high_vol.iloc[-1])),
    }
    return states


def _combination_entries(
    returns: pd.Series,
    trend_up: pd.Series,
    candidates: list[tuple[str, pd.Series, dict[str, Any]]],
    *,
    min_observations: int,
) -> list[dict[str, Any]]:
    """Pairwise gating combinations among strong, cross-kind components.

    Redundancy (the "correlated voters" trap) is surfaced as the overlap of
    the two condition masks; an INCREMENTAL verdict additionally requires the
    joint mean to beat BOTH components — the pairwise ablation criterion.
    """

    combinations: list[dict[str, Any]] = []
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            left_name, left_mask, left_entry = candidates[left_index]
            right_name, right_mask, right_entry = candidates[right_index]
            if left_entry["kind"] == right_entry["kind"]:
                continue
            joint_mask = left_mask & right_mask
            joint = _condition_stats(
                returns,
                joint_mask,
                trend_up,
                name=f"{left_name} AND {right_name}",
                kind="combination",
            )
            overlap_base = min(int(left_mask.sum()), int(right_mask.sum()))
            joint["component_names"] = [left_name, right_name]
            joint["component_overlap_fraction"] = (
                float(joint_mask.sum() / overlap_base) if overlap_base else None
            )
            joint["incremental_vs_components"] = (
                joint["mean_daily"] is not None
                and left_entry["mean_daily"] is not None
                and right_entry["mean_daily"] is not None
                and joint["mean_daily"] > left_entry["mean_daily"]
                and joint["mean_daily"] > right_entry["mean_daily"]
            )
            if joint["n"] < min_observations:
                joint["verdict"] = "GATING_TOO_THIN"
            elif joint["incremental_vs_components"]:
                joint["verdict"] = "INCREMENTAL_CANDIDATE"
            else:
                joint["verdict"] = "NO_INCREMENTAL_VALUE"
            combinations.append(joint)
    return combinations


def _weekday_windows(
    week_entries: list[dict[str, Any]], min_observations: int
) -> dict[str, Any]:
    scored = [
        entry
        for entry in week_entries
        if entry["n"] >= min_observations and entry["mean_daily"] is not None
    ]
    if not scored:
        return {"status": "INSUFFICIENT_DATA"}
    best = max(scored, key=lambda entry: entry["shrunk_mean_daily"])
    worst = min(scored, key=lambda entry: entry["shrunk_mean_daily"])
    best_day = best["name"].split("=")[1]
    worst_day = worst["name"].split("=")[1]
    previous = {
        "MON": "FRI", "TUE": "MON", "WED": "TUE", "THU": "WED", "FRI": "THU"
    }
    return {
        "best_target_session": best_day,
        "buy": (
            f"MOC on the prior session ({previous[best_day]}); the close-to-"
            f"close return ending {best_day} is the measured interval"
        ),
        "sell": f"MOC on {best_day} (one-session hold matches the evidence)",
        "best_evidence": best,
        "worst_target_session": worst_day,
        "worst_time_to_buy": f"MOC on {previous[worst_day]} (earns {worst_day})",
        "worst_evidence": worst,
    }


def _month_windows(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    windows = {}
    if "turn_of_month" in entries:
        windows["turn_of_month"] = {
            "buy": "MOC on the session before the last trading day of the month",
            "sell": "MOC on the third session of the new month",
            "evidence": entries["turn_of_month"],
        }
    if "month_end_window" in entries:
        windows["month_end_window"] = {
            "buy": "MOC five sessions before month end",
            "sell": "MOC on the final session of the month",
            "evidence": entries["month_end_window"],
        }
    if "quarter_end_window" in entries:
        windows["quarter_end_window"] = {
            "buy": "MOC five sessions before quarter end",
            "sell": "MOC on the final session of the quarter",
            "evidence": entries["quarter_end_window"],
        }
    return windows


def _year_windows(
    year_entries: list[dict[str, Any]], min_observations: int
) -> dict[str, Any]:
    scored = [
        entry
        for entry in year_entries
        if entry["n"] >= min_observations and entry["mean_daily"] is not None
    ]
    if not scored:
        return {"status": "INSUFFICIENT_DATA"}
    ordered = sorted(
        scored, key=lambda entry: entry["shrunk_mean_daily"], reverse=True
    )
    best = ordered[0]
    worst = ordered[-1]
    best_month = best["name"].split("=")[1]
    worst_month = worst["name"].split("=")[1]
    return {
        "best_month": best_month,
        "buy": f"MOC on the final session of the month before {best_month}",
        "sell": f"MOC on the final session of {best_month}",
        "best_evidence": best,
        "worst_month": worst_month,
        "worst_time_to_buy": f"the sessions entering {worst_month}",
        "worst_evidence": worst,
        "ranking": [
            {
                "month": entry["name"].split("=")[1],
                "shrunk_mean_daily": entry["shrunk_mean_daily"],
                "n": entry["n"],
            }
            for entry in ordered
        ],
    }


def _alignment_scan(
    entries: dict[str, dict[str, Any]],
    active_states: list[str],
    *,
    as_of: pd.Timestamp,
    scan_sessions: int,
) -> dict[str, Any]:
    """Score upcoming sessions by the sum of active shrunk condition means.

    Calendar conditions are known in advance; instrument-state conditions are
    frozen at their current values and clearly labeled as such.
    """

    calendar_obj = NYSECalendar()
    # Month-position features (turn-of-month, session-of-month) are counted
    # within each month, so the feature index must start at the CURRENT
    # month's first session or the first partial month would mislabel its
    # leading sessions as a month start.
    month_start = as_of.date().replace(day=1)
    # Extend well past the scan horizon: month/quarter-END distances are only
    # correct for months whose final session is inside the feature index.
    horizon_days = max(200, int(scan_sessions * 1.7) + 60)
    extended_index = calendar_obj.sessions(
        month_start, as_of.date() + timedelta(days=horizon_days)
    )
    future_index = extended_index[extended_index > as_of]
    if len(future_index) == 0:
        return {"status": "NO_FUTURE_SESSIONS"}
    future_calendar = calendar_features(
        extended_index, holidays=_holidays(extended_index)
    ).loc[future_index]
    future_masks = _condition_masks(future_calendar)
    state_contribution = sum(
        entries[name]["reliability_weighted_daily"] for name in active_states
    )
    rows = []
    for session in future_index[:scan_sessions]:
        active = [
            name
            for name, (_, mask) in future_masks.items()
            if name in entries and bool(mask.loc[session])
        ]
        score = state_contribution + sum(
            entries[name]["reliability_weighted_daily"] for name in active
        )
        # Win score: reliability-weighted blend of the active conditions'
        # historical hit rates, shrunk toward the uninformative 50 by a unit
        # prior so unreliable conditions cannot manufacture confidence.
        weight_sum, hit_sum = 1.0, 0.5
        for name in list(active) + list(active_states):
            entry = entries[name]
            if entry.get("hit_rate") is None:
                continue
            weight = float(entry["dsr_probability"])
            weight_sum += weight
            hit_sum += weight * float(entry["hit_rate"])
        rows.append(
            {
                "session": session.date().isoformat(),
                "weekday": _WEEKDAY_NAMES[session.dayofweek],
                "active_calendar_conditions": active,
                "active_state_conditions": active_states,
                "alignment_score_daily": float(score),
                "expected_daily_bp": float(score * 10_000.0),
                "win_score_0_100": int(round(100.0 * hit_sum / weight_sum)),
            }
        )
    ordered = sorted(
        rows, key=lambda row: row["alignment_score_daily"], reverse=True
    )
    return {
        "policy": (
            "score = sum of DSR-reliability-weighted, family-shrunk condition "
            "means active that session (the promoted stack's reliability x "
            "magnitude convention); win score = reliability-weighted blend of "
            "active-condition hit rates shrunk toward 50; state conditions "
            "are frozen at today's values"
        ),
        "all_stars_aligned": ordered[:5],
        "worst_sessions": ordered[-5:][::-1],
        "sessions_scanned": len(rows),
        "calendar": rows,
    }


_EDGE_CONFIGS = (
    (
        "configs/spy-tom-edge-v1.yaml",
        ("edge_preholdout", "edge_holdout"),
        "SPY turn-of-month calendar edge (McConnell & Xu 2008)",
        "python -m edgestack.edges.turn_of_month next-trade",
    ),
    (
        "configs/reversal-edge-v1.yaml",
        ("targeted_preholdout", "targeted_holdout"),
        "five-name S&P 500 five-session reversal basket",
        "python -m edgestack.edges.reversal_edge holdout (sealed; signal only)",
    ),
)


def validated_edge_tier(symbol: str, *, root: Path | None) -> dict[str, Any]:
    """Surface gauntlet-validated edges applicable to one symbol.

    This is the only tier that may call anything an edge: it reads the
    campaign catalog's persisted gate ledger and reports an edge as VALIDATED
    only when its pre-holdout AND single-use holdout gates are PASS. The
    diagnostic conditions elsewhere in the report never outrank this tier.
    """

    edges: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "policy": (
            "VALIDATED entries passed the frozen gauntlet including the "
            "single-use holdout; every diagnostic below is subordinate"
        ),
        "edges": edges,
    }
    if root is None:
        result["status"] = "NOT_CHECKED_NO_ARTIFACT_ROOT"
        return result
    catalog_path = Path(root) / "artifacts" / "edgestack.sqlite"
    if not catalog_path.is_file():
        result["status"] = "NO_CAMPAIGN_CATALOG_AT_ROOT"
        return result
    from edgestack.storage.catalog import Catalog

    catalog = Catalog(catalog_path)
    for config_name, gate_phases, description, command in _EDGE_CONFIGS:
        config_path = Path(root) / config_name
        if not config_path.is_file():
            continue
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        campaign_id = str(payload.get("campaign_id", ""))
        strategy = payload.get("strategy", {})
        if not isinstance(strategy, dict):
            strategy = {}
        data_section = payload.get("data", {})
        if not isinstance(data_section, dict):
            data_section = {}
        edge_symbol = str(data_section.get("symbol", "")).upper()
        universe = str(strategy.get("universe", ""))
        applies: bool | None
        if edge_symbol:
            applies = symbol.upper() == edge_symbol
        elif universe:
            # Universe-scoped edges need a live membership check the advisor
            # cannot perform offline; never claim applicability it can't see.
            applies = None
        else:
            applies = False
        try:
            catalog.require_passed(campaign_id, list(gate_phases))
            status = "VALIDATED_GATES_PASSED"
        except RuntimeError:
            status = "NOT_VALIDATED_GATES_ABSENT_OR_FAILED"
        edges.append(
            {
                "campaign_id": campaign_id,
                "description": description,
                "applies_to_symbol": (
                    applies
                    if applies is not None
                    else "UNKNOWN_REQUIRES_CURRENT_MEMBERSHIP_CHECK"
                ),
                "status": status,
                "bias_tier": str(strategy.get("bias_tier", ""))
                if isinstance(strategy, dict)
                else "",
                "next_step": command,
            }
        )
    result["status"] = "CHECKED"
    return result


def advise(
    bars: pd.DataFrame,
    *,
    symbol: str,
    as_of: date | None = None,
    buy_session: date | None = None,
    buy_hour: str | None = None,
    min_observations: int = 60,
    family_alpha: float = 0.05,
    scan_sessions: int = 63,
    provenance_warnings: tuple[str, ...] = (),
    root: Path | None = None,
) -> dict[str, Any]:
    """Build the complete diagnostic timing report for one instrument."""

    frame = _symbol_frame(bars, symbol)
    if as_of is not None:
        frame = frame.loc[: pd.Timestamp(as_of)]
    prices = frame["adjusted_close"]
    if len(prices) < 252:
        raise ValueError("advisor needs at least one year of daily history")
    index = pd.DatetimeIndex(prices.index)
    returns = prices.pct_change(fill_method=None)
    calendar = calendar_features(index, holidays=_holidays(index))
    ma200 = prices.rolling(200, min_periods=200).mean()
    trend_up = (prices > ma200).shift(1, fill_value=False).astype(bool)

    conditions: dict[str, tuple[str, pd.Series]] = dict(
        _condition_masks(calendar)
    )
    states = _state_masks(prices)
    active_states = [
        name for name, (_, _, currently_active) in states.items() if currently_active
    ]
    for name, (kind, mask, _) in states.items():
        conditions[name] = (kind, mask)

    entries: dict[str, dict[str, Any]] = {}
    for name, (kind, mask) in conditions.items():
        entries[name] = _condition_stats(
            returns, mask.astype(bool), trend_up, name=name, kind=kind
        )
    combination_candidates = [
        (name, conditions[name][1].astype(bool), entry)
        for name, entry in sorted(entries.items())
        if entry["hac_t"] is not None
        and abs(entry["hac_t"]) >= _COMBINATION_COMPONENT_T
        and entry["n"] >= min_observations
    ]
    combinations = _combination_entries(
        returns,
        trend_up,
        combination_candidates,
        min_observations=min_observations,
    )
    family_size = len(entries) + len(combinations)
    for entry in entries.values():
        _classify(entry, family_size, family_alpha)
    for entry in combinations:
        _classify(entry, family_size, family_alpha)

    tailwinds = sorted(
        (
            entry
            for entry in entries.values()
            if entry["classification"].startswith("TAILWIND")
        ),
        key=lambda entry: entry["shrunk_mean_daily"],
        reverse=True,
    )
    headwinds = sorted(
        (
            entry
            for entry in entries.values()
            if entry["classification"].startswith("HEADWIND")
        ),
        key=lambda entry: entry["shrunk_mean_daily"],
    )

    last_session = pd.Timestamp(index[-1])
    report: dict[str, Any] = {
        "status": "DIAGNOSTIC_NOT_A_VALIDATED_EDGE_NOT_AN_ORDER",
        "validated_edges": validated_edge_tier(symbol, root=root),
        "symbol": symbol,
        "as_of_session": last_session.date().isoformat(),
        "history_sessions": int(len(prices)),
        "family_size_tested": family_size,
        "multiplicity_policy": (
            "every condition, state, and combination below counts toward one "
            "Bonferroni family; RAW_ONLY classifications did not survive it"
        ),
        "tailwinds": tailwinds,
        "headwinds": headwinds,
        "all_conditions": {name: entries[name] for name in sorted(entries)},
        "combinations": combinations,
        "timing": {
            "week": _weekday_windows(
                [entries[f"weekday={name}"] for name in _WEEKDAY_NAMES],
                min_observations,
            ),
            "month": _month_windows(entries),
            "year": _year_windows(
                [entries[f"month={name}"] for name in _MONTH_NAMES],
                min_observations,
            ),
            "anchors": _anchor_assessment(frame),
            "execution": _EXECUTION_GUIDANCE,
        },
        "alignment": _alignment_scan(
            entries, active_states, as_of=last_session, scan_sessions=scan_sessions
        ),
        "current_year_context": _current_year_context(prices, returns, trend_up),
        "news": {
            "status": "DATA_UNAVAILABLE",
            "reason": (
                "no licensed, timestamped news source exists in the free data "
                "layer; scraping headlines would inject unvalidated, "
                "unreproducible inputs"
            ),
        },
        "provenance_warnings": list(provenance_warnings)
        + [
            "SURVIVORSHIP_AND_SINGLE_SYMBOL: single-instrument conditional "
            "means are descriptive; they are NOT the campaign gauntlet.",
        ],
        "disclaimer": DISCLAIMER,
    }
    if buy_session is not None:
        report["buy_time_assessment"] = _buy_time_assessment(
            entries,
            conditions,
            active_states,
            buy_session=pd.Timestamp(buy_session),
            last_session=last_session,
        )
        report["buy_time_assessment"]["choice_review"] = _choice_review(
            entries,
            report["alignment"],
            report["timing"],
            buy_session=pd.Timestamp(buy_session),
            buy_hour=buy_hour,
        )
    return report


def _current_year_context(
    prices: pd.Series, returns: pd.Series, trend_up: pd.Series
) -> dict[str, Any]:
    """Descriptive current-year facts computed only from the loaded history."""

    year = int(pd.Timestamp(prices.index[-1]).year)
    this_year = returns[returns.index.year == year].dropna()
    history_vol = returns.rolling(21, min_periods=21).std().dropna()
    current_vol = float(history_vol.iloc[-1]) if len(history_vol) else None
    return {
        "policy": "DESCRIPTIVE_ONLY_NOT_A_FORECAST",
        "year": year,
        "year_to_date_return": (
            float((1.0 + this_year).prod() - 1.0) if len(this_year) else None
        ),
        "trend_state": "ABOVE_MA200" if bool(trend_up.iloc[-1]) else "BELOW_MA200",
        "volatility_percentile_vs_history": (
            float((history_vol <= current_vol).mean())
            if current_vol is not None
            else None
        ),
        "tips": [
            "Condition means are long-run averages; the current year can and "
            "does deviate from them for months at a time.",
            "Treat any single-year pattern as noise unless it also appears in "
            "the full-history conditional evidence above.",
        ],
    }


def _choice_review(
    entries: dict[str, dict[str, Any]],
    alignment: dict[str, Any],
    timing: dict[str, Any],
    *,
    buy_session: pd.Timestamp,
    buy_hour: str | None,
) -> dict[str, Any]:
    """Rank the user's chosen day/hour and propose better executable choices.

    Ranks come from the same reliability-weighted evidence as the alignment
    scan; alternatives are only ever other SESSIONS and the two AUCTION
    anchors — never intraday hours the data cannot see.
    """

    weekday_rank: dict[str, Any] = {}
    chosen_weekday = _WEEKDAY_NAMES[buy_session.dayofweek] if (
        buy_session.dayofweek < 5
    ) else None
    ordered_weekdays = sorted(
        (entries[f"weekday={name}"] for name in _WEEKDAY_NAMES),
        key=lambda entry: entry["reliability_weighted_daily"],
        reverse=True,
    )
    if chosen_weekday is not None:
        position = next(
            index
            for index, entry in enumerate(ordered_weekdays, start=1)
            if entry["name"] == f"weekday={chosen_weekday}"
        )
        weekday_rank = {
            "chosen_weekday": chosen_weekday,
            "rank_of_5": position,
            "better_weekdays": [
                entry["name"].split("=")[1]
                for entry in ordered_weekdays[: position - 1]
            ],
        }
    scanned = {
        row["session"]: row
        for row in alignment.get("all_stars_aligned", [])
        + alignment.get("worst_sessions", [])
    }
    chosen_label = buy_session.date().isoformat()
    better_sessions = [
        row
        for row in alignment.get("all_stars_aligned", [])
        if row["session"] != chosen_label
        and (
            chosen_label not in scanned
            or row["alignment_score_daily"]
            > scanned[chosen_label]["alignment_score_daily"]
        )
    ][:3]
    anchors = timing.get("anchors", {})
    hour_review: dict[str, Any] = {"requested_hour_et": buy_hour}
    if buy_hour is not None:
        anchor, verdict = _classify_hour(buy_hour)
        hour_review["verdict"] = verdict
        hour_review["requested_anchor"] = anchor
        if anchors.get("status") == "TWO_ANCHORS_ONLY":
            hour_review["recommended_buy_anchor"] = anchors["best_buy_anchor"]
            hour_review["recommended_sell_anchor"] = anchors["matching_sell_anchor"]
        hour_review["finer_than_anchors"] = (
            "DATA_UNAVAILABLE: only the opening and closing auctions are "
            "measured; no hourly or 15-minute rating exists"
        )
    sell_plan = _sell_plan(entries, timing, buy_session=buy_session)
    return {
        "weekday_rank": weekday_rank,
        "better_upcoming_sessions": better_sessions,
        "hour_review": hour_review,
        "sell_plan_by_horizon": sell_plan,
        "revalidation_schedule": [
            "re-run this assessment after every completed close between now "
            "and the entry session; a rating flip or a fresher, higher-"
            "scoring session supersedes this one",
            "final check at 15:45 ET on the entry session (decision freeze): "
            "confirm the rating still holds, the quote is fresh, and no gap "
            "or event invalidates the setup — then do not reselect",
            "after entry, re-rate only at the planned exit checkpoints; "
            "mid-hold re-optimization is how tested plans decay into "
            "improvisation",
        ],
    }


def _sell_plan(
    entries: dict[str, dict[str, Any]],
    timing: dict[str, Any],
    *,
    buy_session: pd.Timestamp,
) -> dict[str, Any]:
    """Horizon-matched exit guidance derived from the same evidence tables."""

    anchors = timing.get("anchors", {})
    day_exit = (
        anchors.get("matching_sell_anchor", "same-day CLOSE_AUCTION")
        if anchors.get("status") == "TWO_ANCHORS_ONLY"
        else "same-day CLOSE_AUCTION"
    )
    week = timing.get("week", {})
    week_exit = (
        f"MOC on the next {week['best_target_session']} after entry "
        "(the week's strongest measured session closes the hold)"
        if "best_target_session" in week
        else "MOC five sessions after entry"
    )
    month_windows = timing.get("month", {})
    if "turn_of_month" in month_windows:
        month_exit = (
            "MOC on the third session of the following month (turn-of-month "
            "window close)"
        )
    else:
        month_exit = "MOC twenty-one sessions after entry"
    year = timing.get("year", {})
    if "best_month" in year:
        year_exit = (
            f"MOC on the final session of the next {year['best_month']} "
            f"(strongest month); avoid initiating exits into {year['worst_month']}"
        )
    else:
        year_exit = "MOC roughly 252 sessions after entry"
    return {
        "day": {
            "exit": day_exit,
            "basis": "overnight-vs-intraday split of the daily record",
        },
        "week": {"exit": week_exit, "basis": "weekday conditional evidence"},
        "month": {
            "exit": month_exit,
            "basis": "turn-of-month/month-end window evidence",
        },
        "year": {"exit": year_exit, "basis": "month-of-year conditional evidence"},
        "always": (
            "the 2 x ATR(14) reference stop and the cancel-if rules outrank "
            "every calendar exit above"
        ),
    }


def _buy_time_assessment(
    entries: dict[str, dict[str, Any]],
    conditions: dict[str, tuple[str, pd.Series]],
    active_states: list[str],
    *,
    buy_session: pd.Timestamp,
    last_session: pd.Timestamp,
) -> dict[str, Any]:
    """Rate one intended buy session from the conditions active on it."""

    calendar_obj = NYSECalendar()
    if not calendar_obj.is_session(buy_session):
        return {
            "status": "NOT_A_TRADING_SESSION",
            "requested": buy_session.date().isoformat(),
        }
    probe_index = calendar_obj.sessions(
        buy_session.date() - timedelta(days=120),
        buy_session.date() + timedelta(days=120),
    )
    probe_calendar = calendar_features(
        probe_index, holidays=_holidays(probe_index)
    )
    probe_masks = _condition_masks(probe_calendar)
    active_calendar = [
        name
        for name, (_, mask) in probe_masks.items()
        if name in entries and bool(mask.loc[buy_session])
    ]
    state_note = (
        "state conditions reflect the latest completed session "
        f"({last_session.date().isoformat()}), not the buy session"
    )
    active = active_calendar + active_states
    positives = [
        entries[name]
        for name in active
        if entries[name]["reliability_weighted_daily"] > 0
    ]
    negatives = [
        entries[name]
        for name in active
        if entries[name]["reliability_weighted_daily"] < 0
    ]
    score = float(
        sum(entries[name]["reliability_weighted_daily"] for name in active)
    )
    if score > 0 and positives:
        rating = "POSITIVE"
    elif score < 0 and negatives:
        rating = "NEGATIVE"
    else:
        rating = "NEUTRAL_OR_MIXED"
    return {
        "buy_session": buy_session.date().isoformat(),
        "state_conditions_note": state_note,
        "active_conditions": active,
        "tailwinds_active": positives,
        "headwinds_active": negatives,
        "net_reliability_weighted_daily_return": score,
        "overall_rating": rating,
        "dark_side_reminder": (
            "check regime_down_mean on every active tailwind and "
            "regime_up_mean on every active headwind before acting"
        ),
        "if_you_trade_anyway": [
            "halve the intended size",
            "use a limit-on-close at 0.25 x ATR from the decision price, "
            "never a chased market order",
            "predefine a 2 x ATR(14) reference stop and the loss it implies",
            "cancel if the quote is stale, a bar is missing, or the price "
            "gapped more than your preregistered fraction",
            "log the decision before the fill so hindsight cannot edit it",
        ],
    }


__all__ = ["advise", "validated_edge_tier"]
