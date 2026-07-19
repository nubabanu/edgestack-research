"""Coverage audit for point-in-time universe reconstruction.

Answers one question before any campaign relies on the Wikipedia change-log
reconstruction: for the names REMOVED from the S&P 500, how much of their
price history does the hash-pinned Stooq bulk archive actually hold? The
per-year coverage fraction decides how far back a ``PIT_APPROXIMATION``
universe is honest; symbols without coverage stay visible in the report
instead of silently shrinking the universe.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from edgestack.data.sources import _vendor_symbol
from edgestack.data.universe import MembershipChange


def stooq_member_key(symbol: str) -> str:
    """Return the bulk-archive member leaf name for one canonical symbol."""

    return f"{_vendor_symbol(symbol, 'stooq')}.txt"


def summarize_pit_coverage(
    changes: Sequence[MembershipChange],
    member_keys: Collection[str],
    *,
    start: date,
    end: date,
) -> dict[str, Any]:
    """Cross Wikipedia removals against the Stooq bulk member index.

    Report-only: symbols are matched by member name, not by loading price
    data, so a hit means "a delisted-capable series exists", not that its
    history is complete. Ticker renames and mergers may alias; the unmatched
    list is published so an operator-maintained alias map can close gaps.
    """

    if end < start:
        raise ValueError("end must be on or after start")
    lowered = {str(key).lower() for key in member_keys}
    removals: list[tuple[date, str]] = sorted(
        {
            (change.effective_date, change.removed_symbol)
            for change in changes
            if change.removed_symbol is not None
            and start <= change.effective_date <= end
        }
    )
    per_year: dict[int, dict[str, int]] = {}
    uncovered: list[dict[str, str]] = []
    for effective, symbol in removals:
        bucket = per_year.setdefault(effective.year, {"removed": 0, "covered": 0})
        bucket["removed"] += 1
        if stooq_member_key(symbol).lower() in lowered:
            bucket["covered"] += 1
        else:
            uncovered.append(
                {"symbol": symbol, "removed_on": effective.isoformat()}
            )
    years = {
        str(year): {
            "removed": counts["removed"],
            "covered": counts["covered"],
            "coverage_fraction": (
                counts["covered"] / counts["removed"] if counts["removed"] else None
            ),
        }
        for year, counts in sorted(per_year.items())
    }
    total_removed = sum(counts["removed"] for counts in per_year.values())
    total_covered = sum(counts["covered"] for counts in per_year.values())
    return {
        "policy": "REPORT_ONLY_COVERAGE_AUDIT",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "removed_symbols": total_removed,
        "covered_symbols": total_covered,
        "coverage_fraction": (
            total_covered / total_removed if total_removed else None
        ),
        "per_year": years,
        "uncovered": uncovered,
        "caveats": [
            "Member-name matching does not verify history completeness.",
            "Ticker renames/mergers need an operator alias map to match.",
            "Wikipedia's change log thins out before ~2000; absence of a "
            "logged removal is not evidence of continuous membership.",
        ],
    }


def universe_bias_delta(
    bars: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    execution_lag: int = 2,
) -> dict[str, Any]:
    """Measure how much universe survivorship inflates two replicated signals.

    Runs gross decile long-short streams for 5-day reversal and 12-1 momentum
    twice on the same bars: once over the full panel (current members applied
    backward — the survivorship-biased convention) and once with each name
    masked outside its recorded membership interval. The mean/Sharpe deltas
    convert the SURVIVORSHIP_BIASED watermark into a measured number.
    Report-only; gross of costs because both variants pay identical costs.
    """

    from edgestack.features.cross_sectional import (
        decile_weights,
        momentum_12_1,
        short_term_reversal,
    )
    from edgestack.stats.tests import annualized_sharpe

    required = {"symbol", "session", "adjusted_close"}
    if not required.issubset(bars.columns):
        raise ValueError("bars require symbol, session, and adjusted_close")
    if not {"symbol", "start"}.issubset(universe.columns):
        raise ValueError("universe requires symbol and start columns")
    equities = universe
    if "asset_type" in universe.columns:
        equities = universe.loc[universe["asset_type"].astype(str) != "etf"]
    intervals = {
        str(row.symbol): (
            pd.Timestamp(row.start),
            (
                pd.Timestamp(row.end)
                if "end" in universe.columns and pd.notna(row.end)
                else None
            ),
        )
        for row in equities.itertuples(index=False)
    }
    frame = bars.loc[bars["symbol"].astype(str).isin(intervals)].copy()
    frame["session"] = pd.to_datetime(frame["session"])
    prices = frame.pivot_table(
        index="session", columns="symbol", values="adjusted_close", aggfunc="first"
    ).sort_index()
    membership = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    for symbol, (start, end) in intervals.items():
        if symbol not in membership.columns:
            continue
        column = (membership.index >= start) & (
            (membership.index < end) if end is not None else True
        )
        membership[symbol] = column
    pit_prices = prices.where(membership)
    signals = {
        "reversal_5d": lambda values: short_term_reversal(values, lookback=5),
        "momentum_12_1": lambda values: momentum_12_1(values),
    }
    report: dict[str, Any] = {
        "policy": "REPORT_ONLY_BIAS_QUANTIFICATION",
        "sessions": int(len(prices)),
        "symbols": int(prices.shape[1]),
        "pit_masked_cells_fraction": float(1.0 - membership.to_numpy().mean()),
        "signals": {},
    }
    for name, compute in signals.items():
        entry: dict[str, Any] = {}
        for label, panel in (("survivorship_biased", prices), ("pit_masked", pit_prices)):
            weights = decile_weights(compute(panel))
            returns = panel.pct_change(fill_method=None)
            stream = (
                weights.shift(execution_lag).mul(returns).sum(axis=1, min_count=1)
            )
            values = stream.to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            entry[label] = {
                "gross_mean_daily": float(values.mean()) if values.size else None,
                "annualized_sharpe": (
                    float(annualized_sharpe(values)) if values.size else None
                ),
                "observations": int(values.size),
            }
        biased = entry["survivorship_biased"]
        masked = entry["pit_masked"]
        entry["bias_delta"] = {
            "gross_mean_daily": (
                biased["gross_mean_daily"] - masked["gross_mean_daily"]
                if biased["gross_mean_daily"] is not None
                and masked["gross_mean_daily"] is not None
                else None
            ),
            "annualized_sharpe": (
                biased["annualized_sharpe"] - masked["annualized_sharpe"]
                if biased["annualized_sharpe"] is not None
                and masked["annualized_sharpe"] is not None
                else None
            ),
        }
        report["signals"][name] = entry
    return report


__all__ = ["stooq_member_key", "summarize_pit_coverage", "universe_bias_delta"]
