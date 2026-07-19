"""Post-close automation: fresh basket scan, forward ledger, calendars.

One nightly run, after the NYSE close, keeps the whole paper loop honest and
hands-free: regenerate the promoted five-name reversal signal from the just-
completed close (the no-chase protocol forbids reusing an old scan), append
fills/marks/exits for prior signals to the tamper-evident forward ledger at
official closes, and refresh the advisor tailwind calendars the mobile app
serves.

The scan refuses to run unless the campaign's pre-holdout and holdout gates
are PASS in the catalog — automation never bypasses the gauntlet.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from edgestack.data.calendars import NYSECalendar
from edgestack.data.sources import (
    FallbackDailyBarSource,
    StooqDailyBarSource,
    YahooDailyBarSource,
    bars_to_frame,
)
from edgestack.disclaimer import DISCLAIMER
from edgestack.live.forward_ledger import ForwardLedger
from edgestack.models import AssetKey, BarRequest
from edgestack.provenance import canonical_sha256
from edgestack.storage.catalog import Catalog

DEFAULT_CAMPAIGN = "reversal-edge-v1-20260715-001"
_LOOKBACK_SESSIONS = 5
_TOP_K = 5
_HOLDING_SESSIONS = 5
_RISK_BUDGET_USD = 500.0
_PAPER_CAPITAL_USD = 100_000.0


def last_completed_session(now: datetime | None = None) -> date:
    """Latest NYSE session whose closing auction has already printed."""

    calendar = NYSECalendar()
    current = now or datetime.now(UTC)
    candidate = current.date()
    if calendar.is_session(candidate) and current >= calendar.close_time(candidate):
        return candidate
    return calendar.previous_session(candidate, inclusive=False).date()


async def _fetch_panel(
    symbols: tuple[str, ...], start: date, end: date, *, concurrency: int = 8
) -> dict[str, pd.DataFrame]:
    chain = FallbackDailyBarSource((StooqDailyBarSource(), YahooDailyBarSource()))
    semaphore = asyncio.Semaphore(concurrency)

    async def fetch(symbol: str) -> tuple[str, pd.DataFrame | None]:
        async with semaphore:
            try:
                batch = await chain.fetch_bars(
                    BarRequest(AssetKey(symbol), start, end, adjusted=True)
                )
                return symbol, bars_to_frame(batch)
            except Exception:
                return symbol, None

    results = await asyncio.gather(*(fetch(symbol) for symbol in symbols))
    return {symbol: frame for symbol, frame in results if frame is not None}


def _wilder_atr14(frame: pd.DataFrame) -> float:
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    return float(atr.iloc[-1])


def build_signal_payload(
    panel: dict[str, pd.DataFrame],
    equity_symbols: tuple[str, ...],
    *,
    as_of: date,
    source_label: str,
) -> dict[str, Any]:
    """Apply the frozen five-name contract to completed closes only."""

    calendar = NYSECalendar()
    scores: dict[str, tuple[float, float, float]] = {}
    for symbol in equity_symbols:
        frame = panel.get(symbol)
        if frame is None or len(frame) < _LOOKBACK_SESSIONS + 15:
            continue
        frame = frame[pd.to_datetime(frame["session"]).dt.date <= as_of]
        if len(frame) < _LOOKBACK_SESSIONS + 15:
            continue
        adjusted = frame["adjusted_close"].astype(float)
        trailing = float(
            adjusted.iloc[-1] / adjusted.iloc[-1 - _LOOKBACK_SESSIONS] - 1.0
        )
        if not math.isfinite(trailing):
            continue
        scores[symbol] = (trailing, float(adjusted.iloc[-1]), _wilder_atr14(frame))
    if len(scores) < _TOP_K:
        raise RuntimeError("not enough usable equities for the basket scan")
    ordered = sorted(scores.items(), key=lambda item: item[1][0])
    universe_moves = np.array([value[0] for value in scores.values()])
    future = calendar.sessions(as_of, as_of + timedelta(days=30))
    upcoming = [session.date() for session in future if session.date() > as_of]
    entry_session = upcoming[0]
    exit_session = upcoming[_HOLDING_SESSIONS]
    candidates = []
    for rank, (symbol, (trailing, close, atr)) in enumerate(
        ordered[:_TOP_K], start=1
    ):
        percentile = float((universe_moves > trailing).mean())
        identity = canonical_sha256(
            {"symbol": symbol, "session": as_of.isoformat(), "rank": rank}
        )
        candidates.append(
            {
                "recommendation_id": f"rec-{identity[:16]}",
                "rank": rank,
                "symbol": symbol,
                "direction": "LONG",
                "confidence_ordinal_not_probability": int(round(100 * percentile)),
                "signal_close_usd": round(close, 2),
                "trailing_5_session_return": round(trailing, 6),
                "atr14_wilder_usd": round(atr, 2),
                "tested_10pct_reference_shares": int(
                    (_PAPER_CAPITAL_USD * 0.10) // close
                ),
                "risk_capped_reference_shares": int(
                    _RISK_BUDGET_USD // max(2.0 * atr, 1e-9)
                ),
                "two_atr_reference_price_usd": round(close - 2.0 * atr, 2),
                "event_risk": (
                    f"MOVE_{abs(trailing) * 100:.0f}PCT_IN_5_SESSIONS_"
                    "REVIEW_NEWS_BEFORE_ENTRY"
                ),
            }
        )
    return {
        "artifact_type": "PAPER_ONLY_CONDITIONAL_ENTRY_PLAN",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "market_as_of": f"{as_of.isoformat()}_CLOSE",
        "bias_tier": "SURVIVORSHIP_BIASED",
        "status": "ACTIONABLE_PAPER_BASKET_PENDING_PRE_CLOSE_REVALIDATION",
        "strategy": "CURRENT_SP500_RAW_5D_REVERSAL_TOP5_HALF_GROSS",
        "data": {
            "source": source_label,
            "snapshot_policy": (
                "one whole adjusted-OHLCV series per instrument through the "
                "completed close; no provider splicing"
            ),
            "requested_equities": len(equity_symbols),
            "usable_equities": len(scores),
            "failed_equities": len(equity_symbols) - len(scores),
            "warnings": [
                "UNOFFICIAL_FREE_PROVIDER",
                "SURVIVORSHIP_BIASED_CURRENT_CONSTITUENTS",
                "AUTOMATED_POST_CLOSE_SCAN",
            ],
        },
        "entry": {
            "session": entry_session.isoformat(),
            "order_type": "MOC",
            "planned_submission_time": "15:45:00 America/New_York",
            "expected_execution_time": "16:00:00 America/New_York",
            "cancel_if": [
                "quote is stale or unavailable",
                "the symbol is no longer in the frozen current-constituent universe",
                "the broker's closing-auction cutoff is missed",
                "the full basket cannot be implemented",
            ],
            "no_chase": (
                "If the cutoff is missed, do not enter continuously or at the "
                "next open; wait for a new completed-close scan."
            ),
        },
        "exit": {
            "session": exit_session.isoformat(),
            "order_type": "MOC",
            "planned_submission_time": "15:45:00 America/New_York",
            "reason": "five earned close-to-close return intervals after the entry auction",
        },
        "portfolio": {
            "paper_capital_usd": _PAPER_CAPITAL_USD,
            "tested_new_account_gross_target": 0.5,
            "tested_maximum_weight_per_name": 0.1,
            "paper_risk_budget_per_name_usd": _RISK_BUDGET_USD,
            "risk_capped_reference_gross_usd": round(
                sum(
                    item["risk_capped_reference_shares"] * item["signal_close_usd"]
                    for item in candidates
                ),
                2,
            ),
            "risk_reference": (
                "2 x Wilder ATR14; a risk control, not a validated alpha overlay"
            ),
        },
        "candidates": candidates,
        "shorts": [],
        "short_status": (
            "DISABLED_ALL_DECLARED_SHORT_REVERSAL_RULES_FAILED_PREHOLDOUT_VALIDATION"
        ),
        "interpretation": (
            "The five names are one tested cross-sectional basket. Individual "
            "ranks are not independently validated stock forecasts, and "
            "confidence is an ordinal score rather than a success probability. "
            "Do not replace a name with rank six or cherry-pick only rank one "
            "and call it the tested strategy."
        ),
        "disclaimer": DISCLAIMER,
    }


def _close_on(panel: dict[str, pd.DataFrame], symbol: str, session: str) -> float | None:
    frame = panel.get(symbol)
    if frame is None:
        return None
    rows = frame[pd.to_datetime(frame["session"]).dt.date == date.fromisoformat(session)]
    if rows.empty:
        return None
    return float(rows["adjusted_close"].iloc[-1])


def update_forward_ledger(
    ledger: ForwardLedger,
    panel: dict[str, pd.DataFrame],
    *,
    as_of: date,
) -> dict[str, int]:
    """Append fills, marks, and exits from completed official closes."""

    calendar = NYSECalendar()
    counts = {"fills": 0, "marks": 0, "exits": 0}
    for open_signal in ledger.open_signals(before_session=as_of.isoformat()):
        entry = open_signal["entry_session"]
        exit_ = open_signal["exit_session"]
        for item in open_signal["payload"]["candidates"]:
            symbol = str(item["symbol"])
            recommendation = str(item["recommendation_id"])
            fill_price = _close_on(panel, symbol, entry)
            if fill_price is not None:
                if ledger.record_event(
                    recommendation, symbol=symbol, event="FILL",
                    session=entry, price=fill_price,
                ):
                    counts["fills"] += 1
            if exit_ <= as_of.isoformat():
                exit_price = _close_on(panel, symbol, exit_)
                if exit_price is not None and ledger.record_event(
                    recommendation, symbol=symbol, event="EXIT",
                    session=exit_, price=exit_price,
                ):
                    counts["exits"] += 1
            elif entry < as_of.isoformat():
                mark_price = _close_on(panel, symbol, as_of.isoformat())
                if (
                    mark_price is not None
                    and calendar.is_session(as_of)
                    and ledger.record_event(
                        recommendation, symbol=symbol, event="MARK",
                        session=as_of.isoformat(), price=mark_price,
                    )
                ):
                    counts["marks"] += 1
    return counts


def run_post_close(
    *,
    root: str | Path = ".",
    campaign_id: str = DEFAULT_CAMPAIGN,
    calendar_symbols: tuple[str, ...] = ("SPY", "QQQ", "GLD", "ACN", "CTSH"),
) -> dict[str, Any]:
    """Run the complete nightly loop; every step is idempotent per session."""

    base = Path(root).resolve()
    catalog = Catalog(base / "artifacts" / "edgestack.sqlite")
    catalog.require_passed(campaign_id, ["targeted_preholdout", "targeted_holdout"])
    as_of = last_completed_session()

    from edgestack.data.universe import WikipediaSP500UniverseSource

    memberships = asyncio.run(
        WikipediaSP500UniverseSource(include_etfs=False).memberships(
            as_of - timedelta(days=30), as_of
        )
    )
    equities = tuple(sorted({item.asset.symbol for item in memberships}))
    # Point-in-time membership snapshot: one immutable file per session. From
    # the first snapshot onward the forward universe is survivorship-free by
    # construction; the historical panel's bias stays stamped, not fixed.
    snapshot_dir = base / "artifacts" / "universe"
    snapshot_path = snapshot_dir / f"membership-{as_of.isoformat()}.json"
    if not snapshot_path.is_file():
        import json as json_module

        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json_module.dumps(
                {
                    "as_of": as_of.isoformat(),
                    "source": "Wikipedia current S&P 500 membership",
                    "watermark": "POINT_IN_TIME_FROM_CAPTURE_DATE_ONLY",
                    "symbols": list(equities),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    campaign_root = base / "artifacts" / "campaigns" / campaign_id
    ledger = ForwardLedger(campaign_root / "forward" / "ledger.sqlite")
    ledger_symbols = {
        str(item["symbol"])
        for open_signal in ledger.open_signals(before_session=as_of.isoformat())
        for item in open_signal["payload"]["candidates"]
    }
    fetch_symbols = tuple(sorted(set(equities) | ledger_symbols))
    panel = asyncio.run(
        _fetch_panel(fetch_symbols, as_of - timedelta(days=120), as_of)
    )
    # Calendar symbols need deep history for the advisor's conditional
    # statistics; a few long series are cheap compared to the universe sweep.
    calendar_panel = asyncio.run(
        _fetch_panel(
            tuple(calendar_symbols), as_of - timedelta(days=365 * 20), as_of
        )
    )

    signal_path = campaign_root / "live" / f"{as_of.isoformat()}-signal.json"
    wrote_signal = False
    if not signal_path.is_file():
        payload = build_signal_payload(
            panel,
            tuple(symbol for symbol in equities if symbol in panel),
            as_of=as_of,
            source_label="Stooq/Yahoo free chain via edgestack post-close job",
        )
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        import json as json_module

        signal_path.write_text(
            json_module.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ledger.record_signal(payload)
        wrote_signal = True
    else:
        import json as json_module

        ledger.record_signal(
            json_module.loads(signal_path.read_text(encoding="utf-8"))
        )

    ledger_counts = update_forward_ledger(ledger, panel, as_of=as_of)
    scorecard = ledger.scorecard()
    import json as json_module

    (campaign_root / "forward").mkdir(parents=True, exist_ok=True)
    (campaign_root / "forward" / "scorecard.json").write_text(
        json_module.dumps(scorecard, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    from edgestack.advisor import advise

    advisor_dir = base / "artifacts" / "advisor"
    advisor_dir.mkdir(parents=True, exist_ok=True)
    calendars: dict[str, str] = {}
    for symbol in calendar_symbols:
        frame = calendar_panel.get(symbol)
        if frame is None:
            calendars[symbol] = "FETCH_FAILED"
            continue
        try:
            report = advise(frame, symbol=symbol, scan_sessions=42, root=base)
            calendar_payload = {
                "status": report["status"],
                "symbol": report["symbol"],
                "as_of_session": report["as_of_session"],
                "policy": report["alignment"]["policy"],
                "anchors": report["timing"]["anchors"],
                "calendar": report["alignment"]["calendar"],
                "validated_edges": report["validated_edges"],
                "provenance_warnings": report["provenance_warnings"],
                "disclaimer": report["disclaimer"],
            }
            text = json_module.dumps(
                calendar_payload, indent=2, sort_keys=True, default=str
            ) + "\n"
            (advisor_dir / f"tailwind-calendar-{symbol}.json").write_text(
                text, encoding="utf-8"
            )
            if symbol == calendar_symbols[0]:
                (advisor_dir / "tailwind-calendar.json").write_text(
                    text, encoding="utf-8"
                )
            calendars[symbol] = "OK"
        except Exception as error:  # a calendar failure must not kill the scan
            calendars[symbol] = f"FAILED: {error}"
    summary = {
        "as_of": as_of.isoformat(),
        "signal_written": wrote_signal,
        "signal_path": str(signal_path),
        "ledger": ledger_counts,
        "forward_scorecard": scorecard,
        "calendars": calendars,
        "universe_size": len(equities),
        "fetched": len(panel),
    }
    summary["telegram"] = _notify_telegram(summary, signal_path)
    return summary


def _notify_telegram(summary: dict[str, Any], signal_path: Path) -> str:
    """Push the nightly result to Telegram when credentials are configured.

    Reads EDGESTACK_TELEGRAM_TOKEN and EDGESTACK_TELEGRAM_CHAT from the
    environment; absent credentials mean a clean, explicit skip. Delivery is
    at-least-once and idempotent per session via the event id; a send failure
    never fails the scan itself.
    """

    import os

    token = os.environ.get("EDGESTACK_TELEGRAM_TOKEN", "").strip()
    chat = os.environ.get("EDGESTACK_TELEGRAM_CHAT", "").strip()
    if not token or not chat:
        return "SKIPPED_NO_CREDENTIALS"
    import json as json_module

    from datetime import UTC, datetime

    from edgestack.live.notify import TelegramChannel
    from edgestack.models import AlertEvent

    try:
        payload = json_module.loads(signal_path.read_text(encoding="utf-8"))
        candidates = payload.get("candidates", [])
        names = " ".join(
            f"{item['symbol']}" for item in candidates
        ) or "none"
        entry = payload.get("entry", {})
        message = (
            f"EdgeStack post-close {summary['as_of']}\n"
            f"Basket ({len(candidates)}): {names}\n"
            f"Entry MOC {entry.get('session', '?')} · submit by 15:45 ET\n"
            f"Exit MOC {payload.get('exit', {}).get('session', '?')}\n"
            f"Signal {'FRESH' if summary['signal_written'] else 'already existed'}"
            f" · forward fills {summary['ledger']}\n"
            "PAPER ONLY · not financial advice · revalidate before the close"
        )
        event = AlertEvent(
            event_id=f"post-close-{summary['as_of']}",
            recommendation_id=str(payload.get("freeze_id", "unknown"))[:16],
            revision=1,
            event_type="POST_CLOSE_SCAN",
            message=message,
            created_at=datetime.now(UTC),
        )
        channel = TelegramChannel(token=token, chat_id=chat)
        asyncio.run(channel.send(event, event.event_id))
        return "SENT"
    except Exception as error:  # notification failure must not fail the scan
        return f"FAILED: {error}"


__all__ = [
    "ForwardLedger",
    "build_signal_payload",
    "last_completed_session",
    "run_post_close",
    "update_forward_ledger",
]
