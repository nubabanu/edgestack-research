"""Pre-close heads-up: warn 10-15 minutes before the decision freeze.

Runs on a weekday schedule shortly before the 15:45 America/New_York MOC
decision freeze. It reads TODAY's row from the calendars the nightly job
already published (no recomputation), applies the same alignment threshold
and estimated-earnings suppression as the nightly entry signals, attaches a
15-minute-delayed quote and the peer-healing context, and pushes one
Telegram message. Silent, clean skips on non-sessions, empty signals, or
missing credentials — this job may never fail loudly.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

IT_SERVICES_PEERS = ("ACN", "CTSH", "EPAM", "IBM")
DEFAULT_SYMBOLS = ("SPY", "QQQ", "GLD", "ACN", "CTSH", "EPAM")
_THRESHOLD_BP = 15.0


def peers_healing(
    frames: Mapping[str, pd.DataFrame], peers: Sequence[str] = IT_SERVICES_PEERS
) -> tuple[int, int]:
    """(healing, total): peers with positive 5-session return above 20d MA."""

    healing = total = 0
    for peer in peers:
        frame = frames.get(peer)
        if frame is None or len(frame) < 21:
            continue
        adjusted = frame["adjusted_close"].astype(float).to_numpy()
        close = frame["close"].astype(float).to_numpy()
        total += 1
        if adjusted[-1] / adjusted[-6] - 1.0 > 0 and close[-1] > close[-20:].mean():
            healing += 1
    return healing, total


def preclose_lines(
    rows: Sequence[dict[str, Any]],
    *,
    threshold_bp: float = _THRESHOLD_BP,
    earnings_windows: Mapping[str, tuple[str, str]] | None = None,
    quotes: Mapping[str, str] | None = None,
    peers_note: str | None = None,
) -> list[str]:
    """One line per symbol whose TODAY row clears the bar outside earnings."""

    lines: list[str] = []
    for row in rows:
        expected = float(row.get("expected_daily_bp") or 0.0)
        if expected < threshold_bp:
            continue
        symbol = str(row["symbol"])
        session = str(row["session"])[:10]
        window = (earnings_windows or {}).get(symbol)
        if window and window[0] <= session <= window[1]:
            continue
        peer = f" {peers_note}" if peers_note and symbol in IT_SERVICES_PEERS else ""
        quote = (quotes or {}).get(symbol)
        quote_note = f" · quote {quote} (15-min delayed)" if quote else ""
        lines.append(
            f"PRE_CLOSE (diagnostic): {symbol} today {session}"
            f" ~{expected:.0f}bp{peer} — decision freeze 15:45 ET{quote_note}"
        )
    return lines


def _today_calendar_rows(
    advisor_dir: Path, symbols: Sequence[str], today: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        path = advisor_dir / f"tailwind-calendar-{symbol}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("calendar", []):
            if str(row.get("session", ""))[:10] == today:
                rows.append({**row, "symbol": symbol})
                break
    return rows


def _delayed_quotes(symbols: Sequence[str]) -> dict[str, str]:
    from edgestack.data.sources import YahooQuoteSource
    from edgestack.models import AssetKey

    quotes: dict[str, str] = {}
    try:
        fetched = asyncio.run(
            YahooQuoteSource().fetch_quotes(
                tuple(AssetKey(symbol) for symbol in symbols)
            )
        )
        for quote in fetched:
            quotes[quote.asset.symbol] = f"${quote.price:.2f}"
    except Exception:  # a quote failure must not block the heads-up
        pass
    return quotes


def run_pre_close_check(
    *,
    root: str | Path = ".",
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    threshold_bp: float = _THRESHOLD_BP,
) -> dict[str, Any]:
    """Evaluate today's published calendars and push the heads-up."""

    from edgestack.data.calendars import NYSECalendar

    base = Path(root).resolve()
    today = date.today()
    if not NYSECalendar().is_session(today):
        return {"status": "NON_SESSION", "date": today.isoformat()}
    rows = _today_calendar_rows(
        base / "artifacts" / "advisor", symbols, today.isoformat()
    )
    if not rows:
        return {"status": "NO_CALENDAR_ROWS_FOR_TODAY", "date": today.isoformat()}

    earnings_windows: dict[str, tuple[str, str]] = {}
    try:
        from edgestack.agenttools import _earnings_estimate

        for symbol in {str(row["symbol"]) for row in rows}:
            estimate = _earnings_estimate(symbol, base)
            if "window_start" in estimate:
                earnings_windows[symbol] = (
                    estimate["window_start"],
                    estimate["window_end"],
                )
    except Exception:
        pass

    candidates = [
        str(row["symbol"])
        for row in rows
        if float(row.get("expected_daily_bp") or 0.0) >= threshold_bp
    ]
    quotes = _delayed_quotes(candidates) if candidates else {}

    peers_note: str | None = None
    try:
        from edgestack.live.daily_job import _fetch_panel

        frames = asyncio.run(
            _fetch_panel(IT_SERVICES_PEERS, today - timedelta(days=60), today)
        )
        healing, total = peers_healing(frames)
        if total:
            peers_note = f"[peers healing: {healing}/{total}]"
    except Exception:
        pass

    lines = preclose_lines(
        rows,
        threshold_bp=threshold_bp,
        earnings_windows=earnings_windows,
        quotes=quotes,
        peers_note=peers_note,
    )
    summary: dict[str, Any] = {
        "status": "SIGNALS" if lines else "NO_SIGNALS",
        "date": today.isoformat(),
        "lines": lines,
        "peers": peers_note,
    }
    if lines:
        summary["telegram"] = _send(lines, today.isoformat())
    return summary


def _send(lines: Sequence[str], as_of: str) -> str:
    import os

    token = os.environ.get("EDGESTACK_TELEGRAM_TOKEN", "").strip()
    chat = os.environ.get("EDGESTACK_TELEGRAM_CHAT", "").strip()
    if not token or not chat:
        return "SKIPPED_NO_CREDENTIALS"
    from edgestack.live.notify import TelegramChannel
    from edgestack.models import AlertEvent

    message = (
        "\n".join(lines)
        + "\nPAPER ONLY · not financial advice · revalidate before the close"
    )
    event = AlertEvent(
        event_id=f"pre-close-{as_of}",
        recommendation_id="pre-close",
        revision=1,
        event_type="PRE_CLOSE_CHECK",
        message=message,
        created_at=datetime.now(UTC),
    )
    try:
        receipt = asyncio.run(
            TelegramChannel(token=token, chat_id=chat).send(event, event.event_id)
        )
        return f"SENT:{receipt}"
    except Exception as error:  # delivery failure must not fail the job
        return f"FAILED: {error}"
