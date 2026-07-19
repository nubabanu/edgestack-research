"""Forward-only intraday bar collector (free Alpaca IEX feed).

Free historical intraday data effectively does not exist, so this module
builds the archive the honest way: capture each completed session going
forward. With free Alpaca keys (`ALPACA_KEY_ID` / `ALPACA_SECRET_KEY`) it
pulls one-minute IEX-feed bars for the declared symbols and writes one
immutable parquet per symbol per session under ``artifacts/intraday/``.
Without keys it exits cleanly with a DATA_UNAVAILABLE message — it never
fabricates and never fails the surrounding job.

The IEX feed covers a small fraction of consolidated volume; every file is
stamped ``IEX_FEED_PARTIAL_VOLUME`` so no later study can mistake it for
primary-exchange auction data.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

_BARS_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
_DEFAULT_SYMBOLS = ("SPY", "QQQ", "GLD", "ACN", "CTSH")
WATERMARK = "IEX_FEED_PARTIAL_VOLUME"


def _credentials() -> dict[str, str] | None:
    key = os.environ.get("ALPACA_KEY_ID", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _fetch_symbol_bars(client: httpx.Client, symbol: str, session: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params: dict[str, str] = {
            "timeframe": "1Min",
            "start": f"{session}T00:00:00Z",
            "end": f"{session}T23:59:59Z",
            "feed": "iex",
            "limit": "10000",
        }
        if page_token:
            params["page_token"] = page_token
        response = client.get(_BARS_URL.format(symbol=symbol), params=params)
        response.raise_for_status()
        payload = response.json()
        rows.extend(payload.get("bars") or [])
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "n": "trades",
                "vw": "vwap",
            }
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def capture(
    session: str,
    symbols: Sequence[str] = _DEFAULT_SYMBOLS,
    *,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Capture one completed session; idempotent per (symbol, session)."""

    headers = _credentials()
    if headers is None:
        return {
            "status": "DATA_UNAVAILABLE",
            "reason": (
                "no ALPACA_KEY_ID/ALPACA_SECRET_KEY in the environment; a free"
                " Alpaca account provides them and enables forward capture"
            ),
        }
    base = Path(root).resolve() / "artifacts" / "intraday"
    summary: dict[str, Any] = {
        "status": "CAPTURED",
        "session": session,
        "watermark": WATERMARK,
        "symbols": {},
    }
    with httpx.Client(headers=headers, timeout=30.0) as client:
        for symbol in symbols:
            target = base / symbol.upper() / f"{session}.parquet"
            if target.is_file():
                summary["symbols"][symbol] = "ALREADY_CAPTURED"
                continue
            try:
                frame = _fetch_symbol_bars(client, symbol.upper(), session)
            except Exception as error:  # one symbol must not kill the capture
                summary["symbols"][symbol] = f"FAILED: {error}"
                continue
            if not len(frame):
                summary["symbols"][symbol] = "NO_BARS_RETURNED"
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(target, index=False)
            manifest = {
                "symbol": symbol.upper(),
                "session": session,
                "bars": len(frame),
                "feed": "iex",
                "watermark": WATERMARK,
                "captured_at": datetime.now(UTC).isoformat(),
            }
            target.with_suffix(".json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summary["symbols"][symbol] = f"OK ({len(frame)} bars)"
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        default="",
        help="Session date YYYY-MM-DD; default is the last completed session.",
    )
    parser.add_argument("--symbols", default=",".join(_DEFAULT_SYMBOLS))
    parser.add_argument("--root", default=".")
    arguments = parser.parse_args(argv)
    session = arguments.session
    if not session:
        from edgestack.live.daily_job import last_completed_session

        session = last_completed_session().isoformat()
    symbols = [s.strip().upper() for s in arguments.symbols.split(",") if s.strip()]
    summary = capture(session, symbols, root=arguments.root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
