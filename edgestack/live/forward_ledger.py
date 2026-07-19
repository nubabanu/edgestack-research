"""Append-only forward paper ledger for the promoted five-name basket.

Forward evidence only counts when it cannot be rewritten: every row is
INSERT-once, marks must move forward in session time, and nothing here ever
recomputes history. The ledger records what each nightly scan said and what
subsequently happened at the official closes, so the basket accrues a
tamper-evident post-freeze track record.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edgestack.provenance import canonical_sha256


class ForwardLedger:
    """Crash-safe, append-only record of signals, fills, marks, and exits."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS forward_signals (
                    signal_session TEXT PRIMARY KEY,
                    entry_session TEXT NOT NULL,
                    exit_session TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS forward_events (
                    recommendation_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    event TEXT NOT NULL CHECK(event IN ('FILL', 'MARK', 'EXIT')),
                    session TEXT NOT NULL,
                    price REAL NOT NULL CHECK(price > 0),
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (recommendation_id, event, session)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def record_signal(self, payload: Mapping[str, Any]) -> bool:
        """Persist one scan exactly once, keyed by its market session."""

        session = str(payload["market_as_of"]).split("_")[0]
        entry = str(payload["entry"]["session"])
        exit_ = str(payload["exit"]["session"])
        digest = canonical_sha256(dict(payload))
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO forward_signals VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session,
                    entry,
                    exit_,
                    digest,
                    datetime.now(UTC).isoformat(),
                    json.dumps(dict(payload), sort_keys=True, default=str),
                ),
            )
            return cursor.rowcount == 1

    def record_event(
        self,
        recommendation_id: str,
        *,
        symbol: str,
        event: str,
        session: str,
        price: float,
    ) -> bool:
        """Append one immutable close observation; duplicates are no-ops.

        A retroactive insert (a session at or before the latest recorded one
        for the same recommendation and event type) is rejected outright.
        """

        if event not in {"FILL", "MARK", "EXIT"}:
            raise ValueError("event must be FILL, MARK, or EXIT")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = connection.execute(
                """SELECT MAX(session) AS latest FROM forward_events
                WHERE recommendation_id=? AND event='MARK'""",
                (recommendation_id,),
            ).fetchone()
            if (
                event == "MARK"
                and latest["latest"] is not None
                and session <= str(latest["latest"])
            ):
                return False
            cursor = connection.execute(
                "INSERT OR IGNORE INTO forward_events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    recommendation_id,
                    symbol,
                    event,
                    session,
                    float(price),
                    datetime.now(UTC).isoformat(),
                ),
            )
            return cursor.rowcount == 1

    def open_signals(self, *, before_session: str) -> tuple[dict[str, Any], ...]:
        """Signals whose entry has occurred and whose exit is not yet recorded."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM forward_signals WHERE entry_session <= ? ORDER BY signal_session",
                (before_session,),
            ).fetchall()
        output = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            recommendation_ids = [
                str(item["recommendation_id"]) for item in payload["candidates"]
            ]
            with self._connect() as connection:
                exits = connection.execute(
                    """SELECT COUNT(*) AS n FROM forward_events
                    WHERE event='EXIT' AND recommendation_id IN ({})""".format(
                        ",".join("?" * len(recommendation_ids))
                    ),
                    recommendation_ids,
                ).fetchone()
            if int(exits["n"]) < len(recommendation_ids):
                output.append(
                    {
                        "signal_session": str(row["signal_session"]),
                        "entry_session": str(row["entry_session"]),
                        "exit_session": str(row["exit_session"]),
                        "payload": payload,
                    }
                )
        return tuple(output)

    def scorecard(self) -> dict[str, Any]:
        """Equal-weight net-of-nothing forward summary from recorded closes.

        Gross close-to-close arithmetic only — costs are reported by the
        tested contract, not reconstructed here; the point is a tamper-evident
        direction-of-travel record, labeled as such.
        """

        with self._connect() as connection:
            signals = connection.execute(
                "SELECT COUNT(*) AS n FROM forward_signals"
            ).fetchone()
            completed = connection.execute(
                """
                SELECT f.recommendation_id, f.symbol, f.price AS fill_price,
                       e.price AS exit_price
                FROM forward_events f
                JOIN forward_events e USING (recommendation_id)
                WHERE f.event='FILL' AND e.event='EXIT'
                """
            ).fetchall()
        returns = [
            float(row["exit_price"]) / float(row["fill_price"]) - 1.0
            for row in completed
        ]
        return {
            "policy": "FORWARD_GROSS_CLOSE_TO_CLOSE_APPEND_ONLY",
            "signals_recorded": int(signals["n"]),
            "completed_positions": len(returns),
            "mean_position_return": (
                sum(returns) / len(returns) if returns else None
            ),
            "win_rate": (
                sum(1 for value in returns if value > 0) / len(returns)
                if returns
                else None
            ),
        }


__all__ = ["ForwardLedger"]
