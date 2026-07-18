"""Append-only SQLite evidence for oil decisions and paper marks."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edgestack.oil.models import OilSnapshot
from edgestack.provenance import canonical_sha256


class OilLedger:
    """Crash-safe decision/event ledger with no update or delete path."""

    def __init__(self, path: str | Path = "artifacts/oil/forward.sqlite") -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS oil_decisions (
                    decision_id TEXT PRIMARY KEY,
                    market_as_of TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('NO_TRADE','WATCH','PAPER_LONG')),
                    payload_sha256 TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS oil_events (
                    decision_id TEXT NOT NULL,
                    horizon TEXT NOT NULL CHECK(horizon IN ('INTRADAY','SWING_3D')),
                    lane TEXT NOT NULL,
                    event TEXT NOT NULL CHECK(event IN ('FILL','MARK','EXIT','ETORO_MARK')),
                    available_at TEXT NOT NULL,
                    price REAL NOT NULL CHECK(price > 0),
                    units REAL NOT NULL CHECK(units >= 0),
                    source TEXT NOT NULL,
                    pnl_usd REAL,
                    equity_after_usd REAL CHECK(equity_after_usd >= 0),
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (decision_id, horizon, lane, event, available_at, source),
                    FOREIGN KEY (decision_id) REFERENCES oil_decisions(decision_id)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def record_snapshot(self, snapshot: OilSnapshot) -> bool:
        payload = snapshot.model_dump(mode="json")
        digest = canonical_sha256(payload)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT payload_sha256 FROM oil_decisions WHERE decision_id=?",
                (snapshot.decision_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["payload_sha256"]) != digest:
                    raise RuntimeError("oil decision identity collision")
                return False
            connection.execute(
                "INSERT INTO oil_decisions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    snapshot.decision_id,
                    snapshot.market_as_of,
                    snapshot.status,
                    digest,
                    datetime.now(UTC).isoformat(),
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ),
            )
        return True

    def record_event(
        self,
        decision_id: str,
        *,
        horizon: str,
        lane: str,
        event: str,
        available_at: datetime,
        price: float,
        units: float,
        source: str,
        pnl_usd: float | None = None,
        equity_after_usd: float | None = None,
    ) -> bool:
        if available_at.tzinfo is None:
            raise ValueError("oil event availability must be timezone-aware")
        if event not in {"FILL", "MARK", "EXIT", "ETORO_MARK"}:
            raise ValueError("invalid oil paper event")
        if price <= 0 or units < 0:
            raise ValueError("oil paper event price/units are invalid")
        if equity_after_usd is not None and equity_after_usd < 0:
            raise ValueError("oil paper equity cannot be negative")
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO oil_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision_id,
                    horizon,
                    lane,
                    event,
                    available_at.astimezone(UTC).isoformat(),
                    float(price),
                    float(units),
                    source,
                    pnl_usd,
                    equity_after_usd,
                    datetime.now(UTC).isoformat(),
                ),
            )
            return cursor.rowcount == 1

    def has_open_position(self) -> bool:
        with self._connect() as connection:
            fills = connection.execute(
                """SELECT decision_id, horizon, lane FROM oil_events
                WHERE event='FILL'"""
            ).fetchall()
            for fill in fills:
                exit_row = connection.execute(
                    """SELECT 1 FROM oil_events WHERE decision_id=? AND horizon=?
                    AND lane=? AND event='EXIT' LIMIT 1""",
                    (fill["decision_id"], fill["horizon"], fill["lane"]),
                ).fetchone()
                if exit_row is None:
                    return True
        return False

    def lane_state(
        self, *, initial_equity_usd: float
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return latest and peak immutable equity marks for each lane."""

        if initial_equity_usd <= 0:
            raise ValueError("initial paper equity must be positive")
        current: dict[str, float] = {}
        peaks: dict[str, float] = {}
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT lane, equity_after_usd, available_at FROM oil_events
                WHERE equity_after_usd IS NOT NULL
                ORDER BY available_at, recorded_at, rowid"""
            ).fetchall()
        for row in rows:
            lane = str(row["lane"])
            value = float(row["equity_after_usd"])
            current[lane] = value
            peaks[lane] = max(peaks.get(lane, initial_equity_usd), value)
        return current, peaks

    def terminated_challenge_lanes(
        self, *, initial_equity_usd: float
    ) -> set[str]:
        """Return lanes that ever crossed the irreversible 30% campaign stop."""

        peaks: dict[str, float] = {}
        terminated: set[str] = set()
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT lane, equity_after_usd FROM oil_events
                WHERE equity_after_usd IS NOT NULL
                ORDER BY available_at, recorded_at, rowid"""
            ).fetchall()
        for row in rows:
            lane = str(row["lane"])
            if not lane.startswith("CHALLENGE_"):
                continue
            value = float(row["equity_after_usd"])
            peak = max(peaks.get(lane, initial_equity_usd), value)
            peaks[lane] = peak
            if value <= 0 or value / peak - 1.0 <= -0.30:
                terminated.add(lane)
        return terminated

    def reconcile_proxy_bars(
        self, frame: Any, *, at: datetime, stop_slippage_bps: float = 15.0
    ) -> int:
        """Append causal USO fills, close marks, and due exits for paper longs."""

        import pandas as pd

        if at.tzinfo is None:
            raise ValueError("oil reconciliation time must be timezone-aware")
        if stop_slippage_bps < 0:
            raise ValueError("oil stop slippage cannot be negative")
        bars = frame.copy(deep=True)
        bars["session"] = pd.to_datetime(bars["session"]).dt.normalize()
        bars["available_at"] = pd.to_datetime(bars["available_at"], utc=True)
        bars = bars.loc[bars["available_at"] <= pd.Timestamp(at)].set_index("session")
        if bars.empty:
            return 0
        factor = bars["adjusted_close"].astype(float) / bars["close"].astype(float)
        bars["adjusted_open"] = bars["open"].astype(float) * factor
        bars["adjusted_low"] = bars["low"].astype(float) * factor
        inserted = 0
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM oil_decisions WHERE status='PAPER_LONG'"
            ).fetchall()
        for row in rows:
            snapshot = OilSnapshot.model_validate_json(str(row["payload_json"]))
            for decision in (snapshot.intraday, snapshot.swing):
                if decision.status != "PAPER_LONG":
                    continue
                entry_day = pd.Timestamp(decision.planned_entry[:10])
                exit_day = pd.Timestamp(decision.planned_exit[:10])
                if entry_day not in bars.index:
                    continue
                entry_bar = bars.loc[entry_day]
                if isinstance(entry_bar, pd.DataFrame):
                    entry_bar = entry_bar.iloc[-1]
                fill_price = float(entry_bar["adjusted_open"])
                fill_at = pd.Timestamp(entry_bar["available_at"]).to_pydatetime()
                for lane in decision.lanes:
                    if lane.status != "ACTIVE" or lane.notional_usd <= 0:
                        continue
                    units = lane.notional_usd / fill_price
                    inserted += int(
                        self.record_event(
                            snapshot.decision_id,
                            horizon=decision.horizon,
                            lane=lane.name,
                            event="FILL",
                            available_at=fill_at,
                            price=fill_price,
                            units=units,
                            source=f"USO_ADJUSTED_OPEN:{entry_day.date()}",
                        )
                    )
                    eligible = bars.loc[
                        (bars.index >= entry_day) & (bars.index <= exit_day)
                    ]
                    for session, mark in eligible.iterrows():
                        scheduled_exit = session == exit_day
                        stop_reference = fill_price * (1.0 - lane.stop_fraction)
                        stop_breached = float(mark["adjusted_low"]) <= stop_reference
                        if stop_breached:
                            slipped_stop = stop_reference * (
                                1.0 - stop_slippage_bps / 10_000.0
                            )
                            mark_price = min(
                                slipped_stop, float(mark["adjusted_open"])
                            )
                        else:
                            mark_price = float(mark["adjusted_close"])
                        pnl = units * (mark_price - fill_price) - lane.estimated_cost_usd
                        equity = max(0.0, lane.equity_usd + pnl)
                        available_at = pd.Timestamp(mark["available_at"]).to_pydatetime()
                        inserted += int(
                            self.record_event(
                                snapshot.decision_id,
                                horizon=decision.horizon,
                                lane=lane.name,
                                event="MARK",
                                available_at=available_at,
                                price=mark_price,
                                units=units,
                                source=f"USO_ADJUSTED_CLOSE:{session.date()}",
                                pnl_usd=pnl,
                                equity_after_usd=equity,
                            )
                        )
                        if stop_breached or scheduled_exit:
                            source = (
                                f"PESSIMISTIC_2ATR_STOP:{session.date()}"
                                if stop_breached
                                else f"USO_ADJUSTED_CLOSE_EXIT:{session.date()}"
                            )
                            inserted += int(
                                self.record_event(
                                    snapshot.decision_id,
                                    horizon=decision.horizon,
                                    lane=lane.name,
                                    event="EXIT",
                                    available_at=available_at,
                                    price=mark_price,
                                    units=units,
                                    source=source,
                                    pnl_usd=pnl,
                                    equity_after_usd=equity,
                                )
                            )
                            break
        return inserted

    def risk_usage(
        self, *, at: datetime
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return same-day realized losses and planned risk of open fills."""

        if at.tzinfo is None:
            raise ValueError("oil risk-usage time must be timezone-aware")
        from zoneinfo import ZoneInfo

        local_day = at.astimezone(ZoneInfo("America/New_York")).date()
        daily: dict[str, float] = {}
        open_risk: dict[str, float] = {}
        with self._connect() as connection:
            exits = connection.execute(
                """SELECT lane, pnl_usd, available_at FROM oil_events
                WHERE event='EXIT' AND pnl_usd < 0"""
            ).fetchall()
            fills = connection.execute(
                """SELECT f.decision_id, f.horizon, f.lane, d.payload_json
                FROM oil_events AS f JOIN oil_decisions AS d
                ON d.decision_id=f.decision_id
                WHERE f.event='FILL' AND NOT EXISTS (
                    SELECT 1 FROM oil_events AS x
                    WHERE x.decision_id=f.decision_id AND x.horizon=f.horizon
                    AND x.lane=f.lane AND x.event='EXIT'
                )"""
            ).fetchall()
        for row in exits:
            available = datetime.fromisoformat(str(row["available_at"]))
            if available.astimezone(ZoneInfo("America/New_York")).date() == local_day:
                lane = str(row["lane"])
                daily[lane] = daily.get(lane, 0.0) + abs(float(row["pnl_usd"]))
        for row in fills:
            payload = json.loads(str(row["payload_json"]))
            key = "intraday" if row["horizon"] == "INTRADAY" else "swing"
            lanes = payload.get(key, {}).get("lanes", [])
            for item in lanes:
                if item.get("name") == row["lane"]:
                    lane = str(row["lane"])
                    open_risk[lane] = open_risk.get(lane, 0.0) + float(
                        item.get("maximum_planned_loss_usd", 0.0)
                    )
                    break
        return daily, open_risk

    def scorecard(self) -> dict[str, Any]:
        with self._connect() as connection:
            decisions = connection.execute(
                "SELECT status, COUNT(*) AS n FROM oil_decisions GROUP BY status"
            ).fetchall()
            events = connection.execute(
                "SELECT event, COUNT(*) AS n FROM oil_events GROUP BY event"
            ).fetchall()
            completed = connection.execute(
                """SELECT lane, COUNT(*) AS n, AVG(pnl_usd) AS mean_pnl,
                SUM(CASE WHEN pnl_usd > 0 THEN 1.0 ELSE 0.0 END) / COUNT(*) AS win_rate,
                MIN(equity_after_usd) AS minimum_equity,
                MAX(equity_after_usd) AS maximum_equity
                FROM oil_events WHERE event='EXIT' AND pnl_usd IS NOT NULL
                GROUP BY lane ORDER BY lane"""
            ).fetchall()
        return {
            "policy": "PAPER_ONLY_APPEND_ONLY_NO_BROKER_PATH",
            "decisions": {str(row["status"]): int(row["n"]) for row in decisions},
            "events": {str(row["event"]): int(row["n"]) for row in events},
            "lanes": [dict(row) for row in completed],
        }


__all__ = ["OilLedger"]
