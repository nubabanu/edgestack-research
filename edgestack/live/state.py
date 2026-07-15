"""SQLite recommendation state machine, transactional outbox, and paper ledger."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from edgestack.models import AlertEvent, Direction, Recommendation, RecommendationState

_ALLOWED: dict[RecommendationState, frozenset[RecommendationState]] = {
    RecommendationState.PROPOSED: frozenset(
        {
            RecommendationState.WAITING,
            RecommendationState.CONFIRMED,
            RecommendationState.CANCELLED,
        }
    ),
    RecommendationState.WAITING: frozenset(
        {
            RecommendationState.CONFIRMED,
            RecommendationState.UPDATED,
            RecommendationState.CANCELLED,
        }
    ),
    RecommendationState.UPDATED: frozenset(
        {
            RecommendationState.WAITING,
            RecommendationState.CONFIRMED,
            RecommendationState.CANCELLED,
        }
    ),
    RecommendationState.CONFIRMED: frozenset(
        {
            RecommendationState.UPDATED,
            RecommendationState.CANCELLED,
            RecommendationState.ENTERED,
        }
    ),
    RecommendationState.CANCELLED: frozenset(),
    RecommendationState.ENTERED: frozenset({RecommendationState.EXITED}),
    RecommendationState.EXITED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class OutboxRecord:
    """One pending external channel delivery."""

    outbox_id: int
    idempotency_key: str
    event: AlertEvent
    channel: str
    attempts: int


class StateStore:
    """Atomically persist state changes and logical notification events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        """Return a WAL-enabled connection."""

        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS recommendations (
                    recommendation_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS transitions (
                    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recommendation_id TEXT NOT NULL,
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    FOREIGN KEY(recommendation_id) REFERENCES recommendations(recommendation_id)
                );
                CREATE TABLE IF NOT EXISTS outbox (
                    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    recommendation_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lease_until TEXT,
                    last_error TEXT,
                    provider_receipt TEXT,
                    FOREIGN KEY(recommendation_id) REFERENCES recommendations(recommendation_id)
                );
                CREATE TABLE IF NOT EXISTS paper_ledger (
                    recommendation_id TEXT PRIMARY KEY,
                    entry_time TEXT,
                    entry_price REAL,
                    shares INTEGER,
                    exit_time TEXT,
                    exit_price REAL,
                    costs REAL NOT NULL DEFAULT 0,
                    net_pnl REAL,
                    FOREIGN KEY(recommendation_id) REFERENCES recommendations(recommendation_id)
                );
                """
            )

    def add(self, recommendation: Recommendation, channels: Iterable[str]) -> bool:
        """Insert a new recommendation and initial logical event exactly once."""

        _aware(recommendation.created_at)
        channel_names = _channels(channels)
        payload = json.dumps(asdict(recommendation), sort_keys=True, default=str)
        now = recommendation.created_at.isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO recommendations VALUES (?, ?, 0, ?, ?, ?)",
                (
                    recommendation.recommendation_id,
                    RecommendationState.PROPOSED.value,
                    payload,
                    now,
                    now,
                ),
            )
            if cursor.rowcount == 0:
                return False
            connection.execute(
                "INSERT INTO transitions(recommendation_id, from_state, to_state, revision, reason, occurred_at) VALUES (?, NULL, ?, 0, ?, ?)",
                (
                    recommendation.recommendation_id,
                    RecommendationState.PROPOSED.value,
                    "scanner proposal",
                    now,
                ),
            )
            self._enqueue_many(
                connection,
                recommendation.recommendation_id,
                0,
                "PROPOSED",
                _proposal_message(recommendation),
                now,
                channel_names,
            )
        return True

    def state(self, recommendation_id: str) -> RecommendationState:
        """Return the current recommendation state."""

        with self.connect() as connection:
            row = connection.execute(
                "SELECT state FROM recommendations WHERE recommendation_id=?",
                (recommendation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(recommendation_id)
        return RecommendationState(row["state"])

    def transition(
        self,
        recommendation_id: str,
        to_state: RecommendationState,
        reason: str,
        message: str,
        channels: Iterable[str],
        *,
        expected_revision: int | None = None,
        occurred_at: datetime | None = None,
        updated_recommendation: Recommendation | None = None,
    ) -> int:
        """Commit a validated state transition and outbox events atomically."""

        occurred = occurred_at or datetime.now(UTC)
        _aware(occurred)
        if not reason.strip() or not message.strip():
            raise ValueError("transition reason and message cannot be empty")
        channel_names = _channels(channels)
        if (
            updated_recommendation is not None
            and updated_recommendation.recommendation_id != recommendation_id
        ):
            raise ValueError("updated payload belongs to another recommendation")
        now = occurred.isoformat()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state, revision FROM recommendations WHERE recommendation_id=?",
                (recommendation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(recommendation_id)
            current = RecommendationState(row["state"])
            revision = int(row["revision"])
            if expected_revision is not None and revision != expected_revision:
                raise RuntimeError("stale recommendation revision")
            if to_state not in _ALLOWED[current]:
                raise ValueError(
                    f"invalid transition {current.value} -> {to_state.value}"
                )
            new_revision = revision + 1
            if updated_recommendation is None:
                connection.execute(
                    "UPDATE recommendations SET state=?, revision=?, updated_at=? WHERE recommendation_id=?",
                    (to_state.value, new_revision, now, recommendation_id),
                )
            else:
                payload = json.dumps(
                    asdict(updated_recommendation), sort_keys=True, default=str
                )
                connection.execute(
                    "UPDATE recommendations SET state=?, revision=?, payload_json=?, updated_at=? WHERE recommendation_id=?",
                    (to_state.value, new_revision, payload, now, recommendation_id),
                )
            connection.execute(
                "INSERT INTO transitions(recommendation_id, from_state, to_state, revision, reason, occurred_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    recommendation_id,
                    current.value,
                    to_state.value,
                    new_revision,
                    reason,
                    now,
                ),
            )
            self._enqueue_many(
                connection,
                recommendation_id,
                new_revision,
                to_state.value,
                message,
                now,
                channel_names,
            )
        return new_revision

    @staticmethod
    def _enqueue_many(
        connection: sqlite3.Connection,
        recommendation_id: str,
        revision: int,
        event_type: str,
        message: str,
        created_at: str,
        channels: Iterable[str],
    ) -> None:
        for channel in sorted(set(channels)):
            key = f"{recommendation_id}:{revision}:{event_type}:{channel}"
            connection.execute(
                """INSERT OR IGNORE INTO outbox(
                idempotency_key, recommendation_id, revision, event_type, message,
                created_at, channel) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    key,
                    recommendation_id,
                    revision,
                    event_type,
                    message,
                    created_at,
                    channel,
                ),
            )

    def lease_outbox(
        self, limit: int = 100, lease_seconds: int = 60
    ) -> tuple[OutboxRecord, ...]:
        """Lease pending/retry deliveries for one dispatcher instance."""

        if limit <= 0 or lease_seconds <= 0:
            raise ValueError("outbox lease limit and duration must be positive")
        now = datetime.now(UTC)
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """SELECT * FROM outbox
                WHERE status IN ('pending','retry')
                   OR (status='leased' AND lease_until < ?)
                ORDER BY outbox_id LIMIT ?""",
                (now.isoformat(), limit),
            ).fetchall()
            ids = [int(row["outbox_id"]) for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                connection.execute(
                    f"UPDATE outbox SET status='leased', lease_until=?, attempts=attempts+1 WHERE outbox_id IN ({placeholders})",
                    (lease_until, *ids),
                )
        return tuple(
            OutboxRecord(
                outbox_id=int(row["outbox_id"]),
                idempotency_key=row["idempotency_key"],
                event=AlertEvent(
                    event_id=(
                        f"{row['recommendation_id']}:{int(row['revision'])}:"
                        f"{row['event_type']}"
                    ),
                    recommendation_id=row["recommendation_id"],
                    revision=int(row["revision"]),
                    event_type=row["event_type"],
                    message=row["message"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                ),
                channel=row["channel"],
                attempts=int(row["attempts"]) + 1,
            )
            for row in rows
        )

    def acknowledge(self, outbox_id: int, provider_receipt: str) -> None:
        """Mark a delivery acknowledged by its provider."""

        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE outbox SET status='sent', lease_until=NULL, provider_receipt=? WHERE outbox_id=? AND status='leased'",
                (provider_receipt, outbox_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("outbox record is not currently leased")

    def retry(self, outbox_id: int, error: str, *, dead_after: int = 8) -> None:
        """Return a failed lease to retry or dead-letter it."""

        if dead_after <= 0:
            raise ValueError("dead_after must be positive")
        with self.connect() as connection:
            row = connection.execute(
                "SELECT attempts, status FROM outbox WHERE outbox_id=?", (outbox_id,)
            ).fetchone()
            if row is None:
                raise KeyError(outbox_id)
            if row["status"] != "leased":
                raise RuntimeError("outbox record is not currently leased")
            status = "dead" if int(row["attempts"]) >= dead_after else "retry"
            connection.execute(
                "UPDATE outbox SET status=?, lease_until=NULL, last_error=? WHERE outbox_id=?",
                (status, error[:1000], outbox_id),
            )

    def logical_event_counts(self) -> dict[str, int]:
        """Return delivery-status counts for restart assertions."""

        with self.connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM outbox GROUP BY status"
            ).fetchall()
        return {row["status"]: int(row["count"]) for row in rows}

    def record_entry(
        self,
        recommendation_id: str,
        when: datetime,
        price: float,
        shares: int,
        costs: float,
    ) -> None:
        """Record a hypothetical entry exactly once."""

        if price <= 0 or shares <= 0 or costs < 0:
            raise ValueError("invalid paper entry")
        _aware(when)
        with self.connect() as connection:
            state_row = connection.execute(
                "SELECT state FROM recommendations WHERE recommendation_id=?",
                (recommendation_id,),
            ).fetchone()
            if state_row is None:
                raise KeyError(recommendation_id)
            if (
                RecommendationState(state_row["state"])
                is not RecommendationState.ENTERED
            ):
                raise ValueError("paper entry requires ENTERED recommendation state")
            connection.execute(
                "INSERT INTO paper_ledger(recommendation_id, entry_time, entry_price, shares, costs) VALUES (?, ?, ?, ?, ?)",
                (recommendation_id, when.isoformat(), price, shares, costs),
            )

    def record_exit(
        self, recommendation_id: str, when: datetime, price: float, exit_costs: float
    ) -> float:
        """Close a paper position exactly once and return direction-adjusted P&L."""

        if price <= 0 or exit_costs < 0:
            raise ValueError("invalid paper exit")
        _aware(when)
        with self.connect() as connection:
            row = connection.execute(
                """SELECT p.entry_price, p.shares, p.costs, p.exit_time,
                          r.state, r.payload_json
                   FROM paper_ledger p JOIN recommendations r USING(recommendation_id)
                   WHERE recommendation_id=?""",
                (recommendation_id,),
            ).fetchone()
            if row is None or row["entry_price"] is None:
                raise KeyError(recommendation_id)
            if row["exit_time"] is not None:
                raise ValueError("paper position is already closed")
            if RecommendationState(row["state"]) is not RecommendationState.EXITED:
                raise ValueError("paper exit requires EXITED recommendation state")
            payload = json.loads(row["payload_json"])
            multiplier = (
                -1.0 if payload.get("direction") == Direction.SHORT.value else 1.0
            )
            pnl = (
                multiplier * (price - float(row["entry_price"])) * int(row["shares"])
                - float(row["costs"])
                - exit_costs
            )
            connection.execute(
                "UPDATE paper_ledger SET exit_time=?, exit_price=?, costs=costs+?, net_pnl=? WHERE recommendation_id=?",
                (when.isoformat(), price, exit_costs, pnl, recommendation_id),
            )
        return pnl


def _proposal_message(recommendation: Recommendation) -> str:
    action = recommendation.entry_plan.verdict.value
    borrow = (
        " Borrow availability is unverified."
        if recommendation.direction is Direction.SHORT
        and not recommendation.borrow_verified
        else ""
    )
    return (
        f"{action}: {recommendation.direction.value} {recommendation.asset.symbol}; "
        f"confidence {recommendation.confidence}/100; plan {recommendation.entry_plan.method}."
        f"{borrow} Recommendation ID {recommendation.recommendation_id}."
    )


def _channels(channels: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(
        sorted({channel.strip() for channel in channels if channel.strip()})
    )
    if not normalized:
        raise ValueError("at least one notification channel is required")
    return normalized


def _aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("paper/live timestamps must be timezone-aware")
