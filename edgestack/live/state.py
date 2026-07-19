"""SQLite recommendation state machine, transactional outbox, and paper ledger."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
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


@dataclass(frozen=True, slots=True)
class SignalDecision:
    """Immutable candidate/skip decision written by one causal signal run."""

    decision_id: str
    security_id: str
    ticker: str
    action: str
    direction: str | None
    veto_reason: str | None
    quote_freshness_seconds: float | None
    intended_entry_at: datetime | None
    intended_exit_at: datetime | None
    leverage: float
    horizon_sessions: int
    payload: Mapping[str, object]


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
                CREATE TABLE IF NOT EXISTS signal_runs (
                    run_id TEXT PRIMARY KEY,
                    campaign_namespace TEXT NOT NULL CHECK(campaign_namespace='loss-aware-v2'),
                    decision_time TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    causal_data_hash TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS signal_decisions (
                    decision_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    security_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('CANDIDATE','SKIP')),
                    direction TEXT,
                    veto_reason TEXT,
                    quote_freshness_seconds REAL,
                    intended_entry_at TEXT,
                    actual_fill_at TEXT,
                    actual_fill_price REAL,
                    intended_exit_at TEXT,
                    leverage REAL NOT NULL,
                    horizon_sessions INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES signal_runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS paper_marks (
                    decision_id TEXT NOT NULL,
                    mark_at TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    price REAL NOT NULL,
                    causal_data_hash TEXT NOT NULL,
                    mae REAL,
                    mfe REAL,
                    PRIMARY KEY(decision_id, mark_at),
                    FOREIGN KEY(decision_id) REFERENCES signal_decisions(decision_id)
                );
                CREATE TABLE IF NOT EXISTS paper_outcomes (
                    decision_id TEXT PRIMARY KEY,
                    exit_at TEXT NOT NULL,
                    exit_price REAL NOT NULL,
                    gross_pnl REAL NOT NULL,
                    costs REAL NOT NULL,
                    net_pnl REAL NOT NULL,
                    stop_gap REAL,
                    stop_slippage REAL,
                    transition_reason TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    FOREIGN KEY(decision_id) REFERENCES signal_decisions(decision_id)
                );
                CREATE TABLE IF NOT EXISTS signal_outbox (
                    idempotency_key TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    FOREIGN KEY(run_id) REFERENCES signal_runs(run_id),
                    FOREIGN KEY(decision_id) REFERENCES signal_decisions(decision_id)
                );
                PRAGMA user_version=2;
                """
            )

    def create_signal_run(
        self,
        *,
        run_id: str,
        decision_time: datetime,
        created_at: datetime,
        causal_data_hash: str,
        config_hash: str,
        decisions: Iterable[SignalDecision],
        channels: Iterable[str] = ("console",),
        payload: Mapping[str, object] | None = None,
    ) -> bool:
        """Atomically create a V2 run, every candidate/skip, and alert events."""

        _aware(decision_time)
        _aware(created_at)
        if created_at < decision_time:
            raise ValueError("created_at cannot precede decision_time")
        if not run_id or len(causal_data_hash) != 64 or len(config_hash) != 64:
            raise ValueError("run identity and SHA-256 hashes are required")
        rows = tuple(decisions)
        if not rows:
            raise ValueError("a signal run must record candidates or skips")
        channel_names = _channels(channels)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """INSERT OR IGNORE INTO signal_runs VALUES
                (?, 'loss-aware-v2', ?, ?, ?, ?, 'RECORDED', ?)""",
                (
                    run_id,
                    decision_time.isoformat(),
                    created_at.isoformat(),
                    causal_data_hash,
                    config_hash,
                    json.dumps(dict(payload or {}), sort_keys=True, default=str),
                ),
            )
            if cursor.rowcount == 0:
                return False
            for item in rows:
                if item.action not in {"CANDIDATE", "SKIP"}:
                    raise ValueError("decision action must be CANDIDATE or SKIP")
                if item.leverage not in {
                    1.0,
                    1.5,
                    2.0,
                } or item.horizon_sessions not in {
                    21,
                    252,
                }:
                    raise ValueError("undeclared leverage or horizon")
                for stamp in (item.intended_entry_at, item.intended_exit_at):
                    if stamp is not None:
                        _aware(stamp)
                connection.execute(
                    """INSERT INTO signal_decisions(
                    decision_id, run_id, security_id, ticker, action, direction,
                    veto_reason, quote_freshness_seconds, intended_entry_at,
                    intended_exit_at, leverage, horizon_sessions, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        item.decision_id,
                        run_id,
                        item.security_id,
                        item.ticker,
                        item.action,
                        item.direction,
                        item.veto_reason,
                        item.quote_freshness_seconds,
                        (
                            item.intended_entry_at.isoformat()
                            if item.intended_entry_at
                            else None
                        ),
                        (
                            item.intended_exit_at.isoformat()
                            if item.intended_exit_at
                            else None
                        ),
                        item.leverage,
                        item.horizon_sessions,
                        json.dumps(dict(item.payload), sort_keys=True, default=str),
                    ),
                )
                for channel in channel_names:
                    key = f"{run_id}:{item.decision_id}:{channel}"
                    connection.execute(
                        "INSERT INTO signal_outbox VALUES (?, ?, ?, ?, ?, ?, 'pending')",
                        (
                            key,
                            run_id,
                            item.decision_id,
                            channel,
                            json.dumps(
                                {"action": item.action, "ticker": item.ticker},
                                sort_keys=True,
                            ),
                            created_at.isoformat(),
                        ),
                    )
        return True

    def record_paper_fill(
        self, decision_id: str, fill_at: datetime, price: float
    ) -> None:
        """Record the first paper fill exactly once and after its decision."""

        _aware(fill_at)
        if price <= 0:
            raise ValueError("fill price must be positive")
        with self.connect() as connection:
            row = connection.execute(
                """SELECT d.actual_fill_at, r.decision_time
                FROM signal_decisions d JOIN signal_runs r USING(run_id)
                WHERE d.decision_id=?""",
                (decision_id,),
            ).fetchone()
            if row is None:
                raise KeyError(decision_id)
            if row["actual_fill_at"] is not None:
                raise ValueError("paper fill is immutable")
            if fill_at <= datetime.fromisoformat(row["decision_time"]):
                raise ValueError("paper fill must be after the decision")
            connection.execute(
                "UPDATE signal_decisions SET actual_fill_at=?, actual_fill_price=? WHERE decision_id=?",
                (fill_at.isoformat(), price, decision_id),
            )

    def record_paper_mark(
        self,
        decision_id: str,
        *,
        mark_at: datetime,
        available_at: datetime,
        price: float,
        causal_data_hash: str,
    ) -> tuple[float, float]:
        """Append a causal mark; retroactive or revised history is rejected."""

        _aware(mark_at)
        _aware(available_at)
        if available_at < mark_at or price <= 0 or len(causal_data_hash) != 64:
            raise ValueError("invalid paper mark or provenance")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT actual_fill_at, actual_fill_price, direction
                FROM signal_decisions WHERE decision_id=?""",
                (decision_id,),
            ).fetchone()
            if row is None:
                raise KeyError(decision_id)
            if row["actual_fill_at"] is None:
                raise ValueError("cannot mark an unfilled decision")
            fill_at = datetime.fromisoformat(row["actual_fill_at"])
            if mark_at < fill_at:
                raise ValueError("mark cannot precede fill")
            latest = connection.execute(
                "SELECT mark_at, available_at FROM paper_marks WHERE decision_id=? ORDER BY mark_at DESC LIMIT 1",
                (decision_id,),
            ).fetchone()
            if latest is not None and (
                mark_at <= datetime.fromisoformat(latest["mark_at"])
                or available_at <= datetime.fromisoformat(latest["available_at"])
            ):
                raise ValueError("retroactive paper marks are prohibited")
            multiplier = -1.0 if row["direction"] == Direction.SHORT.value else 1.0
            current = multiplier * (price / float(row["actual_fill_price"]) - 1)
            prior = connection.execute(
                "SELECT MIN(mae) AS mae, MAX(mfe) AS mfe FROM paper_marks WHERE decision_id=?",
                (decision_id,),
            ).fetchone()
            mae = (
                min(current, float(prior["mae"]))
                if prior["mae"] is not None
                else current
            )
            mfe = (
                max(current, float(prior["mfe"]))
                if prior["mfe"] is not None
                else current
            )
            connection.execute(
                "INSERT INTO paper_marks VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    decision_id,
                    mark_at.isoformat(),
                    available_at.isoformat(),
                    price,
                    causal_data_hash,
                    mae,
                    mfe,
                ),
            )
        return float(mae), float(mfe)

    def record_paper_outcome(
        self,
        decision_id: str,
        *,
        exit_at: datetime,
        exit_price: float,
        gross_pnl: float,
        costs: float,
        stop_gap: float | None,
        stop_slippage: float | None,
        transition_reason: str,
        recorded_at: datetime,
    ) -> None:
        """Write an immutable completed outcome; later rewrites fail."""

        _aware(exit_at)
        _aware(recorded_at)
        if (
            exit_price <= 0
            or costs < 0
            or recorded_at < exit_at
            or not transition_reason
        ):
            raise ValueError("invalid paper outcome")
        with self.connect() as connection:
            timing = connection.execute(
                """SELECT d.actual_fill_at, MAX(m.mark_at) AS latest_mark
                FROM signal_decisions d
                LEFT JOIN paper_marks m USING(decision_id)
                WHERE d.decision_id=? GROUP BY d.decision_id""",
                (decision_id,),
            ).fetchone()
            if timing is None or timing["actual_fill_at"] is None:
                raise ValueError("paper outcome requires a filled decision")
            last_observation = datetime.fromisoformat(
                timing["latest_mark"] or timing["actual_fill_at"]
            )
            if exit_at < last_observation:
                raise ValueError("retroactive paper outcomes are prohibited")
            connection.execute(
                """INSERT INTO paper_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    decision_id,
                    exit_at.isoformat(),
                    exit_price,
                    gross_pnl,
                    costs,
                    gross_pnl - costs,
                    stop_gap,
                    stop_slippage,
                    transition_reason,
                    recorded_at.isoformat(),
                ),
            )

    def paper_scorecard(self) -> dict[str, float | int]:
        """Replay stored V2 evidence without fetching, backfilling, or recomputing signals."""

        with self.connect() as connection:
            decisions = connection.execute(
                "SELECT COUNT(*) AS n, SUM(action='CANDIDATE') AS candidates, SUM(action='SKIP') AS skips FROM signal_decisions"
            ).fetchone()
            outcomes = connection.execute(
                "SELECT COUNT(*) AS n, AVG(net_pnl) AS mean, SUM(net_pnl < 0) AS losses, SUM(net_pnl) AS total FROM paper_outcomes"
            ).fetchone()
        completed = int(outcomes["n"] or 0)
        return {
            "decisions": int(decisions["n"] or 0),
            "candidates": int(decisions["candidates"] or 0),
            "skips": int(decisions["skips"] or 0),
            "completed": completed,
            "loss_probability": (
                float(outcomes["losses"] or 0) / completed if completed else 0.0
            ),
            "mean_net_pnl": float(outcomes["mean"] or 0.0),
            "total_net_pnl": float(outcomes["total"] or 0.0),
        }

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
