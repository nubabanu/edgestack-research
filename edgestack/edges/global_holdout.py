"""Global, data-scope holdout ledger independent of campaign names."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from edgestack.provenance import canonical_sha256


@dataclass(frozen=True, slots=True)
class GlobalHoldoutRecord:
    """Durable state of one canonical data/date holdout scope."""

    scope_id: str
    program_id: str
    market: str
    promotion_class: str
    data_snapshot_id: str
    start: str
    end: str
    state: str
    freeze_id: str | None
    evaluator_sha256: str | None
    consumed_at: str | None
    result_sha256: str | None
    result_path: str | None


def global_scope_id(
    *, program_id: str, market: str, promotion_class: str, start: str, end: str
) -> str:
    """Derive one economic-window identity independent of campaign/data copies."""

    return canonical_sha256(
        {
            "scope": "CANONICAL_US_EQUITY_RESEARCH_HOLDOUT",
            "program_id": program_id,
            "market": market,
            "promotion_class": promotion_class,
            "start": start,
            "end": end,
        }
    )


class GlobalHoldoutLedger:
    """Consume a holdout once even if a campaign is cloned or renamed."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS global_holdout_scopes (
                    scope_id TEXT PRIMARY KEY,
                    program_id TEXT NOT NULL,
                    market TEXT NOT NULL,
                    promotion_class TEXT NOT NULL,
                    data_snapshot_id TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('UNSPENT','CONSUMED','SEALED')),
                    freeze_id TEXT,
                    evaluator_sha256 TEXT,
                    consumed_at TEXT,
                    result_sha256 TEXT,
                    result_path TEXT,
                    UNIQUE(program_id, market, promotion_class, start_date, end_date)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def register(
        self,
        *,
        scope_id: str,
        program_id: str,
        market: str,
        promotion_class: str,
        data_snapshot_id: str,
        start: str,
        end: str,
    ) -> GlobalHoldoutRecord:
        """Register the canonical scope without spending it."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT OR IGNORE INTO global_holdout_scopes(
                    scope_id, program_id, market, promotion_class,
                    data_snapshot_id, start_date, end_date, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'UNSPENT')""",
                (
                    scope_id,
                    program_id,
                    market,
                    promotion_class,
                    data_snapshot_id,
                    start,
                    end,
                ),
            )
            row = connection.execute(
                "SELECT * FROM global_holdout_scopes WHERE scope_id=?", (scope_id,)
            ).fetchone()
            connection.commit()
        if row is None:
            raise RuntimeError(
                "holdout scope identity conflicts with an existing range"
            )
        record = self._record(row)
        if (
            record.program_id,
            record.market,
            record.promotion_class,
            record.data_snapshot_id,
            record.start,
            record.end,
        ) != (
            program_id,
            market,
            promotion_class,
            data_snapshot_id,
            start,
            end,
        ):
            raise RuntimeError("registered holdout scope metadata mismatch")
        return record

    def consume(
        self, *, scope_id: str, freeze_id: str, evaluator_sha256: str
    ) -> GlobalHoldoutRecord:
        """Atomically bind and consume the sole authorization before data access."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM global_holdout_scopes WHERE scope_id=?", (scope_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RuntimeError("global holdout scope was not registered")
            if row["state"] != "UNSPENT":
                connection.rollback()
                raise RuntimeError(
                    "global holdout is already consumed; only its sealed result may replay"
                )
            consumed_at = datetime.now(UTC).isoformat()
            cursor = connection.execute(
                """UPDATE global_holdout_scopes SET
                    state='CONSUMED', freeze_id=?, evaluator_sha256=?, consumed_at=?
                    WHERE scope_id=? AND state='UNSPENT'""",
                (freeze_id, evaluator_sha256, consumed_at, scope_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise RuntimeError("global holdout authorization race was lost")
            connection.commit()
        record = self.get(scope_id)
        if record is None:  # pragma: no cover - SQLite invariant
            raise RuntimeError("consumed holdout record disappeared")
        return record

    def seal(
        self,
        *,
        scope_id: str,
        freeze_id: str,
        result_sha256: str,
        result_path: str | Path,
    ) -> GlobalHoldoutRecord:
        """Attach one durable result identity to a consumed scope."""

        resolved = str(Path(result_path).resolve())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM global_holdout_scopes WHERE scope_id=?", (scope_id,)
            ).fetchone()
            if row is None or row["state"] != "CONSUMED":
                connection.rollback()
                raise RuntimeError("only a consumed global holdout can be sealed")
            if row["freeze_id"] != freeze_id:
                connection.rollback()
                raise RuntimeError("holdout freeze identity changed after consumption")
            connection.execute(
                """UPDATE global_holdout_scopes SET
                    state='SEALED', result_sha256=?, result_path=?
                    WHERE scope_id=? AND state='CONSUMED'""",
                (result_sha256, resolved, scope_id),
            )
            connection.commit()
        record = self.get(scope_id)
        if record is None:  # pragma: no cover - SQLite invariant
            raise RuntimeError("sealed holdout record disappeared")
        return record

    def get(self, scope_id: str) -> GlobalHoldoutRecord | None:
        """Read holdout state without authorizing data access."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM global_holdout_scopes WHERE scope_id=?", (scope_id,)
            ).fetchone()
        return None if row is None else self._record(row)

    @staticmethod
    def _record(row: sqlite3.Row) -> GlobalHoldoutRecord:
        return GlobalHoldoutRecord(
            scope_id=str(row["scope_id"]),
            program_id=str(row["program_id"]),
            market=str(row["market"]),
            promotion_class=str(row["promotion_class"]),
            data_snapshot_id=str(row["data_snapshot_id"]),
            start=str(row["start_date"]),
            end=str(row["end_date"]),
            state=str(row["state"]),
            freeze_id=str(row["freeze_id"]) if row["freeze_id"] else None,
            evaluator_sha256=(
                str(row["evaluator_sha256"]) if row["evaluator_sha256"] else None
            ),
            consumed_at=str(row["consumed_at"]) if row["consumed_at"] else None,
            result_sha256=(str(row["result_sha256"]) if row["result_sha256"] else None),
            result_path=str(row["result_path"]) if row["result_path"] else None,
        )
