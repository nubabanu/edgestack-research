"""SQLite campaign catalog, gate ledger, and holdout audit."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edgestack.models import GateResult, GateStatus


@dataclass(frozen=True, slots=True)
class HoldoutAccessRecord:
    """Durable state of the campaign's single holdout authorization."""

    campaign_id: str
    freeze_id: str
    accessed_at: datetime
    result_sha256: str | None


class Catalog:
    """Crash-safe metadata catalog using WAL mode."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        """Return a configured connection."""

        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS gates (
                    campaign_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    status TEXT NOT NULL,
                    checked_at TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    PRIMARY KEY (campaign_id, phase)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    campaign_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (campaign_id, kind, sha256)
                );
                CREATE TABLE IF NOT EXISTS holdout_access (
                    campaign_id TEXT PRIMARY KEY,
                    freeze_id TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    result_sha256 TEXT
                );
                """
            )

    def create_campaign(self, campaign_id: str, manifest: dict[str, object]) -> None:
        """Register a campaign id exactly once."""

        with self.connect() as connection:
            connection.execute(
                "INSERT INTO campaigns VALUES (?, ?, ?)",
                (
                    campaign_id,
                    json.dumps(manifest, sort_keys=True, default=str),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def campaign(self, campaign_id: str) -> dict[str, Any] | None:
        """Load a registered campaign manifest."""

        with self.connect() as connection:
            row = connection.execute(
                "SELECT manifest_json FROM campaigns WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["manifest_json"])
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid manifest for campaign {campaign_id}")
        return payload

    def record_artifact(
        self, campaign_id: str, kind: str, sha256: str, path: str | Path
    ) -> None:
        """Register one immutable artifact identity."""

        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO artifacts VALUES (?, ?, ?, ?, ?)",
                (
                    campaign_id,
                    kind,
                    sha256,
                    str(Path(path).resolve()),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def artifacts(self, campaign_id: str) -> tuple[dict[str, str], ...]:
        """List registered campaign artifacts in deterministic order."""

        with self.connect() as connection:
            rows = connection.execute(
                """SELECT kind, sha256, path, created_at FROM artifacts
                WHERE campaign_id=? ORDER BY kind, sha256""",
                (campaign_id,),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def artifact_registered(self, campaign_id: str, kind: str, sha256: str) -> bool:
        """Return whether an exact artifact identity is in the immutable ledger."""

        with self.connect() as connection:
            row = connection.execute(
                """SELECT 1 FROM artifacts
                WHERE campaign_id=? AND kind=? AND sha256=? LIMIT 1""",
                (campaign_id, kind, sha256),
            ).fetchone()
        return row is not None

    def record_gate(self, result: GateResult) -> None:
        """Insert or replace the authoritative result for a phase."""

        payload = asdict(result)
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO gates VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(campaign_id, phase) DO UPDATE SET
                status=excluded.status, checked_at=excluded.checked_at,
                summary=excluded.summary, evidence_json=excluded.evidence_json""",
                (
                    result.campaign_id,
                    result.phase,
                    result.status.value,
                    result.checked_at.isoformat(),
                    result.summary,
                    json.dumps(payload["evidence"], sort_keys=True, default=str),
                ),
            )

    def gate(self, campaign_id: str, phase: str) -> GateResult | None:
        """Load a gate result."""

        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM gates WHERE campaign_id=? AND phase=?",
                (campaign_id, phase),
            ).fetchone()
        if row is None:
            return None
        return GateResult(
            campaign_id=row["campaign_id"],
            phase=row["phase"],
            status=GateStatus(row["status"]),
            checked_at=datetime.fromisoformat(row["checked_at"]),
            summary=row["summary"],
            evidence=json.loads(row["evidence_json"]),
        )

    def require_passed(self, campaign_id: str, phases: Iterable[str]) -> None:
        """Raise unless every prerequisite phase is promoted."""

        failed = []
        for phase in phases:
            result = self.gate(campaign_id, phase)
            if result is None or result.status not in {
                GateStatus.PASS,
                GateStatus.NOT_APPLICABLE,
            }:
                failed.append(phase)
        if failed:
            raise RuntimeError(
                f"campaign prerequisites not passed: {', '.join(failed)}"
            )

    def begin_holdout_access(self, campaign_id: str, freeze_id: str) -> None:
        """Atomically consume the campaign's one holdout authorization."""

        try:
            with self.connect() as connection:
                connection.execute(
                    "INSERT INTO holdout_access(campaign_id, freeze_id, accessed_at) VALUES (?, ?, ?)",
                    (campaign_id, freeze_id, datetime.now(UTC).isoformat()),
                )
        except sqlite3.IntegrityError as exc:
            raise RuntimeError(
                "holdout has already been accessed for this campaign"
            ) from exc

    def holdout_access(self, campaign_id: str) -> HoldoutAccessRecord | None:
        """Load the durable single-use holdout state without changing it."""

        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM holdout_access WHERE campaign_id=?", (campaign_id,)
            ).fetchone()
        if row is None:
            return None
        return HoldoutAccessRecord(
            campaign_id=str(row["campaign_id"]),
            freeze_id=str(row["freeze_id"]),
            accessed_at=datetime.fromisoformat(str(row["accessed_at"])),
            result_sha256=(
                str(row["result_sha256"]) if row["result_sha256"] is not None else None
            ),
        )

    def complete_holdout_access(self, campaign_id: str, result_sha256: str) -> None:
        """Attach one immutable result identity to a consumed authorization."""

        with self.connect() as connection:
            row = connection.execute(
                "SELECT result_sha256 FROM holdout_access WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("holdout access was not begun")
            current = row["result_sha256"]
            if current is not None and str(current) != result_sha256:
                raise RuntimeError("holdout result identity is already sealed")
            if current is None:
                connection.execute(
                    "UPDATE holdout_access SET result_sha256=? WHERE campaign_id=?",
                    (result_sha256, campaign_id),
                )
