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


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Durable state for one uniquely identified model/rule experiment."""

    study_id: str
    trial_id: str
    status: str
    device: str
    spec: dict[str, Any]
    metrics: dict[str, Any] | None
    created_at: datetime
    completed_at: datetime | None
    error: str | None


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
                CREATE TABLE IF NOT EXISTS research_experiments (
                    trial_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    device TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    metrics_json TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS research_experiments_study
                    ON research_experiments(study_id, status, trial_id);
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

    def smoke_override_gates(
        self, campaign_id: str | None = None
    ) -> tuple[GateResult, ...]:
        """List gates that passed only through the smoke-profile override.

        A gate row qualifies when its persisted evidence records
        ``smoke_mechanical_override`` as true: the empirical check failed and
        only the synthetic non-promotable profile let the phase proceed.
        """

        query = "SELECT campaign_id, phase FROM gates"
        parameters: tuple[str, ...] = ()
        if campaign_id is not None:
            query += " WHERE campaign_id=?"
            parameters = (campaign_id,)
        query += " ORDER BY campaign_id, phase"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        overridden = []
        for row in rows:
            result = self.gate(row["campaign_id"], row["phase"])
            if result is not None and result.evidence.get("smoke_mechanical_override"):
                overridden.append(result)
        return tuple(overridden)

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

    def claim_experiment(
        self,
        study_id: str,
        trial_id: str,
        spec: dict[str, Any],
        *,
        device: str,
    ) -> bool:
        """Atomically claim a unique experiment, preventing duplicate GPU work."""

        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO research_experiments(
                    trial_id, study_id, status, device, spec_json, created_at
                ) VALUES (?, ?, 'RUNNING', ?, ?, ?)""",
                (
                    trial_id,
                    study_id,
                    device,
                    json.dumps(
                        spec, sort_keys=True, separators=(",", ":"), default=str
                    ),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return cursor.rowcount == 1

    def complete_experiment(self, trial_id: str, metrics: dict[str, Any]) -> None:
        """Seal one experiment's metrics; completed trials are immutable."""

        encoded = json.dumps(
            metrics, sort_keys=True, separators=(",", ":"), default=str
        )
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status, metrics_json FROM research_experiments WHERE trial_id=?",
                (trial_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("experiment was not claimed")
            if row["status"] == "COMPLETE":
                if row["metrics_json"] != encoded:
                    raise RuntimeError("completed experiment metrics are immutable")
                return
            if row["status"] != "RUNNING":
                raise RuntimeError("only a running experiment can be completed")
            connection.execute(
                """UPDATE research_experiments
                SET status='COMPLETE', metrics_json=?, completed_at=?, error=NULL
                WHERE trial_id=?""",
                (encoded, datetime.now(UTC).isoformat(), trial_id),
            )

    def fail_experiment(self, trial_id: str, error: str) -> None:
        """Persist a failed attempt without making it silently retry as new work."""

        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE research_experiments
                SET status='FAILED', completed_at=?, error=?
                WHERE trial_id=? AND status='RUNNING'""",
                (datetime.now(UTC).isoformat(), error, trial_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("only a running experiment can fail")

    def experiments(self, study_id: str) -> tuple[ExperimentRecord, ...]:
        """List a study's trials in stable identity order."""

        with self.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM research_experiments
                WHERE study_id=? ORDER BY trial_id""",
                (study_id,),
            ).fetchall()
        return tuple(
            ExperimentRecord(
                study_id=str(row["study_id"]),
                trial_id=str(row["trial_id"]),
                status=str(row["status"]),
                device=str(row["device"]),
                spec=json.loads(str(row["spec_json"])),
                metrics=(
                    json.loads(str(row["metrics_json"]))
                    if row["metrics_json"] is not None
                    else None
                ),
                created_at=datetime.fromisoformat(str(row["created_at"])),
                completed_at=(
                    datetime.fromisoformat(str(row["completed_at"]))
                    if row["completed_at"] is not None
                    else None
                ),
                error=str(row["error"]) if row["error"] is not None else None,
            )
            for row in rows
        )
