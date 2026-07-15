"""Phase dependency enforcement."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from edgestack.models import GateResult, GateStatus
from edgestack.storage.catalog import Catalog

PHASE_ORDER = (
    "data",
    "replication",
    "discovery",
    "validation",
    "report",
    "score",
    "holdout",
    "live",
)


class Gatekeeper:
    """Persist gates and prevent promotion past a failed predecessor."""

    def __init__(self, catalog: Catalog, campaign_id: str) -> None:
        self.catalog = catalog
        self.campaign_id = campaign_id

    def require_previous(self, phase: str) -> None:
        """Require all phases preceding ``phase`` to have passed."""

        try:
            index = PHASE_ORDER.index(phase)
        except ValueError as exc:
            raise ValueError(f"unknown phase: {phase}") from exc
        self.catalog.require_passed(self.campaign_id, PHASE_ORDER[:index])

    def record(
        self,
        phase: str,
        passed: bool,
        summary: str,
        evidence: Mapping[str, Any] | None = None,
        *,
        blocked: bool = False,
    ) -> GateResult:
        """Record an immutable-in-time gate decision."""

        status = (
            GateStatus.BLOCKED
            if blocked
            else (GateStatus.PASS if passed else GateStatus.FAIL)
        )
        result = GateResult(
            campaign_id=self.campaign_id,
            phase=phase,
            status=status,
            checked_at=datetime.now(UTC),
            summary=summary,
            evidence=evidence or {},
        )
        self.catalog.record_gate(result)
        return result
