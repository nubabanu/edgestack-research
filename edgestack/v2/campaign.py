"""Loss-aware V2 namespace and forward-only freeze governance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edgestack.v2.gates import CapabilityReport, evaluate_capabilities
from edgestack.v2.research import declared_trials

NAMESPACE = "loss-aware-v2"


@dataclass(frozen=True, slots=True)
class V2DiagnosticManifest:
    """Historical diagnostics declaration that cannot consume a holdout."""

    namespace: str
    campaign_id: str
    created_at: datetime
    trial_count: int
    config_sha256: str
    v1_holdout_access: str
    historical_status: str
    promotion_requirement: str
    capabilities: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class V2ForwardFreeze:
    """Frozen paper definition whose evidence clock starts after creation."""

    namespace: str
    campaign_id: str
    freeze_id: str
    frozen_at: datetime
    forward_observations_strictly_after: datetime
    model_sha256: str
    data_contract_sha256: str
    config_sha256: str
    v1_holdout_access: str
    promotion_basis: str


def create_free_only_diagnostic(
    artifact_root: str | Path,
    *,
    campaign_id: str,
    config_path: str | Path,
) -> Path:
    """Persist free-only gate results without reading any V1 holdout artifact."""

    if not campaign_id or Path(campaign_id).name != campaign_id:
        raise ValueError("invalid V2 campaign identifier")
    config_bytes = Path(config_path).read_bytes()
    report = evaluate_capabilities()
    manifest = V2DiagnosticManifest(
        namespace=NAMESPACE,
        campaign_id=campaign_id,
        created_at=datetime.now(UTC),
        trial_count=len(declared_trials()),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        v1_holdout_access="FORBIDDEN_NOT_READ",
        historical_status="DIAGNOSTIC_ONLY",
        promotion_requirement="NEW_FORWARD_PAPER_OBSERVATIONS",
        capabilities=_capabilities(report),
    )
    target = Path(artifact_root) / NAMESPACE / campaign_id
    target.mkdir(parents=True, exist_ok=False)
    output = target / "diagnostic_manifest.json"
    output.write_text(
        json.dumps(asdict(manifest), sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return output


def freeze_forward_model(
    artifact_root: str | Path,
    *,
    campaign_id: str,
    model: dict[str, Any],
    config_sha256: str,
    data_contract_sha256: str,
    capabilities: CapabilityReport,
) -> Path:
    """Freeze a qualifying definition; historical rows can never promote it."""

    if not capabilities.promotable:
        raise RuntimeError("DATA_UNAVAILABLE: every entitled V2 data gate must pass")
    for value in (config_sha256, data_contract_sha256):
        if len(value) != 64:
            raise ValueError("freeze inputs must use SHA-256 identities")
    canonical = json.dumps(model, sort_keys=True, separators=(",", ":"))
    model_hash = hashlib.sha256(canonical.encode()).hexdigest()
    frozen_at = datetime.now(UTC)
    freeze_id = f"v2-{hashlib.sha256(f'{campaign_id}:{model_hash}:{frozen_at.isoformat()}'.encode()).hexdigest()[:20]}"
    freeze = V2ForwardFreeze(
        namespace=NAMESPACE,
        campaign_id=campaign_id,
        freeze_id=freeze_id,
        frozen_at=frozen_at,
        forward_observations_strictly_after=frozen_at,
        model_sha256=model_hash,
        data_contract_sha256=data_contract_sha256,
        config_sha256=config_sha256,
        v1_holdout_access="FORBIDDEN_NOT_READ",
        promotion_basis="NEW_FORWARD_PAPER_OBSERVATIONS_ONLY",
    )
    target = Path(artifact_root) / NAMESPACE / campaign_id
    target.mkdir(parents=True, exist_ok=True)
    output = target / "forward_freeze.json"
    if output.exists():
        raise RuntimeError("V2 model is already frozen; create a new version")
    output.write_text(
        json.dumps(
            {"freeze": asdict(freeze), "model": model},
            sort_keys=True,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def _capabilities(report: CapabilityReport) -> dict[str, dict[str, Any]]:
    return {
        item.name: {
            "status": item.status.value,
            "reason": item.reason,
            "observations": item.observations,
        }
        for item in (
            report.pit_membership,
            report.estimate_vintages,
            report.auction_execution,
        )
    }
