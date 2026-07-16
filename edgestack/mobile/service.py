"""Fail-closed conversion of sealed EdgeStack artifacts into mobile snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, cast

from edgestack.mobile.models import (
    ApiMeta,
    AuditItem,
    EntryInstruction,
    HoldoutEvidence,
    MobileRecommendation,
    MobileSnapshot,
    PortfolioSummary,
)
from edgestack.provenance import sha256_file


class SnapshotUnavailableError(RuntimeError):
    """Raised when sealed artifacts cannot produce an honest mobile payload."""


class MobileSnapshotService:
    """Build one atomic, replay-only companion snapshot."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        campaign_id: str | None = None,
        demo: bool = False,
    ) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self.campaign_id = campaign_id
        self.demo = demo

    def load(self) -> MobileSnapshot:
        """Load packaged demo data or verify and normalize sealed artifacts."""

        if self.demo:
            payload = json.loads(
                resources.files("edgestack.mobile")
                .joinpath("demo_snapshot.json")
                .read_text(encoding="utf-8")
            )
            return MobileSnapshot.model_validate(payload)
        campaign = self._campaign_directory()
        holdout_path = campaign / "holdout" / "result.json"
        if not holdout_path.is_file():
            raise SnapshotUnavailableError("campaign has no sealed holdout result")
        holdout = _mapping(holdout_path)
        if holdout.get("second_evaluation") != "FORBIDDEN_REPLAY_ONLY":
            raise SnapshotUnavailableError("holdout is not marked replay-only")
        if holdout.get("status") != "PASS" or holdout.get("holdout_pass") is not True:
            raise SnapshotUnavailableError("campaign holdout did not pass")
        signal_path = self._latest_signal(campaign)
        signal = _mapping(signal_path)
        if signal.get("bias_tier") not in {"SURVIVORSHIP_BIASED", "POINT_IN_TIME"}:
            raise SnapshotUnavailableError("signal lacks an explicit bias tier")
        return self._normalize(
            campaign.name, holdout_path, holdout, signal_path, signal
        )

    def _campaign_directory(self) -> Path:
        root = self.artifact_root / "campaigns"
        if self.campaign_id:
            if Path(self.campaign_id).name != self.campaign_id:
                raise SnapshotUnavailableError("invalid campaign identifier")
            campaign = root / self.campaign_id
            if not campaign.is_dir():
                raise SnapshotUnavailableError("campaign does not exist")
            return campaign
        candidates = sorted(
            (
                item
                for item in root.glob("*")
                if (item / "holdout" / "result.json").is_file()
                and (item / "live").is_dir()
            ),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        if not candidates:
            raise SnapshotUnavailableError("no sealed mobile campaign is available")
        return candidates[0]

    @staticmethod
    def _latest_signal(campaign: Path) -> Path:
        candidates = sorted((campaign / "live").glob("*.json"))
        if not candidates:
            raise SnapshotUnavailableError("campaign has no mobile paper signal")
        return candidates[-1]

    @staticmethod
    def _normalize(
        campaign_id: str,
        holdout_path: Path,
        holdout: dict[str, Any],
        signal_path: Path,
        signal: dict[str, Any],
    ) -> MobileSnapshot:
        candidates = cast(list[dict[str, Any]], signal.get("candidates", []))
        if not candidates:
            raise SnapshotUnavailableError("paper signal contains no candidates")
        entry = cast(dict[str, Any], signal.get("entry", {}))
        exit_ = cast(dict[str, Any], signal.get("exit", {}))
        portfolio = cast(dict[str, Any], signal.get("portfolio", {}))
        generated = _parse_datetime(signal.get("generated_at_utc"))
        recommendations = tuple(
            MobileRecommendation(
                recommendation_id=str(item["recommendation_id"]),
                rank=int(item["rank"]),
                symbol=str(item["symbol"]),
                direction=str(item["direction"]),
                confidence_ordinal=int(item["confidence_ordinal_not_probability"]),
                signal_close_usd=float(item["signal_close_usd"]),
                trailing_return=float(item["trailing_5_session_return"]),
                suggested_shares=int(item["risk_capped_reference_shares"]),
                reference_stop_usd=float(item["two_atr_reference_price_usd"]),
                event_risk=str(item["event_risk"]),
            )
            for item in candidates
        )
        return MobileSnapshot(
            meta=ApiMeta(
                generated_at=generated,
                market_as_of=str(signal["market_as_of"]),
                source=str(cast(dict[str, Any], signal["data"])["source"]),
                mode="SEALED",
                stale=_is_stale(generated),
            ),
            campaign_id=campaign_id,
            model_name=str(signal["strategy"]),
            model_status="PROMOTED",
            bias_tier=str(signal["bias_tier"]),
            watermark=str(signal["bias_tier"]),
            basket_rule=str(signal["interpretation"]),
            instruction=EntryInstruction(
                entry_session=str(entry["session"]),
                entry_order="MOC",
                submit_by_et=str(entry["planned_submission_time"]),
                exit_session=str(exit_["session"]),
                exit_order="MOC",
                no_chase=str(entry["no_chase"]),
                cancel_if=tuple(map(str, entry["cancel_if"])),
            ),
            portfolio=PortfolioSummary(
                paper_capital_usd=float(portfolio["paper_capital_usd"]),
                target_gross=float(portfolio["tested_new_account_gross_target"]),
                maximum_name_weight=float(portfolio["tested_maximum_weight_per_name"]),
                risk_budget_per_name_usd=float(
                    portfolio["paper_risk_budget_per_name_usd"]
                ),
                shorts_enabled=bool(signal.get("shorts")),
            ),
            recommendations=recommendations,
            holdout=HoldoutEvidence(
                status="PASS",
                start=str(holdout["holdout_start"]),
                end=str(holdout["holdout_end"]),
                observations=int(holdout["observations"]),
                expected_sessions=int(holdout["expected_sessions"]),
                net_mean=float(holdout["net_mean"]),
                benchmark_excess_mean=float(holdout["benchmark_excess_mean"]),
                terminal_wealth=float(holdout["terminal_net_wealth"]),
                benchmark_wealth=float(holdout["terminal_benchmark_wealth"]),
                freeze_id=str(holdout["freeze_id"]),
                result_sha256=sha256_file(holdout_path),
            ),
            audit=(
                AuditItem(
                    occurred_at=generated,
                    event_type="SIGNAL_FROZEN",
                    message=f"Paper signal {sha256_file(signal_path)[:12]} created.",
                ),
                AuditItem(
                    occurred_at=generated,
                    event_type="HOLDOUT_REPLAY",
                    message="Sealed holdout evidence was replayed; no reevaluation occurred.",
                ),
            ),
        )


def _mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SnapshotUnavailableError(f"invalid artifact: {path.name}") from error
    if not isinstance(payload, dict):
        raise SnapshotUnavailableError(f"artifact is not an object: {path.name}")
    return cast(dict[str, Any], payload)


def _parse_datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise SnapshotUnavailableError("artifact timestamp must be timezone-aware")
    return parsed


def _is_stale(generated: datetime) -> bool:
    return (datetime.now(UTC) - generated.astimezone(UTC)).total_seconds() > 36 * 3600


def stable_etag(snapshot: MobileSnapshot) -> str:
    """Return a content identity independent of JSON whitespace."""

    canonical = snapshot.model_dump_json(exclude_none=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
