"""Fail-closed conversion of sealed EdgeStack artifacts into mobile snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, cast

from edgestack.mobile.models import (
    AlignmentLayer,
    AnchorLeg,
    TomPlan,
    ApiMeta,
    AuditItem,
    EntryInstruction,
    HoldoutEvidence,
    HorizonPlan,
    LossAwareV2Summary,
    MobileDataGate,
    MobileLossMetrics,
    MobileRecommendation,
    MobileSnapshot,
    PortfolioSummary,
    SniperPolicy,
    TailwindDay,
    TimingAdvisor,
    TimingAnchors,
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
        advisors = _timing_advisors(self.artifact_root)
        return self._normalize(
            campaign.name,
            holdout_path,
            holdout,
            signal_path,
            signal,
            timing=advisors[0],
            timing_symbols=advisors,
            tom_plan=_tom_plan(self.artifact_root),
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
        *,
        timing: TimingAdvisor,
        timing_symbols: tuple[TimingAdvisor, ...] = (),
        tom_plan: "TomPlan | None" = None,
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
            horizons=_horizon_plans(recommendations, entry, exit_),
            sniper=_sniper_policy(recommendations),
            loss_aware_v2=_free_only_v2(),
            timing=timing,
            timing_symbols=timing_symbols,
            tom_plan=tom_plan,
        )


def _horizon_plans(
    recommendations: tuple[MobileRecommendation, ...],
    entry: dict[str, Any],
    exit_: dict[str, Any],
) -> tuple[HorizonPlan, ...]:
    symbols = tuple(item.symbol for item in recommendations)
    return (
        HorizonPlan(
            horizon="WEEK",
            status="CONDITIONAL_PAPER_SIGNAL",
            title="Five-session reversal basket",
            holding_period="Five earned close-to-close intervals",
            entry_rule=(
                f"Revalidate at 15:30-15:45 ET and submit the complete basket "
                f"as MOC for {entry['session']}."
            ),
            review_rule="Refresh after every completed daily bar; confidence is ordinal, not a probability.",
            exit_rule=f"Exit the complete basket MOC on {exit_['session']} unless a risk cancellation occurs.",
            recommendation_scope="BASKET",
            symbols=symbols,
            evidence="Promoted five-day model with a sealed PASS holdout; individual ranks are not standalone forecasts.",
            invalidation=tuple(map(str, entry["cancel_if"])),
            unlock_requirement="Already unlocked only for the complete frozen basket and exact timing contract.",
        ),
        HorizonPlan(
            horizon="MONTH",
            status="DATA_UNAVAILABLE",
            title="No validated monthly stock recommendation",
            holding_period="Approximately 21 NYSE sessions",
            entry_rule="Do not stretch the five-day reversal signal into a one-month trade.",
            review_rule="Wait for an independently frozen monthly model and untouched forward evidence.",
            exit_rule="No monthly exit is authorized because no monthly recommendation is emitted.",
            recommendation_scope="NONE",
            evidence="The promoted campaign did not validate a standalone 21-session stock-selection model.",
            invalidation=(
                "Any monthly ticker inferred from the weekly ranking is invalid.",
            ),
            unlock_requirement="Requires preregistration, costs, OOS validation, confirmation, freeze, and a new holdout.",
        ),
        HorizonPlan(
            horizon="YEAR",
            status="DATA_UNAVAILABLE",
            title="No validated one-year stock recommendation",
            holding_period="Approximately 252 NYSE sessions",
            entry_rule="Do not use a five-day oversold move as a one-year investment thesis.",
            review_rule="Require point-in-time fundamentals and a separately validated long-horizon model.",
            exit_rule="No annual exit is authorized because no annual recommendation is emitted.",
            recommendation_scope="NONE",
            evidence="EdgeStack has no frozen promoted 252-session single-stock model in this campaign.",
            invalidation=(
                "Any annual ticker inferred from the weekly ranking is invalid.",
            ),
            unlock_requirement="Requires a new causal annual study, full cost/OOS gauntlet, freeze, and future holdout.",
        ),
    )


def _sniper_policy(
    recommendations: tuple[MobileRecommendation, ...],
) -> SniperPolicy:
    return SniperPolicy(
        status="NO_TRADE",
        candidate_symbols=tuple(item.symbol for item in recommendations),
        max_name_weight=0.05,
        max_gross_exposure=0.25,
        max_planned_loss_per_name_usd=100.0,
        max_planned_basket_loss_usd=500.0,
        execution_window="Revalidate 15:30-15:45 ET; complete-basket MOC only.",
        alignments=(
            AlignmentLayer(
                horizon="YEAR",
                status="UNVALIDATED",
                evidence="No promoted 252-session regime or stock-selection model exists.",
            ),
            AlignmentLayer(
                horizon="MONTH",
                status="UNVALIDATED",
                evidence="No promoted 21-session model exists.",
            ),
            AlignmentLayer(
                horizon="WEEK",
                status="PASS",
                evidence="The five-session reversal basket has a sealed PASS holdout.",
            ),
            AlignmentLayer(
                horizon="DAY",
                status="PENDING",
                evidence="Fresh quotes, membership, news, halts, and MOC availability require pre-close revalidation.",
            ),
        ),
        hard_vetoes=(
            "YEAR_ALIGNMENT_UNVALIDATED",
            "MONTH_ALIGNMENT_UNVALIDATED",
            "DAY_REVALIDATION_PENDING",
            "HIGH_EVENT_RISK_IN_WEEKLY_BASKET",
        ),
        release_condition="Remain NO TRADE until all four layers are independently validated and pass on the same causal snapshot.",
        stop_warning="Planned loss is a sizing budget, not a guarantee. Stops can gap, slip, whipsaw, or fail to execute at the trigger price.",
        validation_status="RISK_OVERLAY_NOT_VALIDATED_ALPHA",
    )


def _free_only_v2() -> LossAwareV2Summary:
    """Expose missing entitlements as NO TRADE, never inferred evidence."""

    return LossAwareV2Summary(
        evidence_status="FORWARD_REQUIRED",
        selected_horizon="NONE",
        selected_leverage=1.0,
        loss_metrics=MobileLossMetrics(status="DATA_UNAVAILABLE"),
        data_gates=(
            MobileDataGate(
                name="PIT_MEMBERSHIP",
                status="DATA_UNAVAILABLE",
                reason="Wikipedia reconstruction is PIT_APPROXIMATION, not genuine point-in-time membership.",
            ),
            MobileDataGate(
                name="ESTIMATE_VINTAGES",
                status="DATA_UNAVAILABLE",
                reason="Historical consensus vintages are not configured.",
            ),
            MobileDataGate(
                name="AUCTION_EXECUTION",
                status="DATA_UNAVAILABLE",
                reason="NBBO, trades, imbalances, and official auction prints are not configured.",
            ),
        ),
        enabled_event_vetoes=(),
        timing="NO TRADE; monthly/yearly timing unlocks only after forward evidence and all gates pass.",
    )


_MOBILE_CALENDAR_ROWS = 21


def _timing_advisors(artifact_root: Path) -> tuple[TimingAdvisor, ...]:
    """Load every published advisor calendar; SPY (or first) leads as primary.

    The nightly job writes one ``tailwind-calendar-<SYMBOL>.json`` per symbol
    plus the legacy single-file name; a missing or malformed set degrades to
    one explicit DATA_UNAVAILABLE advisor, never an error.
    """

    advisor_dir = artifact_root / "advisor"
    paths = sorted(advisor_dir.glob("tailwind-calendar-*.json"))
    if not paths:
        legacy = advisor_dir / "tailwind-calendar.json"
        paths = [legacy] if legacy.is_file() else []
    advisors = [
        advisor
        for advisor in (_timing_advisor(path) for path in paths)
        if advisor.status == "AVAILABLE"
    ]
    if not advisors:
        return (_timing_advisor(advisor_dir / "tailwind-calendar.json"),)
    advisors.sort(key=lambda advisor: (advisor.symbol != "SPY", advisor.symbol))
    return tuple(advisors)


def _tom_plan(artifact_root: Path) -> TomPlan | None:
    """Compute the validated turn-of-month plan, absent unless gates PASS.

    ``next_trade`` itself refuses without PASS ``edge_preholdout`` and
    ``edge_holdout`` catalog gates, so any failure — missing config, failed
    gates, calendar issues — degrades to an absent section, never an error.
    """

    try:
        from edgestack.edges.turn_of_month import next_trade

        root = artifact_root.parent
        payload = next_trade("configs/spy-tom-edge-v1.yaml", root=root)
        return TomPlan(
            state=str(payload["state"]),
            symbol=str(payload["symbol"]),
            direction=str(payload["direction"]),
            entry_session=str(payload["entry_session"]),
            entry_order=str(payload["entry_order"]),
            first_exposure_session=str(payload["first_exposure_session"]),
            exit_session=str(payload["exit_session"]),
            exit_order=str(payload["exit_order"]),
            maximum_allocation_usd=float(payload["maximum_allocation_usd"]),
            sizing=str(payload["sizing"]),
            stop=str(payload["stop"]),
        )
    except Exception:
        return None


def _timing_advisor(path: Path) -> TimingAdvisor:
    """Parse one tailwind-calendar artifact, failing to UNAVAILABLE."""

    unavailable = TimingAdvisor(
        status="DATA_UNAVAILABLE",
        symbol="NONE",
        as_of_session="",
        policy=(
            "no advisor calendar artifact; generate one with "
            "'edgestack tailwind-calendar --symbol SPY --output "
            "artifacts/advisor/tailwind-calendar.json'"
        ),
    )
    if not path.is_file():
        return unavailable
    try:
        payload = _mapping(path)
        anchors_raw = cast(dict[str, Any], payload.get("anchors", {}))
        anchors: TimingAnchors | None = None
        if anchors_raw.get("status") == "TWO_ANCHORS_ONLY":
            legs = cast(dict[str, Any], anchors_raw.get("legs", {}))

            def leg(name: str) -> AnchorLeg:
                data = cast(dict[str, Any], legs.get(name, {}))
                mean = data.get("mean_daily")
                return AnchorLeg(
                    n=int(data.get("n", 0)),
                    mean_daily_bp=(
                        float(mean) * 10_000.0 if mean is not None else None
                    ),
                    hit_rate=(
                        float(data["hit_rate"])
                        if data.get("hit_rate") is not None
                        else None
                    ),
                )

            anchors = TimingAnchors(
                status="TWO_ANCHORS_ONLY",
                best_buy_anchor=str(anchors_raw["best_buy_anchor"]),
                matching_sell_anchor=str(anchors_raw["matching_sell_anchor"]),
                overnight=leg("overnight"),
                intraday=leg("intraday"),
                finer_granularity=str(anchors_raw["fifteen_minute_calendar"]),
            )
        rows = tuple(
            TailwindDay(
                session=str(row["session"]),
                weekday=str(row["weekday"]),
                win_score=int(row["win_score_0_100"]),
                expected_daily_bp=float(row["expected_daily_bp"]),
                conditions=tuple(
                    str(item) for item in row["active_calendar_conditions"]
                ),
            )
            for row in cast(
                list[dict[str, Any]], payload.get("calendar", [])
            )[:_MOBILE_CALENDAR_ROWS]
        )
        if not rows:
            return unavailable
        return TimingAdvisor(
            status="AVAILABLE",
            symbol=str(payload["symbol"]),
            as_of_session=str(payload["as_of_session"]),
            policy=str(payload["policy"]),
            anchors=anchors,
            calendar=rows,
        )
    except (KeyError, TypeError, ValueError, SnapshotUnavailableError):
        return unavailable


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
