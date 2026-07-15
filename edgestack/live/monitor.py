"""Pure pending-recommendation revalidation logic."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from edgestack.models import (
    EntryPlan,
    Quote,
    Recommendation,
    RecommendationState,
    TimingVerdict,
)


@dataclass(frozen=True, slots=True)
class RevalidationResult:
    """Result of rechecking an elapsed wait/trigger plan."""

    state: RecommendationState
    reason: str
    message: str
    entry_plan: EntryPlan


def revalidate_pending(
    recommendation: Recommendation,
    quote: Quote,
    *,
    scan_price: float,
    atr_value: float,
    regime_gate_passes: bool,
    now: datetime,
    max_quote_age: timedelta = timedelta(minutes=30),
    trigger_passes: bool | None = None,
) -> RevalidationResult:
    """Confirm, adjust, or cancel after a wait without assuming a current fill."""

    plan = recommendation.entry_plan
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("revalidation time must be timezone-aware")
    if max_quote_age <= timedelta(0):
        raise ValueError("max_quote_age must be positive")
    if scan_price <= 0 or atr_value <= 0:
        raise ValueError("scan price and ATR must be positive")
    if quote.asset != recommendation.asset:
        return _cancel(recommendation, plan, "quote belongs to another asset")
    if quote.price <= 0:
        return _cancel(recommendation, plan, "quote price is invalid")
    age = now - quote.provider_time
    if age < timedelta(0) or quote.received_at > now:
        return _cancel(recommendation, plan, "quote timestamp is in the future")
    if quote.halted:
        return _cancel(recommendation, plan, "instrument is halted")
    if age > max_quote_age:
        return _cancel(recommendation, plan, f"quote is stale by {age}")
    if not regime_gate_passes:
        return _cancel(recommendation, plan, "validated regime gate no longer passes")
    if plan.validity_end is not None and now > plan.validity_end:
        return _cancel(recommendation, plan, "base edge validity window expired")
    tolerance = 2.0 * atr_value
    difference = quote.price - scan_price
    if abs(difference) > tolerance:
        return _cancel(recommendation, plan, "price moved beyond +/-2xATR tolerance")
    if now < plan.earliest_execution:
        return _waiting(
            recommendation,
            plan,
            f"earliest execution is {plan.earliest_execution.isoformat()}",
        )
    expired = plan.expiry_at is not None and now >= plan.expiry_at
    if plan.verdict is TimingVerdict.WAIT_FOR_TRIGGER:
        if expired:
            if plan.expiry_action != "enter_at_next_eligible_close":
                return _cancel(recommendation, plan, "entry trigger expired")
        elif trigger_passes is not True:
            return _waiting(recommendation, plan, "entry trigger has not fired")
    elif expired and plan.expiry_action is None:
        return _cancel(recommendation, plan, "entry plan expired")
    if (
        plan.limit_price is not None
        and abs(quote.price - plan.limit_price) > 0.5 * atr_value
    ):
        adjusted = replace(
            plan, limit_price=quote.price, data_timestamp=quote.provider_time
        )
        return RevalidationResult(
            RecommendationState.UPDATED,
            "price remains valid but limit requires adjustment",
            f"Plan ADJUSTED: {recommendation.asset.symbol} new indicative limit {quote.price:.2f}.",
            adjusted,
        )
    return RevalidationResult(
        RecommendationState.CONFIRMED,
        "all frozen entry conditions still pass",
        f"Recommendation HOLDS: {recommendation.direction.value} {recommendation.asset.symbol} at the plan's next eligible execution.",
        plan,
    )


def _cancel(
    recommendation: Recommendation, plan: EntryPlan, reason: str
) -> RevalidationResult:
    return RevalidationResult(
        RecommendationState.CANCELLED,
        reason,
        f"Recommendation CANCELLED: {recommendation.asset.symbol}: {reason}. Do nothing.",
        plan,
    )


def _waiting(
    recommendation: Recommendation, plan: EntryPlan, reason: str
) -> RevalidationResult:
    return RevalidationResult(
        RecommendationState.WAITING,
        reason,
        f"Recommendation remains WAITING: {recommendation.asset.symbol}: {reason}.",
        plan,
    )
