"""Daily final-WORKS scanner and recommendation construction."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from edgestack.models import (
    AssetKey,
    Direction,
    EntryPlan,
    Recommendation,
    TimingVerdict,
)
from edgestack.report.ranker import RankedRecommendations, rank_recommendations


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """Frozen-model candidate before recommendation persistence."""

    asset: AssetKey
    direction: Direction
    confidence: int
    expected_net_return: float
    expected_return_ci: tuple[float, float]
    holding_period: int
    entry_plan: EntryPlan
    driving_edges: tuple[str, ...]
    all_edges_final_works: bool
    composite_promoted: bool
    borrow_verified: bool = False


def scan(
    candidates: Iterable[ScoredCandidate],
    *,
    as_of: datetime | None = None,
    top_n: int = 5,
    minimum_confidence: int = 60,
) -> RankedRecommendations:
    """Return honest top-N paper candidates from a promoted frozen model."""

    created_at = as_of or datetime.now(UTC)
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("scan as_of must be timezone-aware")
    recommendations: list[Recommendation] = []
    identities: set[str] = set()
    for item in candidates:
        if not 0 <= item.confidence <= 100:
            raise ValueError("candidate confidence must be in [0, 100]")
        if item.holding_period <= 0:
            raise ValueError("candidate holding period must be positive")
        if item.entry_plan.direction is not item.direction:
            raise ValueError("candidate and entry-plan directions disagree")
        identity = (
            f"{created_at.date().isoformat()}:{item.asset.exchange}:"
            f"{item.asset.asset_type}:{item.asset.symbol}:"
            f"{item.direction.value}:{item.holding_period}"
        )
        if identity in identities:
            raise ValueError(f"duplicate daily candidate identity: {identity}")
        identities.add(identity)
        recommendation_id = "rec-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        plan = item.entry_plan
        if not item.all_edges_final_works or not item.composite_promoted:
            reasons = []
            if not item.all_edges_final_works:
                reasons.append("one or more driving edges lack a final WORKS verdict")
            if not item.composite_promoted:
                reasons.append("frozen composite was not promoted")
            plan = replace(
                plan,
                verdict=TimingVerdict.SKIP,
                rationale=f"{plan.rationale} SKIP: {'; '.join(reasons)}.",
            )
        recommendations.append(
            Recommendation(
                recommendation_id=recommendation_id,
                asset=item.asset,
                direction=item.direction,
                confidence=item.confidence,
                expected_net_return=item.expected_net_return,
                expected_return_ci=item.expected_return_ci,
                holding_period=item.holding_period,
                entry_plan=plan,
                driving_edges=item.driving_edges,
                created_at=created_at,
                borrow_verified=item.borrow_verified,
            )
        )
    return rank_recommendations(
        recommendations, top_n=top_n, minimum_confidence=minimum_confidence
    )
