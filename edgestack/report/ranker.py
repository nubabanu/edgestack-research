"""Honest top-N long/short ranking without padding."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from edgestack.models import Direction, Recommendation, TimingVerdict


@dataclass(frozen=True, slots=True)
class RankedRecommendations:
    """Actionable rankings plus rejected audit rows."""

    longs: tuple[Recommendation, ...]
    shorts: tuple[Recommendation, ...]
    skipped: tuple[Recommendation, ...]


def rank_recommendations(
    candidates: Iterable[Recommendation],
    *,
    top_n: int = 5,
    minimum_confidence: int = 60,
) -> RankedRecommendations:
    """Select at most ``top_n`` names per direction, never padding."""

    if top_n <= 0 or not 0 <= minimum_confidence <= 100:
        raise ValueError("invalid ranking thresholds")
    actionable: list[Recommendation] = []
    skipped: list[Recommendation] = []
    identities: set[str] = set()
    for candidate in candidates:
        if not 0 <= candidate.confidence <= 100:
            raise ValueError("recommendation confidence must be in [0, 100]")
        if candidate.direction is not candidate.entry_plan.direction:
            raise ValueError("recommendation and entry-plan directions disagree")
        if candidate.recommendation_id in identities:
            raise ValueError(
                f"duplicate recommendation ID: {candidate.recommendation_id}"
            )
        identities.add(candidate.recommendation_id)
        if (
            candidate.confidence < minimum_confidence
            or candidate.entry_plan.verdict is TimingVerdict.SKIP
        ):
            skipped.append(candidate)
        else:
            actionable.append(candidate)

    def key(item: Recommendation) -> tuple[int, float, str, str]:
        return (
            -item.confidence,
            -item.expected_net_return,
            item.asset.symbol,
            item.recommendation_id,
        )

    longs = sorted(
        (item for item in actionable if item.direction is Direction.LONG), key=key
    )[:top_n]
    shorts = sorted(
        (item for item in actionable if item.direction is Direction.SHORT), key=key
    )[:top_n]
    return RankedRecommendations(
        tuple(longs), tuple(shorts), tuple(sorted(skipped, key=key))
    )
