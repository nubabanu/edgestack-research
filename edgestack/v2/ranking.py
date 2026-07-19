"""Return-first and default loss-first Sniper rankings."""

from __future__ import annotations

from dataclasses import dataclass

from edgestack.v2.metrics import LossMetrics


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """A statistically/OOS-qualified candidate available for ranking."""

    candidate_id: str
    net_mean: float
    sharpe: float
    loss: LossMetrics
    statistical_pass: bool
    oos_pass: bool


def loss_first(candidates: tuple[RankedCandidate, ...]) -> tuple[RankedCandidate, ...]:
    """Minimize tail/loss/MAE/streak risk before maximizing return."""

    eligible = [item for item in candidates if item.statistical_pass and item.oos_pass]
    return tuple(
        sorted(
            eligible,
            key=lambda item: (
                item.loss.expected_shortfall_95,
                item.loss.loss_probability,
                -item.loss.trade_mae,
                item.loss.losing_streak_p90,
                -item.net_mean,
                item.candidate_id,
            ),
        )
    )


def return_first(
    candidates: tuple[RankedCandidate, ...],
) -> tuple[RankedCandidate, ...]:
    """Provide a transparent conventional comparison ranking."""

    eligible = [item for item in candidates if item.statistical_pass and item.oos_pass]
    return tuple(
        sorted(
            eligible, key=lambda item: (-item.sharpe, -item.net_mean, item.candidate_id)
        )
    )
