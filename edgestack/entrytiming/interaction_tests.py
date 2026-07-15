"""Governance helpers for overlay parameter neighborhoods."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class OverlayEvidence:
    """Incremental evidence for one overlay parameter setting."""

    parameter: float
    incremental_mean: float
    incremental_sharpe: float
    hac_t: float
    fdr_pass: bool
    dsr_probability: float
    oos_t: float
    positive_window_fraction: float
    cost_sensitivity: tuple[tuple[float, float], ...] = ()
    cost_sensitivity_pass: bool = False
    pbo: float | None = None
    pbo_pass: bool = False


@dataclass(frozen=True, slots=True)
class OverlayDecision:
    """Pre-registered enable/disable decision."""

    enabled: bool
    selected_parameter: float | None
    reason: str


def interaction_decision(
    evidence: Sequence[OverlayEvidence],
    *,
    plateau_within: float = 0.20,
) -> OverlayDecision:
    """Enable only a significant configuration supported by an adjacent plateau."""

    if not evidence:
        return OverlayDecision(False, None, "no evidence")
    ordered = sorted(evidence, key=lambda item: item.parameter)
    eligible = [
        item
        for item in ordered
        if item.incremental_mean > 0
        and item.hac_t > 3.0
        and item.fdr_pass
        and item.dsr_probability > 0.95
        and item.oos_t > 2.0
        and item.positive_window_fraction > 0.5
        and item.cost_sensitivity_pass
        and item.pbo_pass
    ]
    if not eligible:
        return OverlayDecision(False, None, "failed incremental statistical gauntlet")
    best = max(eligible, key=lambda item: item.incremental_sharpe)
    floor = best.incremental_sharpe * (1.0 - plateau_within)
    eligible_parameters = {
        item.parameter for item in eligible if item.incremental_sharpe >= floor
    }
    parameter_order = [item.parameter for item in ordered]
    for left, right in pairwise(parameter_order):
        if left in eligible_parameters and right in eligible_parameters:
            center = min((left, right), key=lambda value: abs(value - best.parameter))
            return OverlayDecision(True, center, "adjacent robust parameter plateau")
    return OverlayDecision(False, None, "spiky optimum without adjacent plateau")
