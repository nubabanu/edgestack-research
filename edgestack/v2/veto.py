"""Preregistered event-risk vetoes and plateau enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import StrEnum

import numpy as np

from edgestack.data.calendars import NYSECalendar
from edgestack.models import CorporateEvent, CorporateEventKind


class VetoKind(StrEnum):
    """Declared V2 veto families."""

    NONE = "NONE"
    EARNINGS_WINDOW = "EARNINGS_WINDOW"
    NEGATIVE_GUIDANCE = "NEGATIVE_GUIDANCE"
    ACTIVE_HALT = "ACTIVE_HALT"
    GAP_PERCENT = "GAP_PERCENT"
    GAP_ATR = "GAP_ATR"


@dataclass(frozen=True, slots=True)
class VetoSpec:
    """One fixed veto configuration."""

    kind: VetoKind
    threshold: float | None = None
    pre_entry_sessions: int = 5

    @property
    def label(self) -> str:
        suffix = "" if self.threshold is None else f":{self.threshold:g}"
        return f"{self.kind.value}{suffix}"


@dataclass(frozen=True, slots=True)
class VetoEvidence:
    """Incremental veto evidence used for fail-closed enablement."""

    spec: VetoSpec
    incremental_net_return: float
    loss_probability_change: float
    expected_shortfall_change: float
    mae_change: float
    incremental_sharpe: float

    @property
    def improves_loss(self) -> bool:
        return (
            self.loss_probability_change <= 0
            and self.expected_shortfall_change <= 0
            and self.mae_change >= 0
        )


def declared_vetoes() -> tuple[VetoSpec, ...]:
    """Return the immutable V2 veto neighborhood."""

    return (
        VetoSpec(VetoKind.NONE),
        VetoSpec(VetoKind.EARNINGS_WINDOW),
        VetoSpec(VetoKind.NEGATIVE_GUIDANCE),
        VetoSpec(VetoKind.ACTIVE_HALT),
        *(VetoSpec(VetoKind.GAP_PERCENT, value) for value in (0.03, 0.05, 0.08)),
        *(VetoSpec(VetoKind.GAP_ATR, value) for value in (1.5, 2.0, 2.5)),
    )


def event_is_vetoed(
    spec: VetoSpec,
    events: tuple[CorporateEvent, ...],
    *,
    decision_time: datetime,
    hold_end: datetime,
) -> bool:
    """Apply only events available at the decision cutoff."""

    if spec.kind in {VetoKind.NONE, VetoKind.GAP_PERCENT, VetoKind.GAP_ATR}:
        return False
    if decision_time.tzinfo is None or hold_end.tzinfo is None:
        raise ValueError("veto bounds must be timezone-aware")
    causal = [event for event in events if event.available_at <= decision_time]
    if spec.kind is VetoKind.NEGATIVE_GUIDANCE:
        return any(
            item.kind is CorporateEventKind.GUIDANCE
            and item.sentiment is not None
            and item.sentiment < 0
            for item in causal
        )
    if spec.kind is VetoKind.ACTIVE_HALT:
        return any(
            item.kind is CorporateEventKind.TRADING_HALT
            and item.event_time <= decision_time
            and _halt_end(item) > decision_time
            for item in causal
        )
    target = {CorporateEventKind.EARNINGS, CorporateEventKind.PRELIMINARY_RESULTS}
    window_start = decision_time
    calendar = NYSECalendar()
    session = decision_time.date()
    for _ in range(spec.pre_entry_sessions):
        session = calendar.previous_session(session).date()
    window_start = datetime.combine(session, time.min, tzinfo=decision_time.tzinfo)
    return any(
        item.kind in target and window_start <= item.event_time <= hold_end
        for item in causal
    )


def _halt_end(event: CorporateEvent) -> datetime:
    value = event.metadata.get("ended_at")
    if value is None:
        return datetime.max.replace(tzinfo=event.event_time.tzinfo)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("halt ended_at must be timezone-aware")
    return parsed


def gap_is_vetoed(spec: VetoSpec, gap_return: float, atr_fraction: float) -> bool:
    """Apply fixed percent or ATR gap neighborhoods symmetrically."""

    if spec.kind is VetoKind.GAP_PERCENT:
        if spec.threshold is None:
            raise ValueError("gap-percent veto requires a threshold")
        return abs(gap_return) >= spec.threshold
    if spec.kind is VetoKind.GAP_ATR:
        if atr_fraction <= 0:
            raise ValueError("atr_fraction must be positive")
        if spec.threshold is None:
            raise ValueError("gap-ATR veto requires a threshold")
        return abs(gap_return) / atr_fraction >= spec.threshold
    return False


def enabled_plateau(evidence: tuple[VetoEvidence, ...]) -> tuple[str, ...]:
    """Enable only loss-improving settings with an adjacent 20%-of-best plateau."""

    by_kind: dict[VetoKind, list[VetoEvidence]] = {}
    for item in evidence:
        if item.spec.threshold is not None:
            by_kind.setdefault(item.spec.kind, []).append(item)
    enabled: list[str] = []
    for items in by_kind.values():
        ordered = sorted(
            items,
            key=lambda item: (
                item.spec.threshold
                if item.spec.threshold is not None
                else float("-inf")
            ),
        )
        best = max((item.incremental_sharpe for item in ordered), default=-np.inf)
        tolerance = 0.2 * abs(best)
        eligible = [
            item.incremental_net_return >= 0
            and item.improves_loss
            and np.sign(item.incremental_sharpe) == np.sign(best)
            and abs(item.incremental_sharpe - best) <= tolerance
            for item in ordered
        ]
        for index in range(len(ordered) - 1):
            if eligible[index] and eligible[index + 1]:
                enabled.extend(
                    (ordered[index].spec.label, ordered[index + 1].spec.label)
                )
    return tuple(dict.fromkeys(enabled))
