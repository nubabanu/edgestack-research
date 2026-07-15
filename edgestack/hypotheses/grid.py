"""Lazy, deterministic hypothesis enumeration.

Identifiers are SHA-256 hashes of canonical JSON rather than iteration-order
dependent counters. Consequently a hypothesis has the same identity across
machines, Python versions, and grid subsets.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from edgestack.models import (
    Direction,
    RationaleCategory,
    Session,
)
from edgestack.models import (
    HypothesisSpec as HypothesisSpec,
)


@dataclass(frozen=True, slots=True, order=True)
class Predicate:
    """One categorical restriction in the hypothesis grammar."""

    family: str
    value: str

    def __post_init__(self) -> None:
        if not self.family or not self.value:
            raise ValueError("predicate family and value cannot be empty")


def canonical_json(value: Mapping[str, Any] | Sequence[Any]) -> str:
    """Serialize research identity data in a stable, locale-free form."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hypothesis_id(identity: Mapping[str, Any] | Sequence[Any]) -> str:
    """Return the full SHA-256 hex digest for canonical identity data."""

    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


DEFAULT_PREDICATES: dict[str, tuple[str, ...]] = {
    "weekday": ("MON", "TUE", "WED", "THU", "FRI"),
    "month": tuple(str(value) for value in range(1, 13)),
    "turn_of_month": ("TOM", "REST"),
    "holiday": ("PRE", "POST"),
    "fomc": ("DAY_BEFORE", "DAY_OF", "EVENT_WEEK"),
    "opex": ("WEEK",),
}


DEFAULT_RATIONALES: dict[str, RationaleCategory] = {
    "weekday": RationaleCategory.FLOW,
    "month": RationaleCategory.FLOW,
    "turn_of_month": RationaleCategory.FLOW,
    "holiday": RationaleCategory.BEHAVIORAL,
    "fomc": RationaleCategory.RISK_PREMIUM,
    "opex": RationaleCategory.MICROSTRUCTURE,
    "sector": RationaleCategory.RISK_PREMIUM,
}


@dataclass(frozen=True, slots=True)
class GridConfig:
    """Configuration for single-family and pairwise hypothesis generation."""

    predicate_levels: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_PREDICATES)
    )
    sectors: tuple[str, ...] = ()
    holding_periods: tuple[int, ...] = (1, 3, 5, 21)
    directions: tuple[Direction, ...] = (Direction.LONG, Direction.SHORT)
    sessions: tuple[Session, ...] = (
        Session.CLOSE_TO_CLOSE,
        Session.OVERNIGHT,
        Session.INTRADAY,
    )
    include_any: bool = True
    include_pairwise: bool = True
    excluded_family_pairs: frozenset[frozenset[str]] = frozenset()

    def levels(self) -> dict[str, tuple[str, ...]]:
        """Return validated levels including the configured current sectors."""

        levels = {name: tuple(values) for name, values in self.predicate_levels.items()}
        if self.sectors:
            levels["sector"] = tuple(self.sectors)
        for family, values in levels.items():
            if not family or not values or any(not value for value in values):
                raise ValueError("each predicate family needs non-empty unique levels")
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate levels in predicate family {family}")
        if any(period < 1 for period in self.holding_periods):
            raise ValueError("holding periods must be positive")
        return levels


def _predicate_sets(config: GridConfig) -> Iterator[tuple[Predicate, ...]]:
    levels = config.levels()
    if config.include_any:
        yield ()
    ordered_families = sorted(levels)
    for family in ordered_families:
        for value in levels[family]:
            yield (Predicate(family, value),)
    if not config.include_pairwise:
        return
    for left_index, left in enumerate(ordered_families):
        for right in ordered_families[left_index + 1 :]:
            if frozenset((left, right)) in config.excluded_family_pairs:
                continue
            for left_value in levels[left]:
                for right_value in levels[right]:
                    yield tuple(
                        sorted(
                            (Predicate(left, left_value), Predicate(right, right_value))
                        )
                    )


def _rationale(predicates: tuple[Predicate, ...]) -> RationaleCategory:
    if not predicates:
        return RationaleCategory.NONE
    values = {
        DEFAULT_RATIONALES.get(item.family, RationaleCategory.NONE)
        for item in predicates
    }
    return values.pop() if len(values) == 1 else RationaleCategory.NONE


def _execution_combinations(config: GridConfig) -> Iterator[tuple[Session, int]]:
    for session in config.sessions:
        if session is Session.CLOSE_TO_CLOSE:
            for holding in config.holding_periods:
                yield session, holding
        else:
            # Overnight and intraday are one-session return conventions, not
            # aliases for a multi-day close-to-close holding period.
            yield session, 1


def iter_hypotheses(config: GridConfig | None = None) -> Iterator[HypothesisSpec]:
    """Lazily yield every compatible, non-duplicated grid declaration."""

    selected = config or GridConfig()
    for predicates in _predicate_sets(selected):
        rationale = _rationale(predicates)
        for session, holding in _execution_combinations(selected):
            for direction in selected.directions:
                predicate_map = {item.family: item.value for item in predicates}
                condition = (
                    "ANY"
                    if not predicates
                    else " AND ".join(
                        f"{item.family}={item.value}" for item in predicates
                    )
                )
                yield HypothesisSpec(
                    family="calendar",
                    description=(
                        f"{direction.value} {session.value} {holding} session(s) when {condition}"
                    ),
                    predicates=predicate_map,
                    session=session,
                    holding_period=holding,
                    direction=direction,
                    rationale=rationale,
                )


def enumerate_hypotheses(config: GridConfig | None = None) -> list[HypothesisSpec]:
    """Materialize the deterministic grid and reject accidental ID collisions."""

    hypotheses = list(iter_hypotheses(config))
    identifiers = [item.hypothesis_id for item in hypotheses]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("hypothesis identity collision detected")
    return hypotheses


def hypothesis_count(config: GridConfig | None = None) -> int:
    """Count registered statistical trials without retaining them."""

    return sum(1 for _ in iter_hypotheses(config))


def cross_sectional_hypotheses() -> tuple[HypothesisSpec, ...]:
    """Register the four canonical continuous cross-sectional families."""

    definitions = (
        ("momentum_12_1", RationaleCategory.BEHAVIORAL, {"lookback": 252, "skip": 21}),
        ("reversal_5d", RationaleCategory.MICROSTRUCTURE, {"lookback": 5}),
        ("low_volatility", RationaleCategory.RISK_PREMIUM, {"window": 252}),
        ("high_52w_proximity", RationaleCategory.BEHAVIORAL, {"window": 252}),
    )
    specs: list[HypothesisSpec] = []
    for family, rationale, metadata in definitions:
        for direction in (Direction.LONG, Direction.SHORT):
            specs.append(
                HypothesisSpec(
                    family=family,
                    description=f"{direction.value} cross-sectional {family}",
                    predicates={},
                    session=Session.CLOSE_TO_CLOSE,
                    holding_period=21 if family != "reversal_5d" else 5,
                    direction=direction,
                    rationale=rationale,
                    parameters=metadata,
                )
            )
    return tuple(specs)


def pead_hypothesis(*, data_available: bool) -> HypothesisSpec | None:
    """Register PEAD only when timestamped estimates and SUE data exist."""

    if not data_available:
        return None
    return HypothesisSpec(
        family="pead_sue",
        description="LONG post-earnings announcement drift ranked by timestamped SUE",
        predicates={},
        session=Session.CLOSE_TO_CLOSE,
        holding_period=21,
        direction=Direction.LONG,
        rationale=RationaleCategory.BEHAVIORAL,
        parameters={"requires": "timestamped_actual_consensus_sue"},
    )
