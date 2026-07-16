"""Timestamped pre-auction filters and deterministic LOC fill simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from edgestack.models import Direction, EntryPlan, OrderType, TimingVerdict


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    """All information frozen before a closing-auction order cutoff."""

    symbol: str
    direction: Direction
    signal_available_at: datetime
    quote_available_at: datetime
    decision_time: datetime
    order_cutoff: datetime
    auction_time: datetime
    previous_close: float
    session_open: float
    decision_price: float
    atr14: float
    event_proximity_sessions: int | None = None
    event_available_at: datetime | None = None

    def __post_init__(self) -> None:
        timestamps = (
            self.signal_available_at,
            self.quote_available_at,
            self.decision_time,
            self.order_cutoff,
            self.auction_time,
        )
        if any(value.tzinfo is None for value in timestamps):
            raise ValueError("decision timestamps must be timezone-aware")
        if self.signal_available_at > self.decision_time:
            raise ValueError("signal was not available at the decision timestamp")
        if self.quote_available_at > self.decision_time:
            raise ValueError("quote was not available at the decision timestamp")
        if not self.decision_time <= self.order_cutoff < self.auction_time:
            raise ValueError("decision must precede cutoff and closing auction")
        if (
            min(
                self.previous_close,
                self.session_open,
                self.decision_price,
                self.atr14,
            )
            <= 0.0
        ):
            raise ValueError("prices and ATR must be positive")
        if self.event_proximity_sessions is not None:
            if self.event_available_at is None:
                raise ValueError("event proximity needs an availability timestamp")
            if self.event_available_at > self.decision_time:
                raise ValueError("event metadata was not available at decision time")


@dataclass(frozen=True, slots=True)
class CausalFilterDecision:
    """Reproducible take/skip result frozen before the auction."""

    snapshot: DecisionSnapshot
    entry_plan: EntryPlan
    gap_fraction: float
    preentry_reversal_atr: float
    event_proximity_sessions: int | None
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuctionFill:
    """One simulated closing-auction result with no same-bar reselection."""

    symbol: str
    filled: bool
    fill_price: float | None
    fill_time: datetime
    reason: str


def gap_fraction(
    session_open: float,
    previous_close: float,
    decision_price: float,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return the fraction of the as-of session move delivered by the open gap."""

    if min(session_open, previous_close, decision_price, epsilon) <= 0.0:
        raise ValueError("prices and epsilon must be positive")
    return abs(session_open - previous_close) / (
        abs(decision_price - previous_close) + epsilon
    )


def preentry_reversal_atr(
    decision_price: float,
    previous_close: float,
    atr14: float,
    direction: Direction,
) -> float:
    """Measure direction-aware mean reversion already consumed before entry."""

    if min(decision_price, previous_close, atr14) <= 0.0:
        raise ValueError("prices and ATR must be positive")
    move = (
        decision_price - previous_close
        if direction is Direction.LONG
        else previous_close - decision_price
    )
    return move / atr14


def trade_similarity(
    correlation_60: float,
    same_sector: bool,
    left_factor_exposure: np.ndarray[Any, np.dtype[np.float64]],
    right_factor_exposure: np.ndarray[Any, np.dtype[np.float64]],
    *,
    correlation_weight: float = 1.0,
    sector_weight: float = 1.0,
    factor_weight: float = 1.0,
) -> float:
    """Return an explicit correlation/sector/factor similarity score."""

    if not -1.0 <= correlation_60 <= 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    if min(correlation_weight, sector_weight, factor_weight) < 0.0:
        raise ValueError("similarity weights cannot be negative")
    left = np.asarray(left_factor_exposure, dtype=float)
    right = np.asarray(right_factor_exposure, dtype=float)
    if (
        left.ndim != 1
        or left.shape != right.shape
        or np.any(~np.isfinite(left + right))
    ):
        raise ValueError("factor exposures must be aligned finite vectors")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    factor_similarity = float(np.dot(left, right) / denominator) if denominator else 0.0
    return float(
        correlation_weight * correlation_60
        + sector_weight * float(same_sector)
        + factor_weight * factor_similarity
    )


def freeze_loc_decision(
    snapshot: DecisionSnapshot,
    *,
    loc_atr_fraction: float = 0.25,
    maximum_preentry_reversal_atr: float = 1.0,
    event_exclusion_sessions: int = 5,
    require_event_data: bool = True,
) -> CausalFilterDecision:
    """Freeze a take/skip decision using only the pre-cutoff snapshot."""

    if loc_atr_fraction <= 0.0 or maximum_preentry_reversal_atr <= 0.0:
        raise ValueError("LOC and pre-entry thresholds must be positive")
    if event_exclusion_sessions < 0:
        raise ValueError("event exclusion cannot be negative")
    gap = gap_fraction(
        snapshot.session_open, snapshot.previous_close, snapshot.decision_price
    )
    consumed = preentry_reversal_atr(
        snapshot.decision_price,
        snapshot.previous_close,
        snapshot.atr14,
        snapshot.direction,
    )
    reasons: list[str] = []
    if consumed > maximum_preentry_reversal_atr:
        reasons.append("PREENTRY_REVERSAL_EXCEEDS_FROZEN_ATR_THRESHOLD")
    proximity = snapshot.event_proximity_sessions
    if proximity is None and require_event_data:
        reasons.append("EVENT_PROXIMITY_DATA_UNAVAILABLE")
    elif proximity is not None and abs(proximity) <= event_exclusion_sessions:
        reasons.append("EVENT_WITHIN_FROZEN_EXCLUSION_WINDOW")
    passed = not reasons
    limit = (
        snapshot.decision_price + loc_atr_fraction * snapshot.atr14
        if snapshot.direction is Direction.LONG
        else snapshot.decision_price - loc_atr_fraction * snapshot.atr14
    )
    verdict = TimingVerdict.ACT_NOW if passed else TimingVerdict.SKIP
    plan = EntryPlan(
        method="FROZEN_PRE_CUTOFF_LOC",
        order_type=OrderType.LOC,
        direction=snapshot.direction,
        verdict=verdict,
        earliest_execution=snapshot.auction_time,
        rationale=("pre-cutoff causal filters passed" if passed else ";".join(reasons)),
        limit_price=limit,
        trigger="CLOSING_AUCTION_AT_OR_BETTER_THAN_LIMIT",
        trigger_value=limit,
        expiry_at=snapshot.auction_time,
        expiry_action="CANCEL_UNFILLED",
        validity_end=snapshot.auction_time,
        data_timestamp=snapshot.quote_available_at,
    )
    return CausalFilterDecision(
        snapshot,
        plan,
        gap,
        consumed,
        proximity,
        passed,
        tuple(reasons),
    )


def simulate_loc_auction(
    decision: CausalFilterDecision,
    *,
    auction_price: float,
    auction_time: datetime,
) -> AuctionFill:
    """Apply the frozen LOC to a later auction price without reselection."""

    snapshot = decision.snapshot
    if auction_price <= 0.0 or not math.isfinite(auction_price):
        raise ValueError("auction price must be finite and positive")
    if auction_time.tzinfo is None or auction_time < snapshot.auction_time:
        raise ValueError("auction fill must occur at or after the frozen auction time")
    if not decision.passed:
        return AuctionFill(
            snapshot.symbol, False, None, auction_time, "FILTERED_BEFORE_CUTOFF"
        )
    limit = decision.entry_plan.limit_price
    if limit is None:
        raise RuntimeError("frozen LOC decision has no limit")
    marketable = (
        auction_price <= limit
        if snapshot.direction is Direction.LONG
        else auction_price >= limit
    )
    return AuctionFill(
        snapshot.symbol,
        marketable,
        auction_price if marketable else None,
        auction_time,
        "FILLED_AT_CLOSING_AUCTION" if marketable else "LOC_LIMIT_NOT_REACHED",
    )
