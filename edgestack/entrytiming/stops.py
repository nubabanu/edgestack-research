"""Paper-risk calculations for stops, time exits, and position sizing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

from edgestack.models import Direction


def atr_stop(
    entry: float, atr_value: float, direction: Direction, k: float = 2.0
) -> float:
    """Return a direction-aware ATR stop price."""

    if entry <= 0 or atr_value <= 0 or k <= 0:
        raise ValueError("entry, ATR, and k must be positive")
    stop = (
        entry - k * atr_value if direction is Direction.LONG else entry + k * atr_value
    )
    return max(0.0, stop)


def time_exit(
    entry_date: date, holding_sessions: int, sessions: Sequence[date]
) -> date:
    """Select the exit session strictly after an entry date."""

    if holding_sessions <= 0:
        raise ValueError("holding_sessions must be positive")
    future = sorted({session for session in sessions if session > entry_date})
    if len(future) < holding_sessions:
        raise ValueError("calendar does not contain the requested exit session")
    return future[holding_sessions - 1]


def vol_scaled_size(
    target_risk_fraction: float,
    risk_measure: float,
    capital: float,
    price: float,
    *,
    max_position_fraction: float = 0.10,
    adv_shares: float | None = None,
    max_adv_fraction: float = 0.01,
) -> int:
    """Conservative integer paper size bounded by risk, capital, and ADV."""

    if (
        not 0 < target_risk_fraction <= 1
        or capital <= 0
        or price <= 0
        or risk_measure <= 0
        or not 0 < max_position_fraction <= 1
        or not 0 < max_adv_fraction <= 1
    ):
        raise ValueError(
            "risk/capital/price inputs and sizing fractions must be positive and valid"
        )
    by_risk = target_risk_fraction * capital / risk_measure
    by_capital = max_position_fraction * capital / price
    size = min(by_risk, by_capital)
    if adv_shares is not None:
        if adv_shares < 0:
            raise ValueError("ADV cannot be negative")
        size = min(size, max_adv_fraction * adv_shares)
    return max(0, math.floor(size))
