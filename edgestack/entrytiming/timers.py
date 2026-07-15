"""Causal entry-plan generators and dispatch."""

from __future__ import annotations

from datetime import datetime

from edgestack.entrytiming.regime import TrendRegime
from edgestack.entrytiming.stops import atr_stop, vol_scaled_size
from edgestack.models import Direction, EntryPlan, OrderType, TimingVerdict


def immediate_at_close(
    direction: Direction,
    earliest_close: datetime,
    rationale: str,
    *,
    data_timestamp: datetime,
) -> EntryPlan:
    """Schedule a next-eligible close auction order."""

    _aware(earliest_close, data_timestamp)
    if earliest_close <= data_timestamp:
        raise ValueError("execution must occur after the information timestamp")
    return EntryPlan(
        method="immediate_at_close",
        order_type=OrderType.MOC,
        direction=direction,
        verdict=TimingVerdict.ACT_NOW,
        earliest_execution=earliest_close,
        rationale=rationale,
        data_timestamp=data_timestamp,
    )


def pullback_with_expiry(
    direction: Direction,
    earliest_execution: datetime,
    data_timestamp: datetime,
    expiry_at: datetime,
    *,
    rsi2_value: float,
    bollinger_value: float,
    rsi_threshold: float = 10.0,
    bollinger_threshold: float = 0.2,
    validity_end: datetime | None = None,
) -> EntryPlan:
    """Create a mirrored RSI(2)/percent-B pullback plan."""

    _aware(earliest_execution, data_timestamp, expiry_at)
    if validity_end is not None:
        _aware(validity_end)
    if earliest_execution <= data_timestamp or expiry_at < earliest_execution:
        raise ValueError("invalid causal execution or expiry timestamps")
    if validity_end is not None and expiry_at > validity_end:
        raise ValueError("pullback expiry cannot exceed the base edge validity window")
    if not 0 < rsi_threshold < 50 or not 0 < bollinger_threshold < 0.5:
        raise ValueError("pullback thresholds must be inside their mirrored ranges")
    if direction is Direction.LONG:
        triggered = rsi2_value < rsi_threshold or bollinger_value < bollinger_threshold
        expression = f"RSI(2)<{rsi_threshold:g} OR %B<{bollinger_threshold:g}"
    else:
        triggered = (
            rsi2_value > 100 - rsi_threshold
            or bollinger_value > 1 - bollinger_threshold
        )
        expression = f"RSI(2)>{100-rsi_threshold:g} OR %B>{1-bollinger_threshold:g}"
    return EntryPlan(
        method="pullback_with_expiry",
        order_type=OrderType.LOC,
        direction=direction,
        verdict=TimingVerdict.ACT_NOW if triggered else TimingVerdict.WAIT_FOR_TRIGGER,
        earliest_execution=earliest_execution,
        rationale="Governed pullback overlay; enter next eligible close after trigger.",
        trigger=expression,
        trigger_value=rsi2_value,
        expiry_at=expiry_at,
        expiry_action="enter_at_next_eligible_close",
        validity_end=validity_end,
        data_timestamp=data_timestamp,
    )


def breakout_confirmation(
    direction: Direction,
    earliest_execution: datetime,
    data_timestamp: datetime,
    *,
    price: float,
    trailing_high: float,
    trailing_low: float,
    moving_average: float,
    expiry_at: datetime | None = None,
    validity_end: datetime | None = None,
) -> EntryPlan:
    """Create a causal momentum breakout-confirmation plan."""

    _aware(earliest_execution, data_timestamp)
    if expiry_at is not None:
        _aware(expiry_at)
    if validity_end is not None:
        _aware(validity_end)
    if earliest_execution <= data_timestamp:
        raise ValueError("execution must occur after signal availability")
    if any(
        value <= 0 for value in (price, trailing_high, trailing_low, moving_average)
    ):
        raise ValueError("breakout price inputs must be positive")
    if expiry_at is not None and expiry_at < earliest_execution:
        raise ValueError("breakout expiry precedes earliest execution")
    if expiry_at is not None and validity_end is not None and expiry_at > validity_end:
        raise ValueError("breakout expiry cannot exceed base edge validity")
    triggered = (
        price >= trailing_high
        if direction is Direction.LONG
        else price <= trailing_low and price < moving_average
    )
    trigger = (
        f"price>={trailing_high:.4f}"
        if direction is Direction.LONG
        else f"price<={trailing_low:.4f} AND price<MA200"
    )
    return EntryPlan(
        method="breakout_confirmation",
        order_type=OrderType.LOC,
        direction=direction,
        verdict=TimingVerdict.ACT_NOW if triggered else TimingVerdict.WAIT_FOR_TRIGGER,
        earliest_execution=earliest_execution,
        rationale="Momentum breakout overlay validated independently of base edge.",
        trigger=trigger,
        trigger_value=price,
        expiry_at=expiry_at,
        expiry_action="cancel" if expiry_at is not None else None,
        validity_end=validity_end,
        data_timestamp=data_timestamp,
    )


def select_timer(
    edge_type: str,
    direction: Direction,
    regime: TrendRegime,
    *,
    countertrend_gate_enabled: bool,
) -> str:
    """Return the governed timer name or a counter-trend skip."""

    countertrend = (direction is Direction.LONG and regime is TrendRegime.DOWN) or (
        direction is Direction.SHORT and regime is TrendRegime.UP
    )
    if countertrend and countertrend_gate_enabled:
        return "skip_countertrend"
    if edge_type in {"reversal", "mean_reversion"}:
        return "pullback_with_expiry"
    if edge_type in {"momentum", "52w_high"}:
        return "breakout_confirmation"
    return "immediate_at_close"


def attach_risk(
    plan: EntryPlan,
    *,
    indicative_price: float,
    atr_value: float,
    capital: float = 100_000.0,
    target_risk_fraction: float = 0.005,
) -> EntryPlan:
    """Attach an indicative stop and paper size to a frozen plan."""

    from dataclasses import replace

    stop = atr_stop(indicative_price, atr_value, plan.direction, 2.0)
    per_share_risk = abs(indicative_price - stop)
    shares = vol_scaled_size(
        target_risk_fraction, per_share_risk, capital, indicative_price
    )
    return replace(plan, stop_price=stop, suggested_shares=shares)


def _aware(*values: datetime) -> None:
    if any(value.tzinfo is None or value.utcoffset() is None for value in values):
        raise ValueError("entry-plan timestamps must be timezone-aware")
