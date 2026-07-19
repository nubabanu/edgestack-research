"""Risk-sized leverage lanes for governed and aggressive paper accounts."""

from __future__ import annotations

from dataclasses import dataclass

from edgestack.oil.models import OilRiskLane


@dataclass(frozen=True, slots=True)
class LaneSpec:
    name: str
    label: str
    risk_fraction: float
    notional_cap: float
    margin_cap: float
    challenge: bool


LANES = (
    LaneSpec("GOVERNED_0_5", "Governed research lane", 0.005, 1.0, 1.0, False),
    LaneSpec("CHALLENGE_1", "Challenge lane · 1% account risk", 0.01, 5.0, 0.5, True),
    LaneSpec("CHALLENGE_2", "Challenge lane · 2% account risk", 0.02, 5.0, 0.5, True),
    LaneSpec("CHALLENGE_5", "Challenge lane · 5% account risk", 0.05, 5.0, 0.5, True),
    LaneSpec(
        "CHALLENGE_10",
        "HIGH_RISK_NON_PROMOTABLE · 10% account risk",
        0.10,
        5.0,
        0.5,
        True,
    ),
)


def unavailable_risk_lanes(
    *,
    equity_usd: float,
    reason: str,
    peak_equity_by_lane: dict[str, float] | None = None,
    equity_by_lane: dict[str, float] | None = None,
    terminated_lanes: set[str] | None = None,
) -> tuple[OilRiskLane, ...]:
    """Return complete visible lanes when price/ATR inputs cannot be sized."""

    if equity_usd <= 0:
        raise ValueError("initial paper equity must be positive")
    peak_map = peak_equity_by_lane or {}
    equity_map = equity_by_lane or {}
    terminated = terminated_lanes or set()
    output: list[OilRiskLane] = []
    for spec in LANES:
        lane_equity = max(0.0, float(equity_map.get(spec.name, equity_usd)))
        peak = max(float(peak_map.get(spec.name, equity_usd)), lane_equity)
        drawdown = lane_equity / peak - 1.0
        is_terminated = spec.challenge and (
            spec.name in terminated or lane_equity <= 0 or drawdown <= -0.30
        )
        output.append(
            OilRiskLane(
                name=spec.name,
                label=spec.label,
                risk_fraction=spec.risk_fraction,
                status="TERMINATED" if is_terminated else "UNAVAILABLE",
                equity_usd=lane_equity,
                peak_equity_usd=peak,
                drawdown_fraction=drawdown,
                leverage=None,
                notional_usd=0.0,
                margin_usd=0.0,
                stop_fraction=0.0,
                stressed_move_fraction=0.0,
                maximum_planned_loss_usd=0.0,
                estimated_cost_usd=0.0,
                reason=(
                    "challenge lane previously crossed its irreversible campaign stop"
                    if is_terminated
                    else reason
                ),
            )
        )
    return tuple(output)


def size_risk_lanes(
    *,
    equity_usd: float,
    price_usd: float,
    atr14_usd: float,
    p99_adverse_gap_fraction: float,
    spread_bps: float,
    overnight_fee_usd_per_unit: float,
    holding_nights: int,
    peak_equity_by_lane: dict[str, float] | None = None,
    equity_by_lane: dict[str, float] | None = None,
    daily_loss_by_lane: dict[str, float] | None = None,
    open_risk_by_lane: dict[str, float] | None = None,
    terminated_lanes: set[str] | None = None,
) -> tuple[OilRiskLane, ...]:
    """Size every declared lane without letting leverage alter account risk."""

    if equity_usd <= 0 or price_usd <= 0 or atr14_usd <= 0:
        raise ValueError("initial equity, price, and ATR must be positive")
    if p99_adverse_gap_fraction < 0 or spread_bps < 0 or holding_nights < 0:
        raise ValueError("gap, spread, and holding nights cannot be negative")
    peak_map = peak_equity_by_lane or {}
    equity_map = equity_by_lane or {}
    daily_map = daily_loss_by_lane or {}
    open_map = open_risk_by_lane or {}
    terminated = terminated_lanes or set()
    stop_fraction = 2.0 * atr14_usd / price_usd
    stressed_move = stop_fraction + p99_adverse_gap_fraction
    output: list[OilRiskLane] = []
    for spec in LANES:
        lane_equity = float(equity_map.get(spec.name, equity_usd))
        if lane_equity < 0:
            lane_equity = 0.0
        peak = max(float(peak_map.get(spec.name, equity_usd)), lane_equity)
        drawdown = lane_equity / peak - 1.0
        base = dict(
            name=spec.name,
            label=spec.label,
            risk_fraction=spec.risk_fraction,
            equity_usd=lane_equity,
            peak_equity_usd=peak,
            drawdown_fraction=drawdown,
            stop_fraction=stop_fraction,
            stressed_move_fraction=stressed_move,
        )
        if spec.challenge and (
            spec.name in terminated or lane_equity <= 0 or drawdown <= -0.30
        ):
            output.append(
                OilRiskLane(
                    **base,
                    status="TERMINATED",
                    leverage=None,
                    notional_usd=0.0,
                    margin_usd=0.0,
                    maximum_planned_loss_usd=0.0,
                    estimated_cost_usd=0.0,
                    reason="challenge lane terminated at a 30% peak-to-trough drawdown",
                )
            )
            continue
        if lane_equity <= 0:
            output.append(
                OilRiskLane(
                    **base,
                    status="UNAVAILABLE",
                    leverage=None,
                    notional_usd=0.0,
                    margin_usd=0.0,
                    maximum_planned_loss_usd=0.0,
                    estimated_cost_usd=0.0,
                    reason="governed lane has non-positive paper equity",
                )
            )
            continue
        risk_budget = lane_equity * spec.risk_fraction
        remaining = risk_budget - float(daily_map.get(spec.name, 0.0)) - float(
            open_map.get(spec.name, 0.0)
        )
        if remaining <= 0:
            output.append(
                OilRiskLane(
                    **base,
                    status="UNAVAILABLE",
                    leverage=None,
                    notional_usd=0.0,
                    margin_usd=0.0,
                    maximum_planned_loss_usd=0.0,
                    estimated_cost_usd=0.0,
                    reason="lane daily/open risk already consumes its account-risk ceiling",
                )
            )
            continue
        notional = min(remaining / stop_fraction, lane_equity * spec.notional_cap)
        planned_loss = notional * stop_fraction
        selected: float | None = None
        for leverage in (10.0, 5.0, 2.0, 1.0):
            margin = notional / leverage
            if margin <= lane_equity * spec.margin_cap and leverage * stressed_move <= 0.50:
                selected = leverage
                break
        if selected is None:
            output.append(
                OilRiskLane(
                    **base,
                    status="UNAVAILABLE",
                    leverage=None,
                    notional_usd=0.0,
                    margin_usd=0.0,
                    maximum_planned_loss_usd=0.0,
                    estimated_cost_usd=0.0,
                    reason="no declared leverage passes margin and stressed-move limits",
                )
            )
            continue
        units = notional / price_usd
        cost = notional * spread_bps / 10_000.0 + (
            units * overnight_fee_usd_per_unit * holding_nights
        )
        if cost > max(planned_loss * 0.10, 0.01):
            output.append(
                OilRiskLane(
                    **base,
                    status="UNAVAILABLE",
                    leverage=None,
                    notional_usd=0.0,
                    margin_usd=0.0,
                    maximum_planned_loss_usd=0.0,
                    estimated_cost_usd=cost,
                    reason="estimated spread/financing exceeds 10% of planned loss",
                )
            )
            continue
        output.append(
            OilRiskLane(
                **base,
                status="ACTIVE",
                leverage=selected,
                notional_usd=notional,
                margin_usd=notional / selected,
                maximum_planned_loss_usd=planned_loss,
                estimated_cost_usd=cost,
                reason="risk-sized; leverage changes margin, not maximum account loss",
            )
        )
    return tuple(output)


__all__ = ["LANES", "LaneSpec", "size_risk_lanes", "unavailable_risk_lanes"]
