"""Transparent, realistic transaction-cost modeling.

Discovery consumes the net return produced here. Spread input is a *full*
quoted spread; each marketable fill pays half. Square-root participation impact
is capped independently on each side. Borrow accrues ACT/365.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from edgestack.stats._types import FloatArray

# Optimistic liquidity assumed when ADV is unknown; keeps missing-data names
# from silently escaping participation impact rather than modeling them well.
DEFAULT_ADV_FALLBACK_DOLLARS = 100_000_000.0


@dataclass(frozen=True, slots=True)
class CostAssumptions:
    """Frozen baseline cost assumptions."""

    portfolio_capital: float = 100_000.0
    commission_per_side: float = 0.0
    etf_full_spread_bps: float = 1.0
    equity_full_spread_bps: float = 3.0
    base_slippage_bps: float = 1.0
    impact_coefficient_bps: float = 10.0
    max_impact_bps: float = 50.0
    easy_borrow_annual: float = 0.003
    turnover_penalty_bps: float = 1.0

    @classmethod
    def from_config(cls, config: Any) -> CostAssumptions:
        """Adapt the shared Pydantic CostConfig without importing it."""

        return cls(
            portfolio_capital=float(config.capital),
            commission_per_side=float(config.commission_per_side),
            etf_full_spread_bps=float(config.etf_full_spread_bps),
            equity_full_spread_bps=float(config.equity_full_spread_bps),
            base_slippage_bps=float(config.base_slippage_bps),
            impact_coefficient_bps=float(config.impact_coefficient_bps),
            max_impact_bps=float(config.max_impact_bps),
            easy_borrow_annual=float(config.easy_borrow_annual),
            turnover_penalty_bps=float(config.turnover_penalty_bps),
        )


@dataclass(frozen=True, slots=True)
class TradeIntent:
    """Cost-relevant details of a paper trade."""

    order_dollars: float
    holding_days: float = 1.0
    is_short: bool = False
    fills: int = 2
    one_way_turnover: float = 1.0
    order_type: Literal["MARKETABLE", "LIMIT", "MOC", "LOC"] = "MARKETABLE"


@dataclass(frozen=True, slots=True)
class MarketContext:
    """Liquidity and instrument context known at order construction."""

    adv_dollars: float
    asset_type: Literal["equity", "etf"] = "equity"
    spread_multiplier: float = 1.0


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Round-trip costs in basis points and return-fraction units."""

    commission_bps: float
    spread_bps: float
    slippage_bps: float
    borrow_bps: float
    turnover_penalty_bps: float
    total_bps: float

    @property
    def return_fraction(self) -> float:
        """Total cost expressed as a decimal return."""

        return self.total_bps / 10_000.0


class CostModel:
    """Estimate per-trade and portfolio time-series costs."""

    def __init__(self, assumptions: CostAssumptions | Any | None = None) -> None:
        if assumptions is None:
            self.assumptions = CostAssumptions()
        elif isinstance(assumptions, CostAssumptions):
            self.assumptions = assumptions
        else:
            self.assumptions = CostAssumptions.from_config(assumptions)

    def estimate(self, intent: TradeIntent, context: MarketContext) -> CostBreakdown:
        """Estimate cost for a complete trade intent.

        Limit/LOC orders are conservatively charged the same baseline unless a
        caller supplies realized fills; an assumed spread capture is never used
        to manufacture alpha. Fill risk is handled by execution simulation.
        """

        if intent.order_dollars <= 0.0 or context.adv_dollars <= 0.0:
            raise ValueError("order_dollars and adv_dollars must be positive")
        if (
            intent.fills < 1
            or intent.holding_days < 0.0
            or intent.one_way_turnover < 0.0
        ):
            raise ValueError(
                "fills must be positive and horizons/turnover non-negative"
            )
        assumptions = self.assumptions
        spread = (
            assumptions.etf_full_spread_bps
            if context.asset_type == "etf"
            else assumptions.equity_full_spread_bps
        )
        spread_bps = spread * context.spread_multiplier * intent.fills / 2.0
        participation = intent.order_dollars / context.adv_dollars
        per_fill_slippage = min(
            assumptions.base_slippage_bps
            + assumptions.impact_coefficient_bps * math.sqrt(max(participation, 0.0)),
            assumptions.max_impact_bps,
        )
        slippage_bps = per_fill_slippage * intent.fills
        commission_bps = (
            assumptions.commission_per_side
            * intent.fills
            / intent.order_dollars
            * 10_000.0
        )
        borrow_bps = (
            assumptions.easy_borrow_annual * intent.holding_days / 365.0 * 10_000.0
            if intent.is_short
            else 0.0
        )
        turnover_bps = assumptions.turnover_penalty_bps * intent.one_way_turnover
        total = commission_bps + spread_bps + slippage_bps + borrow_bps + turnover_bps
        return CostBreakdown(
            commission_bps=commission_bps,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            borrow_bps=borrow_bps,
            turnover_penalty_bps=turnover_bps,
            total_bps=total,
        )

    def portfolio_costs(
        self,
        positions: FloatArray,
        *,
        asset_type: Literal["equity", "etf"] | Sequence[str] = "equity",
        order_dollars: float | None = None,
        adv_dollars: float | FloatArray = DEFAULT_ADV_FALLBACK_DOLLARS,
        short_borrow_days: float = 1.0,
        multiplier: float = 1.0,
        full_spread_bps: FloatArray | None = None,
    ) -> FloatArray:
        """Return daily costs for lagged portfolio weights.

        Position changes are one-way turnover. Each changed dollar is one fill,
        so spread is charged at half the full spread and impact is scaled by the
        traded fraction. Short borrow is charged on maintained short exposure.
        ``full_spread_bps`` optionally overrides the assumed per-instrument
        spread with a measured per-date-per-name matrix; callers must floor
        measured values at the baseline so measurement never lowers costs.
        """

        weights = np.asarray(positions, dtype=float)
        if weights.ndim == 1:
            weights = weights[:, None]
        if weights.ndim != 2:
            raise ValueError("positions must be one- or two-dimensional")
        if multiplier < 0.0:
            raise ValueError("multiplier cannot be negative")
        selected_order_dollars = (
            self.assumptions.portfolio_capital
            if order_dollars is None
            else float(order_dollars)
        )
        if selected_order_dollars <= 0.0:
            raise ValueError("order_dollars must be positive")
        liquidity = np.asarray(adv_dollars, dtype=float)
        if liquidity.ndim == 0:
            liquidity = np.full(weights.shape, float(liquidity))
        elif liquidity.ndim == 1 and liquidity.shape[0] == weights.shape[0]:
            liquidity = liquidity[:, None]
            liquidity = np.broadcast_to(liquidity, weights.shape)
        elif liquidity.shape != weights.shape:
            try:
                liquidity = np.broadcast_to(liquidity, weights.shape)
            except ValueError as error:
                raise ValueError(
                    "adv_dollars cannot be broadcast to positions"
                ) from error
        previous = np.zeros_like(weights)
        previous[1:] = weights[:-1]
        trades = np.abs(weights - previous)
        if isinstance(asset_type, str):
            full_spread: float | np.ndarray[Any, np.dtype[np.float64]] = (
                self.assumptions.etf_full_spread_bps
                if asset_type.lower() == "etf"
                else self.assumptions.equity_full_spread_bps
            )
        else:
            kinds = np.asarray(tuple(asset_type), dtype=str)
            if kinds.ndim != 1 or len(kinds) != weights.shape[1]:
                raise ValueError("asset_type sequence must align with position columns")
            full_spread = np.where(
                np.char.lower(kinds) == "etf",
                self.assumptions.etf_full_spread_bps,
                self.assumptions.equity_full_spread_bps,
            )[None, :]
        if full_spread_bps is not None:
            measured = np.asarray(full_spread_bps, dtype=float)
            if measured.shape != weights.shape:
                raise ValueError(
                    "full_spread_bps must align with the positions matrix"
                )
            if np.any(~np.isfinite(measured)) or np.any(
                measured < np.asarray(full_spread, dtype=float)
            ):
                raise ValueError(
                    "measured spreads must be finite and floored at the "
                    "assumed baseline"
                )
            full_spread = measured
        participation = np.divide(
            trades * selected_order_dollars,
            liquidity,
            out=np.zeros_like(trades),
            where=liquidity > 0.0,
        )
        per_fill_impact = np.minimum(
            self.assumptions.base_slippage_bps
            + self.assumptions.impact_coefficient_bps * np.sqrt(participation),
            self.assumptions.max_impact_bps,
        )
        execution_bps = trades * (full_spread / 2.0 + per_fill_impact)
        turnover_bps = trades * self.assumptions.turnover_penalty_bps
        borrow_bps = (
            np.maximum(-weights, 0.0)
            * self.assumptions.easy_borrow_annual
            * short_borrow_days
            / 365.0
            * 10_000.0
        )
        commission_fraction = (
            (trades > 0.0).astype(float)
            * self.assumptions.commission_per_side
            / selected_order_dollars
        )
        costs = (execution_bps + turnover_bps + borrow_bps).sum(axis=1) / 10_000.0
        costs += commission_fraction.sum(axis=1)
        return cast(FloatArray, costs * multiplier)

    def sensitivity(
        self,
        gross_returns: FloatArray,
        positions: FloatArray,
        *,
        multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
        **kwargs: Any,
    ) -> dict[float, FloatArray]:
        """Return net streams at preregistered cost multipliers."""

        gross = np.asarray(gross_returns, dtype=float)
        return {
            scale: gross - self.portfolio_costs(positions, multiplier=scale, **kwargs)
            for scale in multipliers
        }

    def capacity_curve(
        self,
        gross_returns: FloatArray,
        positions: FloatArray,
        *,
        capital_multipliers: tuple[float, ...] = (1.0, 10.0, 100.0, 1_000.0),
        **kwargs: Any,
    ) -> dict[float, FloatArray]:
        """Return net streams across explicit portfolio-capital assumptions.

        Scaling capital changes only participation-based impact and commission
        fractions; signal returns, spreads, borrow, and turnover penalties stay
        fixed.  This is an implementation-capacity diagnostic, not an alpha
        source and not a substitute for realized implementation shortfall.
        """

        if not capital_multipliers or any(
            value <= 0.0 for value in capital_multipliers
        ):
            raise ValueError("capital_multipliers must contain positive values")
        gross = np.asarray(gross_returns, dtype=float)
        return {
            scale: gross
            - self.portfolio_costs(
                positions,
                order_dollars=self.assumptions.portfolio_capital * scale,
                **kwargs,
            )
            for scale in capital_multipliers
        }


def break_even_cost_multiplier(gross_mean: float, baseline_cost_mean: float) -> float:
    """Return the cost multiplier at which expected net mean reaches zero."""

    if baseline_cost_mean < 0.0:
        raise ValueError("baseline_cost_mean cannot be negative")
    if baseline_cost_mean == 0.0:
        return math.inf if gross_mean > 0.0 else 0.0
    return max(gross_mean / baseline_cost_mean, 0.0)
