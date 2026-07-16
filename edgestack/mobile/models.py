"""Versioned, immutable wire models for the Android paper companion."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from edgestack.disclaimer import DISCLAIMER


class WireModel(BaseModel):
    """Strict immutable base for every mobile payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ApiMeta(WireModel):
    """Version and freshness metadata for one atomic snapshot."""

    schema_version: Literal["1.2"] = "1.2"
    generated_at: datetime
    market_as_of: str
    source: str
    mode: Literal["SEALED", "DEMO"]
    stale: bool


class EntryInstruction(WireModel):
    """Causal paper-only entry and exit instructions."""

    entry_session: str
    entry_order: Literal["MOC"]
    submit_by_et: str
    exit_session: str
    exit_order: Literal["MOC"]
    no_chase: str
    cancel_if: tuple[str, ...]


class MobileRecommendation(WireModel):
    """One ranked constituent of the inseparable paper basket."""

    recommendation_id: str
    rank: int = Field(ge=1, le=10)
    symbol: str = Field(pattern=r"^[A-Z][A-Z0-9.\-]{0,9}$")
    direction: Literal["LONG", "SHORT"]
    confidence_ordinal: int = Field(ge=0, le=100)
    signal_close_usd: float = Field(gt=0)
    trailing_return: float
    suggested_shares: int = Field(ge=0)
    reference_stop_usd: float | None = Field(default=None, gt=0)
    event_risk: str


class HoldoutEvidence(WireModel):
    """Replay-only summary of the sealed final holdout."""

    status: Literal["PASS", "FAIL", "UNAVAILABLE"]
    start: str
    end: str
    observations: int = Field(ge=0)
    expected_sessions: int = Field(ge=0)
    net_mean: float | None
    benchmark_excess_mean: float | None
    terminal_wealth: float | None
    benchmark_wealth: float | None
    freeze_id: str
    result_sha256: str


class PortfolioSummary(WireModel):
    """Frozen paper portfolio contract."""

    paper_capital_usd: float = Field(gt=0)
    target_gross: float = Field(ge=0, le=2)
    maximum_name_weight: float = Field(gt=0, le=1)
    risk_budget_per_name_usd: float = Field(ge=0)
    shorts_enabled: bool


class AuditItem(WireModel):
    """User-visible immutable evidence or lifecycle event."""

    occurred_at: datetime
    event_type: str
    message: str


class HorizonPlan(WireModel):
    """Evidence-aware availability and timing for one investment horizon."""

    horizon: Literal["WEEK", "MONTH", "YEAR"]
    status: Literal["CONDITIONAL_PAPER_SIGNAL", "DATA_UNAVAILABLE"]
    title: str
    holding_period: str
    entry_rule: str
    review_rule: str
    exit_rule: str
    recommendation_scope: Literal["BASKET", "NONE"]
    symbols: tuple[str, ...] = ()
    evidence: str
    invalidation: tuple[str, ...]
    unlock_requirement: str


class AlignmentLayer(WireModel):
    """One causal layer required by the loss-first Sniper policy."""

    horizon: Literal["YEAR", "MONTH", "WEEK", "DAY"]
    status: Literal["PASS", "PENDING", "UNVALIDATED", "FAIL"]
    evidence: str


class SniperPolicy(WireModel):
    """Conservative paper overlay that defaults to no trade."""

    status: Literal["NO_TRADE", "CONDITIONAL_PAPER_CANDIDATE"]
    objective: Literal["LOSS_FIRST"] = "LOSS_FIRST"
    candidate_symbols: tuple[str, ...]
    max_name_weight: float = Field(gt=0, le=1)
    max_gross_exposure: float = Field(gt=0, le=1)
    max_planned_loss_per_name_usd: float = Field(gt=0)
    max_planned_basket_loss_usd: float = Field(gt=0)
    execution_window: str
    alignments: tuple[AlignmentLayer, ...]
    hard_vetoes: tuple[str, ...]
    release_condition: str
    stop_warning: str
    validation_status: Literal["RISK_OVERLAY_NOT_VALIDATED_ALPHA"]


class MobileSnapshot(WireModel):
    """Atomic Android home-screen payload."""

    meta: ApiMeta
    campaign_id: str
    model_name: str
    model_status: Literal["PROMOTED", "REJECTED", "DEMO"]
    bias_tier: Literal["SURVIVORSHIP_BIASED", "POINT_IN_TIME"]
    watermark: str
    basket_rule: str
    instruction: EntryInstruction
    portfolio: PortfolioSummary
    recommendations: tuple[MobileRecommendation, ...]
    skipped: tuple[MobileRecommendation, ...] = ()
    holdout: HoldoutEvidence
    audit: tuple[AuditItem, ...]
    horizons: tuple[HorizonPlan, ...]
    sniper: SniperPolicy
    disclaimer: str = DISCLAIMER

    def model_post_init(self, __context: object) -> None:
        """Reject partial or reordered representations of the tested basket."""

        ranks = [item.rank for item in self.recommendations]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("recommendation ranks must be contiguous and ordered")
        if self.model_status == "PROMOTED" and self.holdout.status != "PASS":
            raise ValueError("a promoted mobile model requires a passed holdout")
        if (
            any(item.direction == "SHORT" for item in self.recommendations)
            and not self.portfolio.shorts_enabled
        ):
            raise ValueError("short recommendation emitted while shorts are disabled")
        if [plan.horizon for plan in self.horizons] != ["WEEK", "MONTH", "YEAR"]:
            raise ValueError("mobile horizons must contain WEEK, MONTH, YEAR in order")
        weekly = self.horizons[0]
        symbols = tuple(item.symbol for item in self.recommendations)
        if weekly.recommendation_scope != "BASKET" or weekly.symbols != symbols:
            raise ValueError("weekly horizon must preserve the complete tested basket")
        if any(
            plan.status == "DATA_UNAVAILABLE"
            and (plan.recommendation_scope != "NONE" or plan.symbols)
            for plan in self.horizons
        ):
            raise ValueError("unavailable horizons cannot emit stock recommendations")
        if [layer.horizon for layer in self.sniper.alignments] != [
            "YEAR",
            "MONTH",
            "WEEK",
            "DAY",
        ]:
            raise ValueError("sniper alignment must contain YEAR, MONTH, WEEK, DAY")
        if self.sniper.candidate_symbols != symbols:
            raise ValueError(
                "sniper watchlist must preserve the complete weekly basket"
            )
        if self.sniper.status == "CONDITIONAL_PAPER_CANDIDATE" and any(
            layer.status != "PASS" for layer in self.sniper.alignments
        ):
            raise ValueError("sniper candidate requires every alignment layer to pass")
        if self.sniper.status == "NO_TRADE" and not self.sniper.hard_vetoes:
            raise ValueError("sniper no-trade status requires a visible hard veto")
