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

    schema_version: Literal["1.0"] = "1.0"
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
