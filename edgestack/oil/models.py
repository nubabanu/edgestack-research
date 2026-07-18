"""Strict public contracts for the paper-only oil subsystem."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgestack.disclaimer import DISCLAIMER


class OilWireModel(BaseModel):
    """Immutable and reject-unknown wire model."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class OilContext(OilWireModel):
    """Short-lived operator facts that free market feeds cannot supply."""

    recorded_at: datetime
    expires_at: datetime
    spread_bps: float = Field(ge=0)
    overnight_fee_usd_per_unit: float = Field(ge=0)
    event_risk: Literal["NORMAL", "ELEVATED", "EXTREME", "UNKNOWN"]
    note: str = ""

    @model_validator(mode="after")
    def validate_window(self) -> OilContext:
        if self.recorded_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("oil context timestamps must be timezone-aware")
        if self.expires_at <= self.recorded_at:
            raise ValueError("oil context expiry must follow its recording time")
        return self


class OilDataGate(OilWireModel):
    """One causal input's availability and provenance boundary."""

    name: Literal[
        "PROXY_BARS",
        "EIA_WPSR",
        "CFTC_POSITIONING",
        "FRED_REFERENCES",
        "OPERATOR_CONTEXT",
        "EXACT_ETORO_HISTORY",
        "FUTURES_CURVE",
        "VALIDATED_INTRADAY",
    ]
    status: Literal["PASS", "FAIL", "DATA_UNAVAILABLE"]
    as_of: datetime | None = None
    reason: str
    raw_sha256: tuple[str, ...] = ()


class OilRiskLane(OilWireModel):
    """Risk-sized counterfactual account lane for one paper decision."""

    name: Literal["GOVERNED_0_5", "CHALLENGE_1", "CHALLENGE_2", "CHALLENGE_5", "CHALLENGE_10"]
    label: str
    risk_fraction: float = Field(gt=0, le=0.10)
    status: Literal["ACTIVE", "UNAVAILABLE", "TERMINATED"]
    equity_usd: float = Field(ge=0)
    peak_equity_usd: float = Field(gt=0)
    drawdown_fraction: float = Field(le=0)
    leverage: float | None = None
    notional_usd: float = Field(ge=0)
    margin_usd: float = Field(ge=0)
    stop_fraction: float = Field(ge=0)
    stressed_move_fraction: float = Field(ge=0)
    maximum_planned_loss_usd: float = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    reason: str

    @model_validator(mode="after")
    def validate_lane(self) -> OilRiskLane:
        if self.peak_equity_usd < self.equity_usd:
            raise ValueError("peak paper equity cannot be below current equity")
        if self.status == "ACTIVE" and self.leverage not in {1.0, 2.0, 5.0, 10.0}:
            raise ValueError("an active oil lane requires declared leverage")
        if self.status != "ACTIVE" and self.leverage is not None:
            raise ValueError("an inactive oil lane cannot expose leverage")
        if self.maximum_planned_loss_usd > self.equity_usd * self.risk_fraction + 1e-8:
            raise ValueError("lane planned loss exceeds its account-risk ceiling")
        return self


class OilHorizonDecision(OilWireModel):
    """Paper decision for one independently governed holding horizon."""

    horizon: Literal["INTRADAY", "SWING_3D"]
    status: Literal["NO_TRADE", "WATCH", "PAPER_LONG"]
    evidence_status: Literal["DIAGNOSTIC", "FORWARD_REQUIRED", "FORWARD_TRACKING", "PROMOTED"]
    direction: Literal["LONG"] = "LONG"
    proxy_symbol: Literal["USO"] = "USO"
    signal_session: str
    planned_entry: str
    planned_exit: str
    reference_price_usd: float | None = Field(default=None, gt=0)
    atr14_usd: float | None = Field(default=None, gt=0)
    p99_adverse_gap_fraction: float = Field(ge=0)
    active_vetoes: tuple[str, ...]
    reasons: tuple[str, ...]
    lanes: tuple[OilRiskLane, ...]

    @model_validator(mode="after")
    def validate_decision(self) -> OilHorizonDecision:
        expected = [
            "GOVERNED_0_5",
            "CHALLENGE_1",
            "CHALLENGE_2",
            "CHALLENGE_5",
            "CHALLENGE_10",
        ]
        if [lane.name for lane in self.lanes] != expected:
            raise ValueError("oil risk lanes must be complete and ordered")
        if self.status == "PAPER_LONG" and self.active_vetoes:
            raise ValueError("a paper-long decision cannot retain a hard veto")
        if self.status == "NO_TRADE" and not self.active_vetoes:
            raise ValueError("a no-trade decision requires a visible hard veto")
        return self


class OilSnapshot(OilWireModel):
    """Atomic paper-oil snapshot shared by CLI, ledger, API, and Android."""

    schema_version: Literal["1.0"] = "1.0"
    campaign_id: str
    decision_id: str
    generated_at: datetime
    market_as_of: str
    status: Literal["NO_TRADE", "WATCH", "PAPER_LONG"]
    watermark: str
    outcome_proxy: Literal["USO"] = "USO"
    basis_warning: str
    proxy_agreement: Literal["BULLISH", "BEARISH", "MIXED", "DATA_UNAVAILABLE"]
    data_gates: tuple[OilDataGate, ...]
    intraday: OilHorizonDecision
    swing: OilHorizonDecision
    provenance_warnings: tuple[str, ...] = ()
    disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def validate_snapshot(self) -> OilSnapshot:
        combined = {self.intraday.status, self.swing.status}
        expected = (
            "PAPER_LONG"
            if "PAPER_LONG" in combined
            else "WATCH"
            if "WATCH" in combined
            else "NO_TRADE"
        )
        if self.status != expected:
            raise ValueError("oil snapshot status must summarize its horizons")
        if "NOT_AN_ORDER" not in self.watermark:
            raise ValueError("oil snapshot requires a paper-only watermark")
        return self


__all__ = [
    "OilContext",
    "OilDataGate",
    "OilHorizonDecision",
    "OilRiskLane",
    "OilSnapshot",
]
