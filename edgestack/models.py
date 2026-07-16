"""Shared immutable domain contracts for EdgeStack.

The models deliberately distinguish event time from availability time.  Research
code must use :class:`CausalDataView` rather than indexing a complete frame
directly, which makes accidental future-data access observable and testable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import pandas as pd


class Direction(StrEnum):
    """Trade direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class Session(StrEnum):
    """Return session used by a hypothesis."""

    CLOSE_TO_CLOSE = "close_to_close"
    OVERNIGHT = "overnight"
    INTRADAY = "intraday"


class ExecutionConvention(StrEnum):
    """Causal signal-to-fill convention."""

    SIGNAL_CLOSE_TO_NEXT_CLOSE = "signal_close_to_next_close"
    PRIOR_CLOSE_TO_NEXT_OPEN = "prior_close_to_next_open"
    SIGNAL_OPEN_TO_NEXT_CLOSE = "signal_open_to_next_close"
    PRE_CUTOFF_TO_MOC = "pre_cutoff_to_moc"


class RationaleCategory(StrEnum):
    """Predeclared economic rationale taxonomy."""

    FLOW = "flow-based"
    BEHAVIORAL = "behavioral"
    RISK_PREMIUM = "risk-premium"
    MICROSTRUCTURE = "microstructure"
    NONE = "none"


class Verdict(StrEnum):
    """Research verdict assigned to an evaluated hypothesis."""

    WORKS = "WORKS"
    WEAK = "WEAK"
    DEAD = "DEAD"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class ExecutionStatus(StrEnum):
    """Whether a declared hypothesis could be evaluated."""

    TESTED = "TESTED"
    UNDERPOWERED = "UNDERPOWERED"
    INVALID = "INVALID"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


class DecayClass(StrEnum):
    """Recent trajectory of an effect."""

    STABLE = "STABLE"
    DECAYING = "DECAYING"
    DEAD = "DEAD"
    REGIME_DEPENDENT = "REGIME_DEPENDENT"
    INSUFFICIENT = "INSUFFICIENT"


class GateStatus(StrEnum):
    """Acceptance-gate outcome."""

    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class DataTier(StrEnum):
    """Evidence tier for time-varying security and event data."""

    POINT_IN_TIME = "POINT_IN_TIME"
    PIT_APPROXIMATION = "PIT_APPROXIMATION"
    SURVIVORSHIP_BIASED = "SURVIVORSHIP_BIASED"


class MarketRecordKind(StrEnum):
    """Canonical intraday record kind."""

    MINUTE_BAR = "MINUTE_BAR"
    NBBO = "NBBO"
    TRADE = "TRADE"
    IMBALANCE = "IMBALANCE"
    AUCTION_PRINT = "AUCTION_PRINT"


class CorporateEventKind(StrEnum):
    """Preregistered event taxonomy used by V2 vetoes."""

    EARNINGS = "EARNINGS"
    PRELIMINARY_RESULTS = "PRELIMINARY_RESULTS"
    GUIDANCE = "GUIDANCE"
    TRADING_HALT = "TRADING_HALT"
    DIVIDEND = "DIVIDEND"
    SPLIT = "SPLIT"


class RecommendationState(StrEnum):
    """Persistent paper-recommendation state."""

    PROPOSED = "PROPOSED"
    WAITING = "WAITING"
    CONFIRMED = "CONFIRMED"
    UPDATED = "UPDATED"
    CANCELLED = "CANCELLED"
    ENTERED = "ENTERED"
    EXITED = "EXITED"


class TimingVerdict(StrEnum):
    """Plain-language timing decision."""

    ACT_NOW = "ACT_NOW"
    WAIT_UNTIL = "WAIT_UNTIL"
    WAIT_FOR_TRIGGER = "WAIT_FOR_TRIGGER"
    SKIP = "SKIP"


class OrderType(StrEnum):
    """Paper order type."""

    LIMIT = "LIMIT"
    LOC = "LOC"
    MOC = "MOC"
    MARKET = "MARKET"


@dataclass(frozen=True, slots=True)
class AssetKey:
    """Stable asset identity separate from a mutable ticker."""

    symbol: str
    exchange: str = "US"
    asset_type: str = "equity"


@dataclass(frozen=True, slots=True)
class SecurityIdentity:
    """Permanent security identity independent of ticker history."""

    security_id: str
    issuer_id: str | None = None
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class TickerValidityInterval:
    """Ticker mapping whose knowledge time is distinct from its effective time."""

    security_id: str
    ticker: str
    exchange: str
    valid_from: datetime
    valid_to: datetime | None
    available_at: datetime
    source: str
    fetched_at: datetime
    content_hash: str


@dataclass(frozen=True, slots=True)
class BarRequest:
    """Historical daily-bar request."""

    asset: AssetKey
    start: date
    end: date
    adjusted: bool = True


@dataclass(frozen=True, slots=True)
class SourceCapabilities:
    """Provider capabilities used for explicit feature gating."""

    name: str
    daily: bool = True
    intraday: bool = False
    raw_and_adjusted: bool = False
    corporate_actions: bool = False
    delayed_minutes: int | None = None


@dataclass(frozen=True, slots=True)
class Bar:
    """Canonical OHLCV observation."""

    asset: AssetKey
    event_time: datetime
    available_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: float | None = None
    dividend: float = 0.0
    split_factor: float = 1.0
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class SourceBatch:
    """Immutable response returned by a market-data adapter."""

    source: str
    request: BarRequest
    bars: tuple[Bar, ...]
    fetched_at: datetime
    raw_sha256: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Quote:
    """Timestamped, entitlement-aware quote."""

    asset: AssetKey
    price: float
    provider_time: datetime
    received_at: datetime
    source: str
    delayed_minutes: int | None = None
    halted: bool = False


@dataclass(frozen=True, slots=True)
class MembershipInterval:
    """Point-in-time universe membership interval."""

    asset: AssetKey
    start: date
    end: date | None
    sector: str | None = None
    available_at: datetime | None = None
    security_id: str | None = None
    source: str = "unknown"
    data_tier: DataTier = DataTier.SURVIVORSHIP_BIASED
    fetched_at: datetime | None = None
    content_hash: str | None = None


@dataclass(frozen=True, slots=True)
class CorporateEvent:
    """Vintage-aware corporate event known at a causal decision time."""

    event_id: str
    security_id: str
    kind: CorporateEventKind
    event_time: datetime
    available_at: datetime
    source: str
    revision: str
    fetched_at: datetime
    content_hash: str
    sentiment: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EstimateVintage:
    """One historical consensus estimate vintage."""

    estimate_id: str
    security_id: str
    metric: str
    period_end: date
    value: float
    event_time: datetime
    available_at: datetime
    revision: str
    source: str
    fetched_at: datetime
    content_hash: str


@dataclass(frozen=True, slots=True)
class IntradayMarketRecord:
    """Normalized quote, trade, imbalance, print, or minute bar."""

    security_id: str
    kind: MarketRecordKind
    event_time: datetime
    available_at: datetime
    source: str
    revision: str
    fetched_at: datetime
    content_hash: str
    price: float | None = None
    size: float | None = None
    bid: float | None = None
    ask: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Content identity for a normalized data snapshot."""

    snapshot_id: str
    as_of: date
    created_at: datetime
    source_hashes: Mapping[str, str]
    universe_hash: str
    bias_tier: str
    warnings: tuple[str, ...] = ()


class CausalDataView:
    """Read-only frame restricted to data available at a decision time."""

    def __init__(self, frame: pd.DataFrame, decision_time: datetime) -> None:
        if "available_at" not in frame.columns:
            raise ValueError("frame must contain an available_at column")
        available = pd.to_datetime(frame["available_at"], utc=True)
        decision = pd.Timestamp(decision_time)
        decision = (
            decision.tz_localize("UTC")
            if decision.tzinfo is None
            else decision.tz_convert("UTC")
        )
        if bool((available > decision).any()):
            raise ValueError("future data present in CausalDataView")
        self._frame = frame.copy(deep=False)
        self.decision_time = decision.to_pydatetime()

    @property
    def frame(self) -> pd.DataFrame:
        """Return a defensive shallow copy of the causal frame."""

        return self._frame.copy(deep=False)

    @classmethod
    def as_of(cls, frame: pd.DataFrame, decision_time: datetime) -> CausalDataView:
        """Filter a complete frame to observations known at ``decision_time``."""

        available = pd.to_datetime(frame["available_at"], utc=True)
        decision = pd.Timestamp(decision_time)
        decision = (
            decision.tz_localize("UTC")
            if decision.tzinfo is None
            else decision.tz_convert("UTC")
        )
        return cls(frame.loc[available <= decision].copy(), decision.to_pydatetime())


def _jsonable(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class HypothesisSpec:
    """Canonical, hash-addressed research hypothesis."""

    family: str
    description: str
    predicates: Mapping[str, str]
    direction: Direction
    session: Session
    holding_period: int | str
    rationale: RationaleCategory = RationaleCategory.NONE
    universe: str = "sp500_current"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    placebo_kind: str | None = None

    def canonical_json(self) -> str:
        """Return deterministic JSON used for trial identity."""

        return json.dumps(
            _jsonable(asdict(self)), sort_keys=True, separators=(",", ":")
        )

    @property
    def hypothesis_id(self) -> str:
        """Stable human-sized ID backed by SHA-256."""

        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"{self.family.lower()}-{digest[:16]}"


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Complete numerical evidence for one hypothesis."""

    hypothesis_id: str
    sample_size: int
    gross_mean: float
    net_mean: float
    hac_t: float
    p_value: float
    sharpe: float
    probabilistic_sharpe: float
    deflated_sharpe_probability: float
    hit_rate: float
    max_drawdown: float
    turnover: float
    exposure: float
    skew: float
    kurtosis: float
    mean_ci: tuple[float, float]
    sharpe_ci: tuple[float, float]
    oos_t: float | None = None
    oos_positive_fraction: float | None = None
    stability_score: float | None = None
    pbo: float | None = None
    holdout_mean: float | None = None
    confirmation_difference_bps: float | None = None
    annotations: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VerdictRecord:
    """Final or provisional verdict plus limitations."""

    hypothesis_id: str
    execution_status: ExecutionStatus
    verdict: Verdict | None
    decay: DecayClass
    reasons: tuple[str, ...]
    evidence: EvidenceBundle | None
    provisional: bool
    bias_tier: str = "SURVIVORSHIP_BIASED"


@dataclass(frozen=True, slots=True)
class StackArtifact:
    """Frozen equal-weight composite definition."""

    stack_id: str
    edge_ids: tuple[str, ...]
    cluster_by_edge: Mapping[str, int]
    weights: Mapping[str, float]
    shrunk_means: Mapping[str, float]
    dsr_reliability: float
    promoted: bool = False


@dataclass(frozen=True, slots=True)
class EntryPlan:
    """Concrete, causal paper-entry plan."""

    method: str
    order_type: OrderType
    direction: Direction
    verdict: TimingVerdict
    earliest_execution: datetime
    rationale: str
    limit_price: float | None = None
    trigger: str | None = None
    trigger_value: float | None = None
    expiry_at: datetime | None = None
    expiry_action: str | None = None
    validity_end: datetime | None = None
    stop_price: float | None = None
    suggested_shares: int | None = None
    data_timestamp: datetime | None = None


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Ranked paper recommendation."""

    recommendation_id: str
    asset: AssetKey
    direction: Direction
    confidence: int
    expected_net_return: float
    expected_return_ci: tuple[float, float]
    holding_period: int
    entry_plan: EntryPlan
    driving_edges: tuple[str, ...]
    created_at: datetime
    bias_tier: str = "SURVIVORSHIP_BIASED"
    borrow_verified: bool = False


@dataclass(frozen=True, slots=True)
class AlertEvent:
    """Idempotent logical notification event."""

    event_id: str
    recommendation_id: str
    revision: int
    event_type: str
    message: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class GateResult:
    """Persisted acceptance-gate result."""

    campaign_id: str
    phase: str
    status: GateStatus
    checked_at: datetime
    summary: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CampaignManifest:
    """Reproducibility identity for a research campaign."""

    campaign_id: str
    created_at: datetime
    as_of: date
    holdout_start: date
    config_sha256: str
    data_snapshot_id: str
    source_tree_sha256: str
    lock_sha256: str
    seed: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HoldoutFreezeManifest:
    """Immutable model definition authorized for one holdout evaluation."""

    campaign_id: str
    freeze_id: str
    frozen_at: datetime
    edge_ids: tuple[str, ...]
    specs_sha256: str
    stack_sha256: str
    overlay_sha256: str
    cost_sha256: str
    config_sha256: str
    bars_sha256: str
    universe_sha256: str
    data_manifest_sha256: str
    source_tree_sha256: str
    lock_sha256: str
    model_mapping_sha256: str
    data_snapshot_id: str


@runtime_checkable
class DailyBarSource(Protocol):
    """Pluggable daily-bar provider."""

    capabilities: SourceCapabilities

    async def fetch_bars(self, request: BarRequest) -> SourceBatch:
        """Fetch one complete instrument series."""


@runtime_checkable
class QuoteSource(Protocol):
    """Pluggable delayed or real-time quote provider."""

    async def fetch_quotes(self, assets: Sequence[AssetKey]) -> Sequence[Quote]:
        """Fetch quotes with provider and receipt timestamps."""


@runtime_checkable
class UniverseSource(Protocol):
    """Point-in-time universe provider."""

    async def memberships(self, start: date, end: date) -> Sequence[MembershipInterval]:
        """Return known membership intervals."""


@runtime_checkable
class CorporateEventSource(Protocol):
    """Vintage-aware corporate-event provider."""

    async def fetch_events(
        self, security_ids: Sequence[str], start: datetime, end: datetime
    ) -> Sequence[CorporateEvent]:
        """Return event vintages with explicit availability timestamps."""


@runtime_checkable
class IntradayMarketSource(Protocol):
    """Provider for entitlement-aware intraday records."""

    async def fetch_intraday(
        self, security_ids: Sequence[str], start: datetime, end: datetime
    ) -> Sequence[IntradayMarketRecord]:
        """Return normalized records without weakening requested coverage."""


@runtime_checkable
class Feature(Protocol):
    """Causal feature contract."""

    required_fields: frozenset[str]
    lookback_sessions: int

    def compute(self, view: CausalDataView) -> pd.Series | pd.DataFrame:
        """Compute a feature from data known at the decision time."""


def ensure_fill_after_signal(signal_time: datetime, fill_time: datetime) -> None:
    """Raise when a simulated fill violates the global causal execution rule."""

    if fill_time <= signal_time:
        raise ValueError("fill_time must be strictly later than signal_time")
