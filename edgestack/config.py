"""Typed YAML configuration and deterministic resolution."""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base configuration model that rejects misspelled keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(StrictModel):
    """Local storage roots."""

    root: Path = Path(".")
    raw: Path = Path("data/raw")
    canonical: Path = Path("data/canonical")
    artifacts: Path = Path("artifacts")
    catalog: Path = Path("artifacts/edgestack.sqlite")


class ProviderConfig(StrictModel):
    """Provider order and rate limits."""

    order: tuple[str, ...] = ("tiingo", "stooq", "yfinance")
    tiingo_api_key_env: str = "TIINGO_API_KEY"
    finnhub_api_key_env: str = "FINNHUB_API_KEY"
    alpha_vantage_api_key_env: str = "ALPHAVANTAGE_API_KEY"
    stooq_bulk_archive: Path | None = None
    stooq_bulk_sha256: str | None = None
    timeout_seconds: float = 30.0
    max_attempts: int = 6
    concurrency: int = 4

    @field_validator("stooq_bulk_sha256")
    @classmethod
    def validate_stooq_bulk_sha256(cls, value: str | None) -> str | None:
        """Normalize and validate an optional pinned Stooq archive digest."""

        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("stooq_bulk_sha256 must be a hexadecimal SHA-256 digest")
        return normalized

    @model_validator(mode="after")
    def validate_stooq_bulk_pair(self) -> ProviderConfig:
        """Require archive paths and their immutable hashes together."""

        if (self.stooq_bulk_archive is None) != (self.stooq_bulk_sha256 is None):
            raise ValueError(
                "stooq_bulk_archive and stooq_bulk_sha256 must be configured together"
            )
        return self


class DataConfig(StrictModel):
    """Historical-data and QA policy."""

    # The frozen Monday-effect replication requires a pre-1975 benchmark.
    # Individual listing eligibility still begins at each asset's first bar.
    start: date = date(1960, 1, 1)
    providers: ProviderConfig = Field(default_factory=ProviderConfig)
    etfs: tuple[str, ...] = (
        "SPY",
        "QQQ",
        "IWM",
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLY",
        "XLI",
    )
    reconciliation_tickers: tuple[str, ...] = (
        "SPY",
        "QQQ",
        "IWM",
        "AAPL",
        "MSFT",
        "AMZN",
        "NVDA",
        "JPM",
        "JNJ",
        "XOM",
        "WMT",
        "PG",
        "KO",
        "PEP",
        "IBM",
        "INTC",
        "CSCO",
        "HD",
        "CAT",
        "BA",
    )
    reconciliation_method: Literal[
        "rebased_total_return", "action_stratified_returns"
    ] = "rebased_total_return"
    reconciliation_tolerance: float = 0.005
    reconciliation_required_fraction: float = 0.99
    missing_bar_max_fraction: float = 0.001
    outlier_sigma: float = 10.0
    stale_sessions: int = 3
    holdout_years: int = 3
    # Reconstruct best-effort PIT membership intervals from the Wikipedia
    # change log and fetch delisted-name history from the bulk archive. The
    # result stays tiered PIT_APPROXIMATION — never licensed PIT evidence.
    universe_pit: bool = False


class GridConfig(StrictModel):
    """Hypothesis grammar."""

    max_interaction_order: Literal[1, 2] = 2
    directions: tuple[str, ...] = ("LONG", "SHORT")
    close_holding_periods: tuple[int, ...] = (1, 3, 5, 21)
    sessions: tuple[str, ...] = ("close_to_close", "overnight", "intraday")
    placebo_replicates: int = 2
    include_cross_sectional: bool = True
    min_observations: int = 100
    # Opt-in post-original families (quarter/month-end windows, Amihud,
    # MAX-lottery, overnight/intraday gap, ETF relative reversal). Off by
    # default so pre-existing campaign grids stay identical.
    extended_families: bool = False


class StatsConfig(StrictModel):
    """Frozen statistical thresholds."""

    seed: int = 42
    hard_t: float = 3.0
    fdr_q: float = 0.05
    dsr_probability: float = 0.95
    bootstrap_reps: int = 2_000
    finalist_bootstrap_reps: int = 10_000
    survivor_fraction_max: float = 0.05
    placebo_survival_max: float = 0.005
    # Family-wide SPA/Reality-Check significance level; 0.05 preserves the
    # original hardcoded behavior exactly.
    family_alpha: float = 0.05
    # Per-survivor truncation-invariance and extra-lag causality checks run
    # after the discovery gauntlet; frozen campaigns never re-run discovery.
    survivor_causality_checks: bool = True

    @model_validator(mode="after")
    def validate_family_alpha(self) -> StatsConfig:
        """Keep the family-test level a genuine significance level."""

        if not 0.0 < self.family_alpha < 1.0:
            raise ValueError("family_alpha must lie strictly between zero and one")
        return self


class HoldoutGateConfig(StrictModel):
    """Versioned final-holdout promotion evaluator.

    ``SIGN_V1`` preserves the original strictly-positive-mean gate exactly.
    ``CI_V2`` additionally requires the stationary-bootstrap confidence
    interval lower bound of every edge and of the composite to be strictly
    positive, and emits report-only regime stratification of the holdout
    streams. The version is bound into the freeze manifest at score time and
    cannot change afterwards; existing sealed holdouts stay sealed under the
    version they were frozen with.
    """

    evaluator_version: Literal["SIGN_V1", "CI_V2"] = "SIGN_V1"


class EvidenceProtocolConfig(StrictModel):
    """Versioned interpretation of replication and discovery evidence.

    ``FROZEN_V1`` preserves the original all-six replication gate.  The
    literature-informed protocol is deliberately a new version: it may treat
    an executed empirical replication miss as evidence rather than a software
    failure, but compensates with stronger discovery hurdles and family-wise
    error control.  This prevents an observed miss from silently mutating the
    original campaign.
    """

    version: Literal["FROZEN_V1", "LITERATURE_V2"] = "FROZEN_V1"
    replication_policy: Literal[
        "ALL_SIX_EMPIRICAL", "EXECUTION_WITH_EMPIRICAL_DIAGNOSTICS"
    ] = "ALL_SIX_EMPIRICAL"
    time_series_t_threshold: float = 3.0
    cross_sectional_t_threshold: float = 3.0
    require_romano_wolf: bool = False
    romano_wolf_alpha: float = 0.05
    sharpe_interval: Literal["PERCENTILE_STATIONARY", "STUDENTIZED_STATIONARY"] = (
        "PERCENTILE_STATIONARY"
    )
    capacity_capital_multipliers: tuple[float, ...] = (1.0,)
    revision_context: Literal[
        "ORIGINAL_PREREGISTRATION", "POST_REPLICATION_PRE_DISCOVERY"
    ] = "ORIGINAL_PREREGISTRATION"

    @model_validator(mode="after")
    def validate_literature_protocol(self) -> EvidenceProtocolConfig:
        """Make the V2 label imply the complete, non-optional safeguard set."""

        if not 0.0 < self.romano_wolf_alpha < 1.0:
            raise ValueError("romano_wolf_alpha must lie strictly between zero and one")
        if (
            self.time_series_t_threshold <= 0.0
            or self.cross_sectional_t_threshold <= 0.0
        ):
            raise ValueError("discovery t-statistic thresholds must be positive")
        if (
            not self.capacity_capital_multipliers
            or any(value <= 0.0 for value in self.capacity_capital_multipliers)
            or tuple(sorted(set(self.capacity_capital_multipliers)))
            != self.capacity_capital_multipliers
        ):
            raise ValueError(
                "capacity_capital_multipliers must be unique, positive, and sorted"
            )
        if self.version == "LITERATURE_V2":
            requirements = (
                self.replication_policy == "EXECUTION_WITH_EMPIRICAL_DIAGNOSTICS",
                self.time_series_t_threshold >= 3.8,
                self.cross_sectional_t_threshold >= 3.4,
                self.require_romano_wolf,
                self.sharpe_interval == "STUDENTIZED_STATIONARY",
                self.revision_context == "POST_REPLICATION_PRE_DISCOVERY",
            )
            if not all(requirements):
                raise ValueError(
                    "LITERATURE_V2 requires diagnostic replication, t>=3.8/3.4, "
                    "Romano-Wolf, studentized Sharpe inference, and an explicit "
                    "post-replication/pre-discovery revision label"
                )
        return self


class ValidationConfig(StrictModel):
    """Out-of-sample and stability policy."""

    min_train_years: int = 5
    test_years: int = 1
    step_years: int = 1
    oos_t: float = 2.0
    oos_positive_fraction: float = 0.5
    stability_min: float = 0.75
    cpcv_groups: int = 6
    cpcv_test_groups: int = 2
    purge_sessions: int = 21
    embargo_sessions: int = 21
    pbo_max: float = 0.20
    rolling_years: int = 5


class CostConfig(StrictModel):
    """Baseline $100k retail research-cost model."""

    capital: float = 100_000.0
    commission_per_side: float = 0.0
    etf_full_spread_bps: float = 1.0
    equity_full_spread_bps: float = 3.0
    base_slippage_bps: float = 1.0
    impact_coefficient_bps: float = 10.0
    max_impact_bps: float = 50.0
    easy_borrow_annual: float = 0.003
    turnover_penalty_bps: float = 1.0
    sensitivity_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    # MEASURED_HL_FLOOR_V2 overlays per-name monthly high-low spread estimates
    # floored at the assumed baseline; measured values can only RAISE costs.
    spread_source: Literal["ASSUMED_V1", "MEASURED_HL_FLOOR_V2"] = "ASSUMED_V1"


class EntryTimingConfig(StrictModel):
    """Pre-registered timing neighborhoods."""

    rsi2_thresholds: tuple[int, ...] = (5, 10, 15)
    bollinger_thresholds: tuple[float, ...] = (0.1, 0.2, 0.3)
    expiry_bars: tuple[int, ...] = (3, 5, 7)
    breakout_windows: tuple[int, ...] = (20, 63, 252)
    atr_multipliers: tuple[float, ...] = (1.5, 2.0, 2.5)
    ma_window: int = 200
    vix_low: float = 15.0
    vix_high: float = 25.0
    plateau_within: float = 0.20
    recency_weighting: bool = False


class ReversalResearchConfig(StrictModel):
    """Opt-in, selection-aware short-horizon reversal research protocol.

    This protocol is deliberately separate from the frozen V1/V2 campaign
    grids.  Enabling it declares every breadth and signal variant up front so
    choosing a five-name portfolio cannot silently change the strategy after
    results are observed.
    """

    enabled: bool = False
    study_version: Literal["v3"] = "v3"
    top_k: tuple[int, ...] = (3, 5, 10, 20, 50)
    variants: tuple[Literal["raw", "sector_neutral", "market_sector_residual"], ...] = (
        "raw",
        "sector_neutral",
        "market_sector_residual",
    )
    lookback_sessions: int = 5
    holding_sessions: int = 5
    beta_window: int = 252
    beta_min_observations: int = 126
    residual_vol_window: int = 20
    decision_time: str = "15:45"
    loc_atr_fraction: float = 0.25
    stop_atr_multiple: float = 2.0
    event_exclusion_sessions: int = 5
    preentry_reversal_atr_max: float = 1.0
    require_point_in_time_universe: bool = True
    allow_survivorship_biased_diagnostic: bool = False
    gpu_devices: tuple[int, ...] = (0, 1)
    # MEASURED_HL_FLOOR_V2 prices the grid with per-name monthly high-low
    # spread estimates floored at the assumed baseline (costs only tighten).
    spread_source: Literal["ASSUMED_V1", "MEASURED_HL_FLOOR_V2"] = "ASSUMED_V1"

    @field_validator("decision_time")
    @classmethod
    def validate_decision_time(cls, value: str) -> str:
        """Require an explicit regular-session HH:MM decision timestamp."""

        import datetime as dt

        parsed = dt.datetime.strptime(value, "%H:%M").time()
        if not dt.time(9, 30) <= parsed < dt.time(16, 0):
            raise ValueError("decision_time must be in the 09:30-15:59 ET session")
        return value

    @model_validator(mode="after")
    def validate_protocol(self) -> ReversalResearchConfig:
        """Reject grids that could create hidden or nonsensical trials."""

        if (
            not self.top_k
            or tuple(sorted(set(self.top_k))) != self.top_k
            or any(value < 1 for value in self.top_k)
        ):
            raise ValueError("top_k must contain unique, positive, sorted values")
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise ValueError("variants must be non-empty and unique")
        if self.lookback_sessions < 1 or self.holding_sessions < 1:
            raise ValueError("lookback and holding sessions must be positive")
        if not 2 <= self.beta_min_observations <= self.beta_window:
            raise ValueError("beta_min_observations must be in [2, beta_window]")
        if self.residual_vol_window < 2:
            raise ValueError("residual_vol_window must be at least two")
        if (
            self.loc_atr_fraction <= 0.0
            or self.stop_atr_multiple <= 0.0
            or self.preentry_reversal_atr_max <= 0.0
            or self.event_exclusion_sessions < 0
        ):
            raise ValueError("execution thresholds must be positive and causal")
        if not self.gpu_devices or any(device < 0 for device in self.gpu_devices):
            raise ValueError("gpu_devices must contain non-negative device indices")
        if len(set(self.gpu_devices)) != len(self.gpu_devices):
            raise ValueError("gpu_devices must be unique")
        return self


class LiveConfig(StrictModel):
    """Paper-assistant scheduling and risk settings."""

    enabled: bool = False
    timezone: str = "America/New_York"
    scan_time: str = "08:30"
    poll_minutes: int = 15
    top_n: int = 5
    minimum_confidence: int = 60
    capital: float = 100_000.0
    target_risk_fraction: float = 0.005
    max_position_fraction: float = 0.10
    atr_stop_multiple: float = 2.0
    allow_unverified_paper_shorts: bool = True
    channels: tuple[str, ...] = ("console",)

    @field_validator("scan_time")
    @classmethod
    def validate_scan_time(cls, value: str) -> str:
        """Validate a simple 24-hour HH:MM value."""

        import datetime as dt

        dt.datetime.strptime(value, "%H:%M")
        return value


class EdgeStackConfig(StrictModel):
    """Root EdgeStack configuration."""

    profile: Literal["smoke", "full"] = "smoke"
    as_of: date | None = None
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    stats: StatsConfig = Field(default_factory=StatsConfig)
    protocol: EvidenceProtocolConfig = Field(default_factory=EvidenceProtocolConfig)
    holdout_gate: HoldoutGateConfig = Field(default_factory=HoldoutGateConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    entrytiming: EntryTimingConfig = Field(default_factory=EntryTimingConfig)
    reversal: ReversalResearchConfig = Field(default_factory=ReversalResearchConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> EdgeStackConfig:
        """Reject internally inconsistent threshold settings."""

        if self.validation.embargo_sessions < max(self.grid.close_holding_periods):
            raise ValueError("embargo_sessions must cover the longest holding period")
        if self.live.minimum_confidence not in range(101):
            raise ValueError("minimum_confidence must be between 0 and 100")
        return self


_ENV_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> EdgeStackConfig:
    """Load and validate an EdgeStack YAML configuration."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return EdgeStackConfig.model_validate(_expand_env(payload))


def dump_resolved_config(config: EdgeStackConfig) -> str:
    """Serialize a resolved configuration deterministically."""

    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True)
