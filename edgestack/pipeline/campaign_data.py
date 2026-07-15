"""Campaign data acquisition and deterministic smoke fixtures.

The smoke profile is deliberately synthetic and non-promotable.  It exists to
exercise every state transition without making a network response part of the
test oracle.  The full profile uses the configured provider chain and retains
all immutable raw responses through :class:`~edgestack.data.cache.DataCache`.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.cache import DataCache
from edgestack.data.calendars import FOMCCalendarSource, NYSECalendar
from edgestack.data.factors import (
    VIXCLS,
    FREDCSVSource,
    KenFrenchDailyFactorsSource,
    ReferenceDataCache,
)
from edgestack.data.quality import (
    CorrectionRecord,
    QAReport,
    ReconciliationResult,
    audit_survivorship,
    causal_winsorize_prices,
    reconcile_action_stratified_returns,
    reconcile_adjusted_series,
    run_quality_audit,
    write_correction_log,
)
from edgestack.data.sources import (
    DailyBarSource,
    FallbackDailyBarSource,
    SourceBatch,
    StooqBulkArchiveDailyBarSource,
    StooqDailyBarSource,
    TiingoDailyBarSource,
    YahooDailyBarSource,
    bars_to_frame,
)
from edgestack.data.universe import WikipediaSP500UniverseSource
from edgestack.models import AssetKey, BarRequest, MembershipInterval
from edgestack.provenance import canonical_sha256

NEW_YORK = ZoneInfo("America/New_York")
_MINIMUM_HISTORY_YEARS = 19.5


@dataclass(frozen=True, slots=True)
class FixedSymbolGateEvidence:
    """Per-symbol eligible and cross-provider history evidence."""

    symbol: str
    required_start: date
    required_end: date
    expected_sessions: int
    observed_sessions: int
    observed_coverage_fraction: float
    observed_span_years: float
    common_sessions: int
    common_coverage_fraction: float
    common_span_years: float
    agreement_fraction: float
    history_pass: bool
    reconciliation_pass: bool
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class _CommonProviderEvidence:
    symbol: str
    sessions: tuple[pd.Timestamp, ...]
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class IngestedCampaignData:
    """Normalized inputs and complete acquisition evidence for one campaign."""

    bars: pd.DataFrame
    factors: pd.DataFrame
    fomc_dates: pd.DatetimeIndex
    memberships: tuple[MembershipInterval, ...]
    qa: QAReport
    snapshot_id: str
    source_hashes: Mapping[str, str]
    warnings: tuple[str, ...]
    failures: Mapping[str, str]
    calendar_match: bool
    long_history_pass: bool
    reconciliation_pass: bool
    all_symbols_fetched: bool
    non_promotable: bool
    fixed_symbol_evidence: tuple[FixedSymbolGateEvidence, ...] = ()
    correction_log_sha256: str | None = None
    correction_count: int = 0

    @property
    def passed(self) -> bool:
        """Return the profile-appropriate data gate decision."""

        if self.non_promotable:
            return bool(
                not self.bars.empty
                and self.calendar_match
                and self.qa.aggregate_missing_fraction < self.qa.missing_bar_threshold
            )
        return bool(
            self.qa.passed
            and self.calendar_match
            and self.long_history_pass
            and self.reconciliation_pass
            and self.all_symbols_fetched
            # VIX is an optional overlay input: unavailability disables that
            # neighborhood but must not invalidate otherwise complete daily bars.
            and not {
                key: value
                for key, value in self.failures.items()
                if key != "fred_vixcls"
            }
        )

    def evidence(self) -> dict[str, Any]:
        """Return JSON-ready data-gate evidence."""

        has_bars = not self.bars.empty and {"symbol", "session"}.issubset(
            self.bars.columns
        )
        if has_bars:
            sessions = pd.to_datetime(self.bars["session"])
            start_value: str | None = sessions.min().date().isoformat()
            end_value: str | None = sessions.max().date().isoformat()
            symbol_count = int(self.bars["symbol"].nunique())
        else:
            start_value = None
            end_value = None
            symbol_count = 0
        vix_available = {
            "VIXCLS",
            "VIXCLS__event_time",
            "VIXCLS__available_at",
        }.issubset(self.factors.columns)
        vix_observations = (
            int(pd.to_numeric(self.factors["VIXCLS"], errors="coerce").notna().sum())
            if "VIXCLS" in self.factors
            else 0
        )
        return {
            "profile_scope": (
                "SYNTHETIC_SMOKE_NON_PROMOTABLE"
                if self.non_promotable
                else "FULL_EMPIRICAL"
            ),
            "snapshot_id": self.snapshot_id,
            "symbols": symbol_count,
            "rows": len(self.bars),
            "start": start_value,
            "end": end_value,
            "aggregate_missing_fraction": self.qa.aggregate_missing_fraction,
            "missing_threshold": self.qa.missing_bar_threshold,
            "calendar_match": self.calendar_match,
            "long_history_20_symbols": self.long_history_pass,
            "reconciliation_pass": self.reconciliation_pass,
            "all_symbols_fetched": self.all_symbols_fetched,
            "reconciliations": [asdict(item) for item in self.qa.reconciliations],
            "fixed_symbol_evidence": [
                asdict(item) for item in self.fixed_symbol_evidence
            ],
            "correction_log_sha256": self.correction_log_sha256,
            "correction_count": self.correction_count,
            "survivorship": (
                asdict(self.qa.survivorship)
                if self.qa.survivorship is not None
                else None
            ),
            "source_hashes": dict(self.source_hashes),
            "vix": {
                "status": "AVAILABLE" if vix_available else "DATA_UNAVAILABLE",
                "series": "VIXCLS",
                "observations": vix_observations,
                "event_time_column": ("VIXCLS__event_time" if vix_available else None),
                "available_at_column": (
                    "VIXCLS__available_at" if vix_available else None
                ),
                "source_snapshot": self.source_hashes.get("fred_vixcls"),
                "failure": self.failures.get("fred_vixcls"),
            },
            "warnings": list(self.warnings),
            "failures": dict(self.failures),
        }


def acquire_campaign_data(
    config: EdgeStackConfig,
    *,
    as_of: date,
    cache: DataCache,
) -> IngestedCampaignData:
    """Acquire a deterministic smoke fixture or the full configured universe."""

    if config.profile == "smoke":
        return deterministic_smoke_data(config, as_of=as_of)
    return asyncio.run(_acquire_full(config, as_of=as_of, cache=cache))


def deterministic_smoke_data(
    config: EdgeStackConfig, *, as_of: date
) -> IngestedCampaignData:
    """Build a reproducible, explicitly synthetic multi-decade campaign fixture."""

    calendar = NYSECalendar()
    start = config.data.start
    if as_of < start:
        raise ValueError("smoke as_of precedes the frozen configured data start")
    sessions = calendar.sessions(start, as_of)
    if sessions.empty:
        raise ValueError("smoke date interval contains no NYSE sessions")
    fixture_timestamp = datetime.combine(as_of, time(23, 59, 59, tzinfo=UTC))
    rng = np.random.default_rng(config.stats.seed)
    count = len(sessions)
    weekday = sessions.dayofweek.to_numpy()
    session_frame = pd.DataFrame(index=sessions)
    session_frame["position"] = np.arange(count)
    within_month = session_frame.groupby([sessions.year, sessions.month], sort=False)[
        "position"
    ].transform(lambda values: np.arange(len(values)))
    from_end = session_frame.groupby([sessions.year, sessions.month], sort=False)[
        "position"
    ].transform(lambda values: np.arange(len(values))[::-1])
    tom = (within_month <= 2).to_numpy() | (from_end <= 0).to_numpy()

    # Eight scheduled meetings per year, always on an observed Wednesday.
    fomc: list[pd.Timestamp] = []
    for year in range(start.year, as_of.year + 1):
        candidates = sessions[(sessions.year == year) & (sessions.dayofweek == 2)]
        if len(candidates):
            locations = np.linspace(
                0, len(candidates) - 1, min(8, len(candidates)), dtype=int
            )
            fomc.extend(candidates[locations])
    fomc_index = pd.DatetimeIndex(sorted(set(fomc)))

    market = rng.normal(0.00018, 0.0075, count)
    market += np.where(tom, 0.0014, -0.00008)
    pre_1975_monday = (sessions < pd.Timestamp("1975-01-01")) & (weekday == 0)
    market[pre_1975_monday] -= 0.0035
    historical_fomc = (
        sessions.isin(fomc_index)
        & (sessions >= pd.Timestamp("1994-01-01"))
        & (sessions <= pd.Timestamp("2013-12-31"))
    )
    market[historical_fomc] += 0.006
    market = np.clip(market, -0.12, 0.12)

    symbols = ("SPY", "QQQ", "IWM", "AAPL", "MSFT", "XLF", "XLK", "XLE")
    sectors = {
        "SPY": "ETF",
        "QQQ": "ETF",
        "IWM": "ETF",
        "AAPL": "Information Technology",
        "MSFT": "Information Technology",
        "XLF": "ETF",
        "XLK": "ETF",
        "XLE": "ETF",
    }
    rows: list[dict[str, Any]] = []
    for symbol_number, symbol in enumerate(symbols):
        local = market + rng.normal(0.0, 0.003 + symbol_number * 0.0002, count)
        # Make the cross-section non-degenerate while retaining common effects.
        local += (symbol_number - len(symbols) / 2) * 0.00001
        close = np.empty(count, dtype=float)
        open_ = np.empty(count, dtype=float)
        prior = 25.0 + symbol_number * 7.0
        for index in range(count):
            overnight = 0.86 * local[index]
            intraday = 0.14 * local[index]
            open_[index] = prior * math.exp(overnight)
            close[index] = open_[index] * math.exp(intraday)
            prior = close[index]
        spread = np.maximum(close * 0.006, 0.01)
        volume = 5_000_000.0 + symbol_number * 350_000.0
        for index, session in enumerate(sessions):
            close_time = datetime.combine(
                session.date(), time(16, 0), tzinfo=NEW_YORK
            ).astimezone(UTC)
            rows.append(
                {
                    "symbol": symbol,
                    "exchange": "US",
                    "asset_type": "etf" if sectors[symbol] == "ETF" else "equity",
                    "session": session,
                    "event_time": close_time,
                    "available_at": close_time + timedelta(minutes=5),
                    "open": open_[index],
                    "high": max(open_[index], close[index]) + spread[index],
                    "low": max(min(open_[index], close[index]) - spread[index], 0.01),
                    "close": close[index],
                    "adjusted_close": close[index],
                    "volume": volume + float((index % 20) * 10_000),
                    "dividend": 0.0,
                    "split_factor": 1.0,
                    "source": "synthetic_smoke",
                }
            )
    bars = pd.DataFrame(rows).sort_values(["symbol", "session"]).reset_index(drop=True)
    factors = pd.DataFrame(
        {
            "session": sessions,
            "mkt_rf": market,
            "smb": np.zeros(count),
            "hml": np.zeros(count),
            "rf": np.zeros(count),
            "market_return": market,
        }
    )
    memberships = tuple(
        MembershipInterval(
            AssetKey(
                symbol,
                asset_type="etf" if sectors[symbol] == "ETF" else "equity",
            ),
            start,
            None,
            sectors[symbol],
            fixture_timestamp,
        )
        for symbol in symbols
    )
    survivorship = audit_survivorship(symbols, symbols, point_in_time=False)
    qa = run_quality_audit(
        bars,
        nyse=calendar,
        stale_sessions=config.data.stale_sessions,
        outlier_sigma=config.data.outlier_sigma,
        survivorship=survivorship,
        missing_bar_threshold=config.data.missing_bar_max_fraction,
    )
    qa = replace(qa, created_at=fixture_timestamp)
    calendar.assert_reference_match(start, as_of)
    content_hashes = {
        "bars": _frame_content_sha256(bars),
        "factors": _frame_content_sha256(factors),
        "fomc": canonical_sha256(
            [session.date().isoformat() for session in fomc_index]
        ),
        "memberships": canonical_sha256([asdict(item) for item in memberships]),
    }
    identity = {
        "kind": "deterministic_smoke_v2",
        "seed": config.stats.seed,
        "start": start.isoformat(),
        "as_of": as_of.isoformat(),
        "rows": len(bars),
        "symbols": symbols,
        "content_hashes": content_hashes,
    }
    snapshot_id = canonical_sha256(identity)
    return IngestedCampaignData(
        bars,
        factors,
        fomc_index,
        memberships,
        qa,
        snapshot_id,
        {
            "synthetic_smoke": snapshot_id,
            **{f"synthetic_{key}": value for key, value in content_hashes.items()},
        },
        (
            "SYNTHETIC_SMOKE_NON_PROMOTABLE: engineering fixture; no empirical "
            "claim or live promotion is permitted.",
            "SURVIVORSHIP_BIASED",
        ),
        {},
        True,
        True,
        True,
        True,
        True,
    )


async def _acquire_full(
    config: EdgeStackConfig, *, as_of: date, cache: DataCache
) -> IngestedCampaignData:
    calendar = NYSECalendar()
    universe_source = WikipediaSP500UniverseSource(
        include_etfs=True,
        reconstruct_history=False,
        raw_sink=cache.raw,
        timeout=config.data.providers.timeout_seconds,
    )
    memberships = await universe_source.memberships(config.data.start, as_of)
    assets = tuple(dict.fromkeys(item.asset for item in memberships))
    sources = _provider_chain(config, cache)
    chain = FallbackDailyBarSource(sources)
    requests = tuple(
        BarRequest(asset, config.data.start, as_of, adjusted=True) for asset in assets
    )
    snapshots_by_asset = {
        request.asset: cached
        for request in requests
        if (cached := cache.exact_snapshot(request)) is not None
    }
    pending_requests = tuple(
        request for request in requests if request.asset not in snapshots_by_asset
    )
    batches, failures = await _fetch_individually(
        chain, pending_requests, concurrency=config.data.providers.concurrency
    )
    for batch in batches:
        snapshots_by_asset[batch.request.asset] = cache.store_batch(batch)
    snapshots = [
        snapshots_by_asset[request.asset]
        for request in requests
        if request.asset in snapshots_by_asset
    ]
    frames = [
        _canonical_adjusted_frame(
            cache.read_frame(item.snapshot_id, representation="adjusted")
        )
        for item in snapshots
    ]
    bars = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["symbol", "session"], kind="stable")
        .reset_index(drop=True)
        if frames
        else pd.DataFrame()
    )
    correction_records: tuple[CorrectionRecord, ...] = ()
    correction_log_sha256: str | None = None
    if not bars.empty:
        bars, correction_records = _apply_causal_corrections(
            bars, sigma=config.data.outlier_sigma
        )

    reference_cache = ReferenceDataCache(cache.canonical_root / "reference")
    reference_warnings: list[str] = []
    reused_count = len(requests) - len(pending_requests)
    if reused_count:
        reference_warnings.append(
            f"EXACT_IMMUTABLE_CACHE_REUSE: {reused_count} whole-series snapshots "
            "matched asset/start/end/adjustment identity."
        )
    source_hashes: dict[str, str] = {
        item.asset.symbol: item.raw_sha256 for item in snapshots
    }
    source_hashes["canonical_research_bars"] = _frame_content_sha256(bars)
    bulk_stooq = next(
        (
            source
            for source in sources
            if isinstance(source, StooqBulkArchiveDailyBarSource)
        ),
        None,
    )
    if bulk_stooq is not None:
        source_hashes["stooq_bulk_archive"] = bulk_stooq.archive_sha256
        reference_warnings.extend(
            (
                "STOOQ_BULK_ARCHIVE: reconciliation uses exact members from "
                f"archive SHA-256 {bulk_stooq.archive_sha256}.",
                "USER_SUPPLIED_SOURCE_ARCHIVE: archive origin is operator-attested; "
                "Stooq does not provide a cryptographic publisher signature.",
            )
        )
    if config.data.reconciliation_method == "action_stratified_returns":
        reference_warnings.append(
            "SINGLE_SOURCE_ACTIONS: Stooq/Yahoo raw price returns are reconciled "
            "on non-action sessions; Yahoo alone supplies dividends, splits, and "
            "the canonical adjusted total-return series."
        )
    factors = pd.DataFrame()
    fomc_dates = pd.DatetimeIndex([])
    try:
        french = await KenFrenchDailyFactorsSource(
            raw_sink=cache.raw,
            timeout=config.data.providers.timeout_seconds,
            max_attempts=config.data.providers.max_attempts,
        ).fetch(config.data.start, as_of)
        french_id = reference_cache.store(french)
        factors = french.frame
        source_hashes["ken_french"] = french_id
        reference_warnings.extend(french.warnings)
    except Exception as error:  # provider diagnostics must survive a stopped run
        failures["ken_french"] = f"{type(error).__name__}: {error}"
    try:
        vix = await FREDCSVSource(
            raw_sink=cache.raw,
            timeout=config.data.providers.timeout_seconds,
            max_attempts=config.data.providers.max_attempts,
        ).fetch_series((VIXCLS,), config.data.start, as_of)
        vix_id = reference_cache.store(vix)
        factors = _merge_reference_factors(factors, vix.frame)
        source_hashes["fred_vixcls"] = vix_id
        reference_warnings.extend(vix.warnings)
    except Exception as error:  # no-key reference failures remain diagnosable
        failures["fred_vixcls"] = f"DATA_UNAVAILABLE: {type(error).__name__}: {error}"
    try:
        meetings = await FOMCCalendarSource(
            raw_sink=cache.raw,
            timeout=config.data.providers.timeout_seconds,
        ).fetch_meetings(max(config.data.start, date(1994, 1, 1)), as_of)
        fomc_dates = pd.DatetimeIndex([item.end for item in meetings])
        source_hashes["fomc"] = canonical_sha256([asdict(item) for item in meetings])
    except Exception as error:
        failures["fomc"] = f"{type(error).__name__}: {error}"

    try:
        correction_log_sha256 = _persist_correction_evidence(
            correction_records,
            cache.canonical_root / "corrections",
        )
        source_hashes["correction_log"] = correction_log_sha256
        if correction_records:
            reference_warnings.append(
                f"CAUSAL_OUTLIER_CORRECTIONS_APPLIED: {len(correction_records)} "
                "research closes were winsorized; source closes remain unchanged and "
                f"the immutable correction log is {correction_log_sha256}."
            )
    except Exception as error:
        failures["correction_log"] = f"{type(error).__name__}: {error}"

    reconciliations, common_evidence, reconciliation_failures = (
        await _reconcile_fixed_symbols(config, cache, as_of, bars)
    )
    failures.update(reconciliation_failures)
    available = tuple(sorted(bars["symbol"].unique())) if not bars.empty else ()
    intended = tuple(item.symbol for item in assets)
    survivorship = audit_survivorship(intended, available, point_in_time=False)
    if bars.empty:
        # A structural empty report is preferable to an unhelpful groupby error.
        qa = QAReport(
            datetime.now(UTC),
            (),
            tuple(reconciliations),
            survivorship,
            config.data.missing_bar_max_fraction,
        )
    else:
        qa = run_quality_audit(
            bars,
            nyse=calendar,
            stale_sessions=config.data.stale_sessions,
            outlier_sigma=config.data.outlier_sigma,
            reconciliations=reconciliations,
            survivorship=survivorship,
            missing_bar_threshold=config.data.missing_bar_max_fraction,
        )
    try:
        calendar.assert_reference_match(config.data.start, as_of)
        calendar_match = True
    except AssertionError:
        calendar_match = False
    fixed_symbol_evidence = _fixed_symbol_gate_evidence(
        config,
        as_of=as_of,
        bars=bars,
        reconciliations=reconciliations,
        common_evidence=common_evidence,
        failures=failures,
    )
    long_history = len(fixed_symbol_evidence) == len(
        config.data.reconciliation_tickers
    ) and all(item.history_pass for item in fixed_symbol_evidence)
    reconciliation_pass = len(fixed_symbol_evidence) == len(
        config.data.reconciliation_tickers
    ) and all(item.reconciliation_pass for item in fixed_symbol_evidence)
    snapshot_id = canonical_sha256(
        {
            "as_of": as_of,
            "universe": source_hashes.get(
                "wikipedia",
                (
                    universe_source.last_snapshot.source_sha256
                    if universe_source.last_snapshot
                    else ""
                ),
            ),
            "snapshots": sorted(item.snapshot_id for item in snapshots),
            "references": source_hashes,
            "fixed_symbol_evidence": [asdict(item) for item in fixed_symbol_evidence],
        }
    )
    if universe_source.last_snapshot is not None:
        source_hashes["wikipedia"] = universe_source.last_snapshot.source_sha256
        reference_warnings.extend(universe_source.last_snapshot.warnings)
    await _close_sources(sources)
    return IngestedCampaignData(
        bars,
        factors,
        fomc_dates,
        memberships,
        qa,
        snapshot_id,
        source_hashes,
        tuple(reference_warnings),
        failures,
        calendar_match,
        long_history,
        reconciliation_pass,
        not survivorship.missing_assets,
        False,
        fixed_symbol_evidence,
        correction_log_sha256,
        len(correction_records),
    )


def _merge_reference_factors(
    existing: pd.DataFrame, additional: pd.DataFrame
) -> pd.DataFrame:
    """Outer-join availability-aware reference observations by session.

    Reference sources may begin on different dates.  The outer join retains the
    complete benchmark history while preserving each source's own event and
    publication timestamps for later causal joins.
    """

    if additional.empty:
        raise ValueError("additional reference frame cannot be empty")
    if "session" not in additional:
        raise ValueError("additional reference frame is missing session")
    if additional["session"].duplicated().any():
        raise ValueError("additional reference frame has duplicate sessions")
    if existing.empty:
        return additional.sort_values("session", kind="stable").reset_index(drop=True)
    if "session" not in existing:
        raise ValueError("existing reference frame is missing session")
    if existing["session"].duplicated().any():
        raise ValueError("existing reference frame has duplicate sessions")
    overlap = (set(existing.columns) & set(additional.columns)) - {"session"}
    if overlap:
        raise ValueError(
            "reference frames have ambiguous overlapping columns: "
            + ", ".join(sorted(overlap))
        )
    return (
        existing.merge(additional, on="session", how="outer", validate="one_to_one")
        .sort_values("session", kind="stable")
        .reset_index(drop=True)
    )


def _canonical_adjusted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose adjusted total-return close under the canonical bar contract."""

    result = frame.copy()
    if "close" in result and "adjusted_close" not in result:
        result["adjusted_close"] = result["close"]
    return result


def _stooq_source(config: EdgeStackConfig, cache: DataCache) -> DailyBarSource:
    archive = config.data.providers.stooq_bulk_archive
    if archive is not None:
        expected_sha256 = config.data.providers.stooq_bulk_sha256
        if expected_sha256 is None:  # protected by typed-config validation
            raise ValueError("Stooq bulk archive requires its pinned SHA-256")
        return StooqBulkArchiveDailyBarSource(
            archive,
            expected_sha256=expected_sha256,
            raw_sink=cache.raw,
        )
    return StooqDailyBarSource(
        raw_sink=cache.raw,
        timeout=config.data.providers.timeout_seconds,
        max_attempts=config.data.providers.max_attempts,
    )


def _provider_chain(
    config: EdgeStackConfig, cache: DataCache
) -> tuple[DailyBarSource, ...]:
    providers: dict[str, DailyBarSource] = {
        "stooq": _stooq_source(config, cache),
        "yfinance": YahooDailyBarSource(
            raw_sink=cache.raw,
            timeout=config.data.providers.timeout_seconds,
            max_attempts=config.data.providers.max_attempts,
        ),
    }
    key = os.environ.get(config.data.providers.tiingo_api_key_env, "")
    if key:
        providers["tiingo"] = TiingoDailyBarSource(
            key,
            raw_sink=cache.raw,
            timeout=config.data.providers.timeout_seconds,
            max_attempts=config.data.providers.max_attempts,
        )
    selected = tuple(
        providers[name] for name in config.data.providers.order if name in providers
    )
    if not selected:
        raise ValueError("configured provider order has no available adapter")
    return selected


async def _fetch_individually(
    source: DailyBarSource,
    requests: Sequence[BarRequest],
    *,
    concurrency: int,
) -> tuple[tuple[SourceBatch, ...], dict[str, str]]:
    if concurrency <= 0:
        raise ValueError("data-provider concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(
        request: BarRequest,
    ) -> tuple[str, SourceBatch | None, str | None]:
        async with semaphore:
            try:
                return request.asset.symbol, await source.fetch_bars(request), None
            except Exception as error:
                return (
                    request.asset.symbol,
                    None,
                    f"{type(error).__name__}: {error}",
                )

    outcomes = await asyncio.gather(*(one(request) for request in requests))
    batches = tuple(item for _, item, _ in outcomes if item is not None)
    failures = {symbol: error for symbol, _, error in outcomes if error is not None}
    return batches, failures


async def _reconcile_fixed_symbols(
    config: EdgeStackConfig,
    cache: DataCache,
    as_of: date,
    master_bars: pd.DataFrame,
) -> tuple[
    tuple[ReconciliationResult, ...],
    tuple[_CommonProviderEvidence, ...],
    dict[str, str],
]:
    stooq = _stooq_source(config, cache)
    yahoo = YahooDailyBarSource(
        raw_sink=cache.raw,
        timeout=config.data.providers.timeout_seconds,
        max_attempts=config.data.providers.max_attempts,
    )
    results: list[ReconciliationResult] = []
    common_evidence: list[_CommonProviderEvidence] = []
    failures: dict[str, str] = {}
    semaphore = asyncio.Semaphore(config.data.providers.concurrency)

    async def compare(
        symbol: str,
    ) -> tuple[
        str,
        ReconciliationResult | None,
        _CommonProviderEvidence,
        str | None,
    ]:
        request = BarRequest(AssetKey(symbol), config.data.start, as_of, adjusted=True)
        async with semaphore:
            fetched = await asyncio.gather(
                stooq.fetch_bars(request),
                yahoo.fetch_bars(request),
                return_exceptions=True,
            )
        provider_errors = [
            f"{provider}={type(value).__name__}: {value}"
            for provider, value in zip(("stooq", "yfinance"), fetched, strict=True)
            if isinstance(value, BaseException)
        ]
        if provider_errors:
            failure = "; ".join(provider_errors)
            return (
                symbol,
                None,
                _CommonProviderEvidence(symbol, (), failure),
                failure,
            )
        left = fetched[0]
        right = fetched[1]
        if not isinstance(left, SourceBatch) or not isinstance(right, SourceBatch):
            failure = "provider returned a non-SourceBatch response"
            return (
                symbol,
                None,
                _CommonProviderEvidence(symbol, (), failure),
                failure,
            )
        # Both payloads are cached even though they are QA-only observations.
        try:
            cache.store_batch(left)
            cache.store_batch(right)
        except Exception as error:
            failure = f"cache={type(error).__name__}: {error}"
            return (
                symbol,
                None,
                _CommonProviderEvidence(symbol, (), failure),
                failure,
            )
        left_frame = bars_to_frame(left)
        right_frame = bars_to_frame(right)
        common = _common_valid_sessions(left_frame, right_frame)
        if config.data.reconciliation_method == "action_stratified_returns":
            comparison_start = (pd.Timestamp(as_of) - pd.DateOffset(years=20)).date()
            result = reconcile_action_stratified_returns(
                left_frame,
                right_frame,
                symbol=symbol,
                source_a="stooq",
                source_b="yfinance",
                comparison_start=comparison_start,
                tolerance=config.data.reconciliation_tolerance,
                required_fraction=config.data.reconciliation_required_fraction,
            )
        else:
            left_adjusted = left_frame.assign(
                close=lambda frame: frame["adjusted_close"]
            )
            right_adjusted = right_frame.assign(
                close=lambda frame: frame["adjusted_close"]
            )
            result = reconcile_adjusted_series(
                left_adjusted,
                right_adjusted,
                symbol=symbol,
                source_a="stooq",
                source_b="yfinance",
                tolerance=config.data.reconciliation_tolerance,
                required_fraction=config.data.reconciliation_required_fraction,
            )
        return (
            symbol,
            result,
            _CommonProviderEvidence(symbol, tuple(common)),
            None,
        )

    compared = await asyncio.gather(
        *(compare(symbol) for symbol in config.data.reconciliation_tickers)
    )
    for symbol, result, coverage, failure in compared:
        if result is not None:
            results.append(result)
        common_evidence.append(coverage)
        if failure is not None:
            failures[f"reconciliation:{symbol}"] = failure
    await _close_sources((stooq, yahoo))
    return tuple(results), tuple(common_evidence), failures


def _fixed_symbol_gate_evidence(
    config: EdgeStackConfig,
    *,
    as_of: date,
    bars: pd.DataFrame,
    reconciliations: Sequence[ReconciliationResult],
    common_evidence: Sequence[_CommonProviderEvidence],
    failures: Mapping[str, str],
) -> tuple[FixedSymbolGateEvidence, ...]:
    """Build per-symbol 20-year eligible/common-session gate evidence."""

    cutoff = (pd.Timestamp(as_of) - pd.DateOffset(years=20)).date()
    expected = NYSECalendar().sessions(cutoff, as_of)
    required_start = expected[0].date() if len(expected) else cutoff
    required_end = expected[-1].date() if len(expected) else as_of
    reconciled_by_symbol = {item.symbol: item for item in reconciliations}
    common_by_symbol = {item.symbol: item for item in common_evidence}
    records: list[FixedSymbolGateEvidence] = []
    for symbol in config.data.reconciliation_tickers:
        observed_values = (
            bars.loc[bars["symbol"] == symbol, "session"]
            if not bars.empty and {"symbol", "session"}.issubset(bars.columns)
            else pd.Series(dtype="datetime64[ns]")
        )
        observed_count, observed_fraction, observed_span = _coverage_metrics(
            observed_values, expected
        )
        provider = common_by_symbol.get(symbol)
        common_count, common_fraction, common_span = _coverage_metrics(
            provider.sessions if provider is not None else (), expected
        )
        result = reconciled_by_symbol.get(symbol)
        history_pass = _history_pass(
            observed_fraction,
            observed_span,
            config.data.missing_bar_max_fraction,
        )
        common_history_pass = _history_pass(
            common_fraction,
            common_span,
            config.data.missing_bar_max_fraction,
        )
        reconciliation_pass = bool(
            result is not None and result.passed and common_history_pass
        )
        failure_parts = []
        if symbol in failures:
            failure_parts.append(f"acquisition={failures[symbol]}")
        reconciliation_failure = failures.get(f"reconciliation:{symbol}")
        if reconciliation_failure:
            failure_parts.append(reconciliation_failure)
        if provider is not None and provider.failure and not reconciliation_failure:
            failure_parts.append(provider.failure)
        if not history_pass and not failure_parts:
            failure_parts.append("eligible master history failed coverage/span gate")
        if not reconciliation_pass and not failure_parts:
            failure_parts.append("common provider history/agreement failed gate")
        records.append(
            FixedSymbolGateEvidence(
                symbol=symbol,
                required_start=required_start,
                required_end=required_end,
                expected_sessions=len(expected),
                observed_sessions=observed_count,
                observed_coverage_fraction=observed_fraction,
                observed_span_years=observed_span,
                common_sessions=common_count,
                common_coverage_fraction=common_fraction,
                common_span_years=common_span,
                agreement_fraction=(
                    result.agreement_fraction if result is not None else 0.0
                ),
                history_pass=history_pass,
                reconciliation_pass=reconciliation_pass,
                failure="; ".join(failure_parts) or None,
            )
        )
    return tuple(records)


def _coverage_metrics(
    observed: Sequence[object] | pd.Series,
    expected: pd.DatetimeIndex,
) -> tuple[int, float, float]:
    if not len(expected):
        return 0, 0.0, 0.0
    observed_values = pd.Series(list(observed), dtype="object")
    converted = pd.to_datetime(observed_values, errors="coerce").dropna()
    observed_index = pd.DatetimeIndex(converted)
    if observed_index.tz is not None:
        observed_index = observed_index.tz_convert(None)
    eligible = expected.intersection(observed_index.normalize().unique()).sort_values()
    coverage = len(eligible) / len(expected)
    span = (eligible[-1] - eligible[0]).days / 365.2425 if len(eligible) >= 2 else 0.0
    return len(eligible), float(coverage), float(span)


def _history_pass(
    coverage_fraction: float,
    span_years: float,
    missing_bar_threshold: float,
) -> bool:
    missing_fraction = 1.0 - coverage_fraction
    return bool(
        missing_fraction < missing_bar_threshold
        and span_years >= _MINIMUM_HISTORY_YEARS
    )


def _common_valid_sessions(left: pd.DataFrame, right: pd.DataFrame) -> pd.DatetimeIndex:
    def valid(frame: pd.DataFrame) -> pd.DatetimeIndex:
        prices = pd.to_numeric(frame["close"], errors="coerce")
        sessions = pd.to_datetime(frame.loc[prices.gt(0) & prices.notna(), "session"])
        return pd.DatetimeIndex(sessions).normalize().unique().sort_values()

    return valid(left).intersection(valid(right)).sort_values()


def _apply_causal_corrections(
    bars: pd.DataFrame, *, sigma: float
) -> tuple[pd.DataFrame, tuple[CorrectionRecord, ...]]:
    """Preserve source close and expose a causally winsorized research close."""

    corrected_frames: list[pd.DataFrame] = []
    records: list[CorrectionRecord] = []
    for symbol, group in bars.groupby("symbol", sort=True):
        corrected, group_records = causal_winsorize_prices(
            group,
            symbol=str(symbol),
            price_column="close",
            sigma=sigma,
        )
        corrected["adjusted_close"] = corrected["research_close"]
        corrected_frames.append(corrected)
        records.extend(group_records)
    combined = pd.concat(corrected_frames, ignore_index=True)
    return (
        combined.sort_values(["symbol", "session"], kind="stable").reset_index(
            drop=True
        ),
        tuple(records),
    )


def _persist_correction_evidence(
    records: Sequence[CorrectionRecord], root: Path
) -> str:
    identity = canonical_sha256([asdict(item) for item in records])
    return write_correction_log(records, root / f"{identity}.json")


def _frame_content_sha256(frame: pd.DataFrame) -> str:
    """Hash frame schema, ordered rows, and values for fixture snapshot identity."""

    digest = hashlib.sha256()
    schema = [(str(column), str(frame[column].dtype)) for column in frame.columns]
    digest.update(canonical_sha256(schema).encode())
    hashed = pd.util.hash_pandas_object(frame, index=False, categorize=True)
    digest.update(hashed.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


async def _close_sources(sources: Sequence[DailyBarSource]) -> None:
    for source in sources:
        close = getattr(source, "aclose", None)
        if close is not None:
            await close()


def memberships_frame(
    memberships: Sequence[MembershipInterval],
) -> pd.DataFrame:
    """Flatten immutable universe intervals for Parquet/JSON persistence."""

    return pd.DataFrame(
        [
            {
                "symbol": item.asset.symbol,
                "exchange": item.asset.exchange,
                "asset_type": item.asset.asset_type,
                "start": item.start,
                "end": item.end,
                "sector": item.sector,
                "available_at": item.available_at,
            }
            for item in memberships
        ]
    )


def synthetic_replication_inputs(
    data: IngestedCampaignData,
) -> dict[str, Any]:
    """Create deterministic smoke-only inputs that exercise all six checks.

    This helper is never used by a full empirical campaign and its output is
    always stamped ``SYNTHETIC_SMOKE_NON_PROMOTABLE``.
    """

    market = data.factors.set_index("session")["market_return"].astype(float)
    spy = market.copy()
    sessions = market.index
    rng = np.random.default_rng(8008)
    momentum = pd.Series(0.00055 + rng.normal(0, 0.004, len(sessions)), index=sessions)
    crash = (sessions >= "2009-01-02") & (sessions <= "2009-03-31")
    momentum.loc[crash] = -0.006
    reversal_gross = np.full(len(sessions), 0.00030)
    reversal_net = np.full(len(sessions), 0.00008)
    bars_by_symbol = {
        symbol: group.sort_values("session").set_index("session")
        for symbol, group in data.bars.loc[
            data.bars["symbol"].isin(["SPY", "QQQ"])
        ].groupby("symbol")
    }
    return {
        "market_returns": market,
        "spy_returns": spy,
        "fomc_dates": data.fomc_dates,
        "bars_by_symbol": bars_by_symbol,
        "momentum_returns": momentum,
        "reversal_gross": reversal_gross,
        "reversal_net": reversal_net,
    }
