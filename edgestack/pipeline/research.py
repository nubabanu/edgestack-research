"""Deterministic research-grid execution used by campaign orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import numpy as np
import pandas as pd

from edgestack.backtest.costs import DEFAULT_ADV_FALLBACK_DOLLARS, CostModel
from edgestack.backtest.engine import (
    BacktestResult,
    close_derived_execution_lag,
    overlapping_cohort_targets,
    vectorized_backtest,
)
from edgestack.backtest.metrics import performance_metrics
from edgestack.config import EdgeStackConfig
from edgestack.data.calendars import NYSECalendar
from edgestack.features.calendar_feats import calendar_features
from edgestack.features.cross_sectional import (
    amihud_illiquidity,
    canonical_features,
    decile_weights,
    max_lottery,
    overnight_intraday_gap,
    short_term_reversal,
)
from edgestack.hypotheses.controls import (
    control_specs,
    matched_random_signal,
    shuffled_date_returns,
)
from edgestack.hypotheses.grid import (
    DEFAULT_PREDICATES,
    EXTENDED_PREDICATES,
    GridConfig,
    conditional_combination_hypotheses,
    cross_sectional_hypotheses,
    enumerate_hypotheses,
)
from edgestack.models import (
    Direction,
    HypothesisSpec,
    RationaleCategory,
    Session,
)
from edgestack.stats.bootstrap import (
    stationary_bootstrap_indices,
)
from edgestack.stats.deflated_sharpe import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from edgestack.stats.multiple_testing import (
    bonferroni,
    discovery_gauntlet,
    romano_wolf_stepdown,
)
from edgestack.stats.reality_check import hansen_spa, white_reality_check
from edgestack.stats.tests import summarize_returns


@dataclass(frozen=True, slots=True)
class PreparedResearch:
    """Causally aligned matrices and precomputed feature state."""

    dates: pd.DatetimeIndex
    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    close_returns: pd.DataFrame
    overnight_returns: pd.DataFrame
    intraday_returns: pd.DataFrame
    calendar: pd.DataFrame
    sector_by_symbol: dict[str, str]
    asset_types: tuple[str, ...]
    market_open: tuple[pd.Timestamp, ...]
    market_close: tuple[pd.Timestamp, ...]


@dataclass(frozen=True, slots=True)
class TrialRun:
    """One complete real or matched-control backtest."""

    spec: HypothesisSpec
    result: BacktestResult
    signal: np.ndarray[Any, np.dtype[np.float64]]
    underlying_returns: np.ndarray[Any, np.dtype[np.float64]]
    adv_dollars: float | np.ndarray[Any, np.dtype[np.float64]]
    asset_type: str | tuple[str, ...]
    benchmark_returns: np.ndarray[Any, np.dtype[np.float64]] | None
    execution_lag: int


@dataclass(frozen=True, slots=True)
class DiscoveryBundle:
    """All declarations, evidence rows, streams, and family-level tests."""

    dates: pd.DatetimeIndex
    specs: tuple[HypothesisSpec, ...]
    metrics: pd.DataFrame
    net_returns: pd.DataFrame
    gross_returns: pd.DataFrame
    survivor_ids: tuple[str, ...]
    provisional_placebo_fraction: float
    survivor_fraction_t_fdr: float
    spa_p_value: float
    reality_check_p_value: float
    romano_wolf_rejection_count: int = 0
    romano_wolf_method: str = "NOT_REQUIRED"


@dataclass(frozen=True, slots=True)
class DiscoveryProgress:
    """A durable-marker opportunity at a bounded discovery boundary.

    The callback is observational.  ``resumable`` remains false because the
    family-test matrix is an ephemeral memory map; campaign-level resume needs
    the runner to transactionally persist that matrix and compact summaries.
    """

    phase: Literal[
        "trial_batches",
        "family_draws",
        "family_models",
        "confidence_intervals",
        "complete",
    ]
    completed: int
    total: int
    completed_trials: int
    resumable: bool = False
    limitation: str = (
        "progress markers are not restart state; runner-managed artifact "
        "persistence is required for crash-safe resume"
    )


ProgressCallback = Callable[[DiscoveryProgress], None]
_CANONICAL_DRAW_BATCH = 32


@dataclass(slots=True)
class _CompactTrial:
    """Metrics retained in place of a full signal/position/return object."""

    spec: HypothesisSpec
    row: dict[str, Any]
    periodic_sharpe: float


@dataclass(frozen=True, slots=True)
class _FamilyPValues:
    """Family-wide White, SPA, and optional Romano-Wolf evidence."""

    spa: float
    reality_check: float
    method: str = "BOUNDED_ARCH_EQUIVALENT"
    romano_wolf_adjusted_p_values: np.ndarray[Any, np.dtype[np.float64]] | None = None
    romano_wolf_method: str = "NOT_REQUIRED"


def prepare_research(
    bars: pd.DataFrame,
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    fomc_dates: pd.DatetimeIndex,
    sector_by_symbol: dict[str, str],
) -> PreparedResearch:
    """Align adjusted prices without reading observations past ``end``."""

    required = {
        "symbol",
        "session",
        "event_time",
        "available_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"campaign bars missing {sorted(missing)}")
    frame = bars.copy()
    frame["session"] = pd.to_datetime(frame["session"]).dt.normalize()
    frame = frame.loc[frame["session"].between(pd.Timestamp(start), pd.Timestamp(end))]
    if frame.empty:
        raise ValueError("no bars in requested research interval")
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    if bool((frame["available_at"] <= frame["event_time"]).any()):
        raise ValueError("every campaign observation must arrive after event_time")
    if bool(frame.duplicated(["symbol", "session"]).any()):
        raise ValueError("duplicate symbol/session observations are not causal")
    ordered = frame.sort_values(["symbol", "session"], kind="stable")
    next_event = ordered.groupby("symbol", sort=False)["event_time"].shift(-1)
    delayed = next_event.notna() & (ordered["available_at"] >= next_event)
    if bool(delayed.any()):
        raise ValueError(
            "a daily bar was not available strictly before its next eligible bar"
        )
    close_column = "adjusted_close" if "adjusted_close" in frame else "close"
    close = frame.pivot(
        index="session", columns="symbol", values=close_column
    ).sort_index()
    raw_close = frame.pivot(
        index="session", columns="symbol", values="close"
    ).reindex_like(close)
    open_raw = frame.pivot(
        index="session", columns="symbol", values="open"
    ).reindex_like(close)
    high_raw = frame.pivot(
        index="session", columns="symbol", values="high"
    ).reindex_like(close)
    low_raw = frame.pivot(index="session", columns="symbol", values="low").reindex_like(
        close
    )
    # Apply the close adjustment ratio to OHLC so overnight/intraday definitions
    # remain internally consistent on split/dividend dates.
    ratio = close.div(raw_close).where(raw_close > 0)
    open_ = open_raw.mul(ratio)
    high = high_raw.mul(ratio)
    low = low_raw.mul(ratio)
    upper_body = open_.where(open_ >= close, close)
    lower_body = open_.where(open_ <= close, close)
    # Multiplying raw OHLC by an adjustment ratio can move mathematically
    # equal endpoints apart by one floating-point ULP.  Reject genuine OHLC
    # conflicts while accepting only machine-precision equality noise.
    high_shortfall = high.lt(upper_body) & ~np.isclose(
        high, upper_body, rtol=1e-12, atol=0.0, equal_nan=True
    )
    low_excess = low.gt(lower_body) & ~np.isclose(
        low, lower_body, rtol=1e-12, atol=0.0, equal_nan=True
    )
    inverted_range = high.lt(low) & ~np.isclose(
        high, low, rtol=1e-12, atol=0.0, equal_nan=True
    )
    invalid_ohlc = high_shortfall | low_excess | inverted_range
    if bool(invalid_ohlc.to_numpy().any()):
        raise ValueError("adjusted OHLC invariants are violated")
    volume = frame.pivot(
        index="session", columns="symbol", values="volume"
    ).reindex_like(close)
    close_returns = close.pct_change(fill_method=None)
    overnight = open_.div(close.shift(1)) - 1.0
    intraday = close.div(open_) - 1.0
    sessions = pd.DatetimeIndex(close.index)
    schedule = NYSECalendar().schedule(sessions.min(), sessions.max()).reindex(sessions)
    if bool(schedule.isna().any(axis=None)):
        raise ValueError("research interval contains a non-NYSE session")
    session_availability = frame.groupby("session", sort=True)["available_at"].max()
    next_market_open = schedule["market_open"].shift(-1)
    delayed_past_open = (
        session_availability.reindex(sessions) >= next_market_open
    ).fillna(False)
    if bool(delayed_past_open.any()):
        raise ValueError(
            "a daily bar was not available strictly before its next eligible open"
        )
    weekdays = pd.date_range(sessions.min(), sessions.max(), freq="B")
    holidays = weekdays.difference(sessions)
    calendar = calendar_features(
        sessions,
        holidays=holidays,
        fomc_dates=fomc_dates,
    )
    if "asset_type" in frame:
        by_symbol = (
            frame.sort_values("session", kind="stable")
            .groupby("symbol", sort=False)["asset_type"]
            .last()
            .astype(str)
        )
        asset_types = tuple(
            "etf" if by_symbol.get(str(symbol), "equity").lower() == "etf" else "equity"
            for symbol in close.columns
        )
    else:
        asset_types = tuple(
            "etf" if sector_by_symbol.get(str(symbol)) == "ETF" else "equity"
            for symbol in close.columns
        )
    return PreparedResearch(
        sessions,
        close,
        open_,
        high,
        low,
        volume,
        close_returns,
        overnight,
        intraday,
        calendar,
        sector_by_symbol,
        asset_types,
        tuple(pd.Timestamp(value) for value in schedule["market_open"]),
        tuple(pd.Timestamp(value) for value in schedule["market_close"]),
    )


def declared_hypotheses(
    prepared: PreparedResearch, config: EdgeStackConfig
) -> tuple[HypothesisSpec, ...]:
    """Materialize the preregistered real-hypothesis family."""

    sectors = tuple(
        sorted(
            {
                sector
                for symbol, sector in prepared.sector_by_symbol.items()
                if sector and sector != "ETF" and symbol in prepared.close.columns
            }
        )
    )
    predicate_levels = dict(DEFAULT_PREDICATES)
    if config.grid.extended_families:
        predicate_levels.update(EXTENDED_PREDICATES)
    grid = GridConfig(
        predicate_levels=predicate_levels,
        sectors=sectors,
        holding_periods=config.grid.close_holding_periods,
        directions=tuple(Direction(value) for value in config.grid.directions),
        sessions=tuple(Session(value) for value in config.grid.sessions),
        include_any=True,
        include_pairwise=config.grid.max_interaction_order == 2,
    )
    specs: list[HypothesisSpec] = enumerate_hypotheses(grid)
    if config.grid.include_cross_sectional:
        specs.extend(
            cross_sectional_hypotheses(extended=config.grid.extended_families)
        )
        if config.grid.extended_families:
            specs.extend(conditional_combination_hypotheses())
    return tuple(specs)


def run_trial(
    prepared: PreparedResearch,
    spec: HypothesisSpec,
    *,
    cost_model: CostModel,
) -> TrialRun:
    """Backtest one declaration with its exact causal execution lag."""

    adv: float | np.ndarray[Any, np.dtype[np.float64]]
    if spec.family == "calendar":
        signal, returns = _calendar_trial_inputs(prepared, spec)
        sector = spec.predicates.get("sector")
        selected = [
            symbol
            for symbol in prepared.close.columns
            if sector is None or prepared.sector_by_symbol.get(str(symbol)) == sector
        ]
        selected_adv = prepared.close.loc[:, selected].mul(
            prepared.volume.loc[:, selected]
        )
        adv = (
            selected_adv.rolling(20, min_periods=1)
            .mean()
            .mean(axis=1)
            .shift(1)
            .fillna(DEFAULT_ADV_FALLBACK_DOLLARS)
            .to_numpy(float)
        )
        selected_types = {
            prepared.asset_types[prepared.close.columns.get_loc(symbol)]
            for symbol in selected
        }
        asset_type: str | tuple[str, ...] = (
            "etf" if selected_types == {"etf"} else "equity"
        )
    else:
        signal, returns = _cross_sectional_trial_inputs(prepared, spec)
        adv_frame = (
            prepared.close.mul(prepared.volume)
            .rolling(20, min_periods=1)
            .mean()
            .shift(1)
        )
        adv = np.nan_to_num(
            adv_frame.to_numpy(dtype=float),
            nan=DEFAULT_ADV_FALLBACK_DOLLARS,
            posinf=DEFAULT_ADV_FALLBACK_DOLLARS,
        )
        asset_type = prepared.asset_types
    benchmark_returns = _spy_benchmark_returns(prepared, spec)
    execution_lag = _trial_execution_lag(spec)
    gross, net, positions = vectorized_backtest(
        signal,
        returns,
        execution_lag=execution_lag,
        cost_model=cost_model,
        asset_type=asset_type,
        adv_dollars=adv,
    )
    holding = int(spec.holding_period) if isinstance(spec.holding_period, int) else 1
    statistics = summarize_returns(
        net,
        holding_period=holding,
        minimum_observations=100,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics = performance_metrics(
            net,
            positions=positions,
            benchmark=benchmark_returns,
        )
    return TrialRun(
        spec,
        BacktestResult(
            spec.hypothesis_id,
            gross,
            net,
            positions,
            statistics,
            metrics,
        ),
        np.asarray(signal, dtype=float),
        np.asarray(returns, dtype=float),
        adv,
        asset_type,
        benchmark_returns,
        execution_lag,
    )


# A surviving trial whose HAC t improves by more than this margin when an
# EXTRA execution lag is inserted is flagged: honest slow signals keep or lose
# strength under delay, while material improvement suggests a timing artifact.
_EXTRA_LAG_T_INFLATION_MARGIN = 1.0


def _truncated_prepared(prepared: PreparedResearch, length: int) -> PreparedResearch:
    """Return the identical research panel with every date past ``length`` removed."""

    return PreparedResearch(
        dates=prepared.dates[:length],
        close=prepared.close.iloc[:length],
        open=prepared.open.iloc[:length],
        high=prepared.high.iloc[:length],
        low=prepared.low.iloc[:length],
        volume=prepared.volume.iloc[:length],
        close_returns=prepared.close_returns.iloc[:length],
        overnight_returns=prepared.overnight_returns.iloc[:length],
        intraday_returns=prepared.intraday_returns.iloc[:length],
        calendar=prepared.calendar.iloc[:length],
        sector_by_symbol=prepared.sector_by_symbol,
        asset_types=prepared.asset_types,
        market_open=prepared.market_open[:length],
        market_close=prepared.market_close[:length],
    )


def _survivor_causality_evidence(
    prepared: PreparedResearch,
    trial: TrialRun,
    *,
    cost_model: CostModel,
) -> dict[str, Any]:
    """Run per-survivor causality invariants on one discovery survivor.

    Hard gate: recomputing the signal with all future sessions removed must
    not change any value before the truncation boundary (a buffer of sessions
    below the boundary is exempt because holding windows and month-boundary
    calendar predicates legitimately need forward sessions to complete).
    Soft gate: an extra execution lag may not materially IMPROVE the HAC t;
    losing strength under delay is expected and never gates.
    """

    spec = trial.spec
    holding = int(spec.holding_period) if isinstance(spec.holding_period, int) else 1
    buffer = max(25, holding + trial.execution_lag + 5)
    total = len(prepared.dates)
    prefix_length = total - buffer
    if prefix_length < 2 * buffer:
        return {
            "causality_prefix_invariant": True,
            "causality_baseline_t": trial.result.return_statistics.hac_t_stat,
            "causality_extra_lag_t": math.nan,
            "causality_lag_inflation": False,
            "causality_pass": True,
            "causality_reason": "SAMPLE_TOO_SHORT_TO_TEST",
        }
    truncated_trial = run_trial(
        _truncated_prepared(prepared, prefix_length), spec, cost_model=cost_model
    )
    boundary = prefix_length - buffer
    prefix_invariant = bool(
        np.allclose(
            np.asarray(trial.signal, dtype=float)[:boundary],
            np.asarray(truncated_trial.signal, dtype=float)[:boundary],
            equal_nan=True,
        )
    )
    _, extra_lag_net, _ = vectorized_backtest(
        trial.signal,
        trial.underlying_returns,
        execution_lag=trial.execution_lag + 1,
        cost_model=cost_model,
        asset_type=trial.asset_type,
        adv_dollars=trial.adv_dollars,
    )
    baseline_t = trial.result.return_statistics.hac_t_stat
    extra_lag_t = summarize_returns(
        extra_lag_net, holding_period=holding, minimum_observations=100
    ).hac_t_stat
    lag_inflation = bool(
        math.isfinite(baseline_t)
        and math.isfinite(extra_lag_t)
        and extra_lag_t > baseline_t + _EXTRA_LAG_T_INFLATION_MARGIN
    )
    passed = prefix_invariant and not lag_inflation
    if not prefix_invariant:
        reason = "SIGNAL_CHANGED_WHEN_FUTURE_SESSIONS_REMOVED"
    elif lag_inflation:
        reason = "HAC_T_IMPROVED_UNDER_EXTRA_EXECUTION_LAG"
    else:
        reason = "PASSED"
    return {
        "causality_prefix_invariant": prefix_invariant,
        "causality_baseline_t": baseline_t,
        "causality_extra_lag_t": extra_lag_t,
        "causality_lag_inflation": lag_inflation,
        "causality_pass": passed,
        "causality_reason": reason,
    }


def run_discovery(
    prepared: PreparedResearch,
    config: EdgeStackConfig,
    *,
    batch_size: int = 64,
    family_strategy_batch: int = 256,
    family_bootstrap_batch: int = 32,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: ProgressCallback | None = None,
) -> DiscoveryBundle:
    """Run the preregistered family with bounded resident memory.

    Full ``TrialRun`` objects are discarded immediately after their compact
    metrics are extracted.  A temporary disk-backed matrix retains only the
    real-family net streams required by the family-wide SPA/Reality Check.
    Returned Parquet-bound streams contain only real discovery survivors,
    which are the only streams consumed by validation and stack construction.

    Progress and checkpoint callbacks fire only at deterministic batch
    boundaries.  They intentionally do not claim restart support: the current
    runner does not persist the temporary family matrix transactionally.
    """

    if batch_size < 1 or family_strategy_batch < 1 or family_bootstrap_batch < 1:
        raise ValueError("all discovery batch sizes must be positive")
    callbacks = tuple(
        callback
        for callback in (progress_callback, checkpoint_callback)
        if callback is not None
    )
    cost_model = CostModel(config.costs)
    real_specs = declared_hypotheses(prepared, config)
    real_by_id = {spec.hypothesis_id: spec for spec in real_specs}
    compact: list[_CompactTrial] = []
    total_real = len(real_specs)
    with TemporaryDirectory(
        prefix="edgestack-discovery-", ignore_cleanup_errors=True
    ) as temporary:
        workdir = Path(temporary)
        real_matrix: np.memmap[Any, np.dtype[np.float64]] | None = None
        if total_real:
            real_matrix = np.memmap(
                workdir / "real-family-net.dat",
                mode="w+",
                dtype=np.float64,
                shape=(len(prepared.dates), total_real),
                order="F",
            )
        try:
            for start in range(0, total_real, batch_size):
                stop = min(start + batch_size, total_real)
                for real_position in range(start, stop):
                    spec = real_specs[real_position]
                    real = run_trial(prepared, spec, cost_model=cost_model)
                    compact.append(_compact_trial(real))
                    if real_matrix is not None:
                        real_matrix[:, real_position] = np.nan_to_num(
                            real.result.net_returns,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                    controls = control_specs(spec, campaign_seed=config.stats.seed)
                    if len(controls) != 2:
                        raise RuntimeError(
                            "every real hypothesis must have exactly two controls"
                        )
                    for control in controls:
                        compact.append(
                            _compact_trial(
                                _control_trial(real, control, cost_model=cost_model)
                            )
                        )
                if real_matrix is not None:
                    real_matrix.flush()
                _emit_progress(
                    callbacks,
                    DiscoveryProgress(
                        "trial_batches",
                        stop,
                        total_real,
                        len(compact),
                    ),
                )

            trial_sharpes = np.asarray(
                [item.periodic_sharpe for item in compact], dtype=float
            )
            rows = [
                _finalize_compact_metrics(
                    item,
                    trial_sharpes=trial_sharpes,
                    n_trials=max(len(compact), 1),
                )
                for item in compact
            ]
            metrics = pd.DataFrame(rows)
            t_thresholds = np.full(len(metrics), config.stats.hard_t, dtype=float)
            if config.protocol.version != "FROZEN_V1":
                calendar_trials = (
                    metrics["family"].astype(str).eq("calendar").to_numpy(dtype=bool)
                )
                t_thresholds[calendar_trials] = config.protocol.time_series_t_threshold
                t_thresholds[~calendar_trials] = (
                    config.protocol.cross_sectional_t_threshold
                )
            gauntlet = discovery_gauntlet(
                sample_sizes=metrics["sample_size"].to_numpy(dtype=int),
                directed_means=metrics["net_mean"].to_numpy(dtype=float),
                t_statistics=metrics["hac_t"].to_numpy(dtype=float),
                p_values=np.nan_to_num(
                    metrics["p_value"].to_numpy(dtype=float), nan=1.0
                ),
                dsr_probabilities=np.nan_to_num(
                    metrics["deflated_sharpe_probability"].to_numpy(dtype=float),
                    nan=0.0,
                ),
                minimum_observations=config.grid.min_observations,
                t_threshold=t_thresholds,
                fdr_q=config.stats.fdr_q,
                dsr_probability=config.stats.dsr_probability,
            )
            metrics["minimum_sample_pass"] = gauntlet.minimum_sample
            metrics["directed_positive_pass"] = gauntlet.directed_positive
            metrics["t_threshold"] = t_thresholds
            metrics["t_pass"] = gauntlet.t_gate
            metrics["bh_pass"] = gauntlet.fdr_gate
            metrics["bh_adjusted_p"] = gauntlet.adjusted_p_values
            raw_p_values = metrics["p_value"].to_numpy(dtype=float)
            bonferroni_inputs = np.where(
                gauntlet.minimum_sample
                & gauntlet.directed_positive
                & np.isfinite(raw_p_values),
                raw_p_values,
                1.0,
            )
            global_bonferroni = bonferroni(bonferroni_inputs, alpha=config.stats.fdr_q)
            metrics["bonferroni_pass"] = global_bonferroni.reject
            metrics["bonferroni_adjusted_p"] = global_bonferroni.adjusted_p_values
            metrics["dsr_pass"] = gauntlet.dsr_gate
            metrics["discovery_survivor_pre_spa"] = gauntlet.survivors

            if real_matrix is not None:
                family = _bounded_family_p_values(
                    real_matrix,
                    n_bootstrap=config.stats.finalist_bootstrap_reps,
                    seed=config.stats.seed,
                    workdir=workdir,
                    strategy_batch=family_strategy_batch,
                    bootstrap_batch=family_bootstrap_batch,
                    callbacks=callbacks,
                    completed_trials=len(compact),
                    romano_wolf_alpha=(
                        config.protocol.romano_wolf_alpha
                        if config.protocol.require_romano_wolf
                        else None
                    ),
                )
            else:
                family = _FamilyPValues(1.0, 1.0, "NO_REAL_FAMILY")
        finally:
            _close_memmap(real_matrix)

        family_pass = (
            family.spa < config.stats.family_alpha
            and family.reality_check < config.stats.family_alpha
        )
        real_mask = metrics["placebo_kind"].isna().to_numpy()
        romano_wolf_adjusted = np.ones(len(metrics), dtype=float)
        romano_wolf_pass = np.ones(len(metrics), dtype=bool)
        if config.protocol.require_romano_wolf:
            real_adjusted = family.romano_wolf_adjusted_p_values
            if real_adjusted is None or len(real_adjusted) != int(real_mask.sum()):
                raise RuntimeError("Romano-Wolf evidence is missing or misaligned")
            romano_wolf_adjusted[real_mask] = real_adjusted
            romano_wolf_pass[real_mask] = (
                real_adjusted <= config.protocol.romano_wolf_alpha
            )
        metrics["family_test_scope"] = "ALL_PREREGISTERED_REAL"
        metrics["family_test_real_count"] = total_real
        metrics["family_test_method"] = family.method
        metrics["spa_pass"] = gauntlet.survivors & family_pass
        metrics["reality_check_pass"] = gauntlet.survivors & family_pass
        metrics["romano_wolf_scope"] = "ALL_PREREGISTERED_REAL"
        metrics["romano_wolf_method"] = family.romano_wolf_method
        metrics["romano_wolf_adjusted_p"] = romano_wolf_adjusted
        metrics["romano_wolf_pass"] = romano_wolf_pass
        metrics["discovery_survivor"] = metrics["spa_pass"] & romano_wolf_pass

        # Ineligible declarations retain an explicit HAC normal interval.
        # Eligible and finalist intervals are then overwritten using shared,
        # disk-backed stationary paths without a reps-by-dates RAM allocation.
        metrics["mean_ci_lower"] = metrics["net_mean"] - 1.96 * (
            metrics["net_mean"].abs() / metrics["hac_t"].abs().replace(0, np.nan)
        )
        metrics["mean_ci_upper"] = metrics["net_mean"] + 1.96 * (
            metrics["net_mean"].abs() / metrics["hac_t"].abs().replace(0, np.nan)
        )
        metrics["sharpe_ci_lower"] = np.nan
        metrics["sharpe_ci_upper"] = np.nan
        base_eligible = (
            gauntlet.minimum_sample
            & gauntlet.directed_positive
            & gauntlet.t_gate
            & gauntlet.fdr_gate
        )
        prelim = np.flatnonzero(gauntlet.survivors)
        _apply_shared_bootstrap_intervals(
            prepared,
            cost_model,
            compact,
            real_by_id,
            metrics,
            plans=(
                (np.flatnonzero(base_eligible), config.stats.bootstrap_reps),
                (prelim, config.stats.finalist_bootstrap_reps),
            ),
            seed=config.stats.seed,
            workdir=workdir,
            bootstrap_batch=family_bootstrap_batch,
            callbacks=callbacks,
        )

        t_fdr = (
            gauntlet.minimum_sample
            & gauntlet.directed_positive
            & gauntlet.t_gate
            & gauntlet.fdr_gate
        )
        real_count = int(real_mask.sum())
        survivor_fraction = (
            float(t_fdr[real_mask].sum() / real_count) if real_count else 0.0
        )
        placebo_mask = ~real_mask
        placebo_count = int(placebo_mask.sum())
        placebo_fraction = (
            float(gauntlet.survivors[placebo_mask].sum() / placebo_count)
            if placebo_count
            else 0.0
        )
        survivors = tuple(
            metrics.loc[
                metrics["discovery_survivor"] & metrics["placebo_kind"].isna(),
                "hypothesis_id",
            ].astype(str)
        )
        survivor_net: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
        survivor_gross: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
        causality_rows: dict[str, dict[str, Any]] = {}
        causality_rejected: set[str] = set()
        for hypothesis_id in survivors:
            trial = run_trial(
                prepared,
                real_by_id[hypothesis_id],
                cost_model=cost_model,
            )
            if config.stats.survivor_causality_checks and isinstance(
                prepared, PreparedResearch
            ):
                evidence = _survivor_causality_evidence(
                    prepared, trial, cost_model=cost_model
                )
                causality_rows[hypothesis_id] = evidence
                if not evidence["causality_pass"]:
                    causality_rejected.add(hypothesis_id)
                    continue
            survivor_net[hypothesis_id] = trial.result.net_returns
            survivor_gross[hypothesis_id] = trial.result.gross_returns
        if causality_rows:
            for column in (
                "causality_prefix_invariant",
                "causality_baseline_t",
                "causality_extra_lag_t",
                "causality_lag_inflation",
                "causality_pass",
                "causality_reason",
            ):
                metrics[column] = metrics["hypothesis_id"].map(
                    {key: row[column] for key, row in causality_rows.items()}
                )
        if causality_rejected:
            metrics.loc[
                metrics["hypothesis_id"].isin(causality_rejected),
                "discovery_survivor",
            ] = False
            survivors = tuple(
                hypothesis_id
                for hypothesis_id in survivors
                if hypothesis_id not in causality_rejected
            )
        net_streams = pd.DataFrame(survivor_net, index=prepared.dates)
        gross_streams = pd.DataFrame(survivor_gross, index=prepared.dates)
        _emit_progress(
            callbacks,
            DiscoveryProgress(
                "complete",
                total_real,
                total_real,
                len(compact),
            ),
        )
        return DiscoveryBundle(
            prepared.dates,
            tuple(item.spec for item in compact),
            metrics,
            net_streams,
            gross_streams,
            survivors,
            placebo_fraction,
            survivor_fraction,
            family.spa,
            family.reality_check,
            (
                int(np.count_nonzero(romano_wolf_pass & real_mask))
                if config.protocol.require_romano_wolf
                else 0
            ),
            family.romano_wolf_method,
        )


def _compact_trial(trial: TrialRun) -> _CompactTrial:
    """Extract reporting and selection inputs, dropping all large arrays."""

    result = trial.result
    stats = result.return_statistics
    active = (
        np.any(np.abs(result.positions) > 1e-12, axis=1)
        if result.positions.ndim == 2
        else np.abs(result.positions) > 1e-12
    )
    sample_size = int(np.count_nonzero(active & np.isfinite(result.gross_returns)))
    gross_finite = result.gross_returns[np.isfinite(result.gross_returns)]
    gross_mean = float(gross_finite.mean()) if gross_finite.size else math.nan
    row: dict[str, Any] = {
        "hypothesis_id": trial.spec.hypothesis_id,
        "parent_id": trial.spec.parameters.get("parent_id"),
        "family": trial.spec.family,
        "description": trial.spec.description,
        "direction": trial.spec.direction.value,
        "session": trial.spec.session.value,
        "holding_period": trial.spec.holding_period,
        "execution_lag": trial.execution_lag,
        "placebo_kind": trial.spec.placebo_kind,
        "sample_size": sample_size,
        "empty_signal": sample_size == 0,
        "gross_mean": gross_mean,
        "net_mean": stats.mean,
        "baseline_cost_mean": gross_mean - stats.mean,
        "hac_t": stats.hac_t_stat,
        "p_value": stats.hac_p_value,
        "hac_lags": stats.hac_lags,
        "sharpe": stats.annualized_sharpe,
        "hit_rate": stats.hit_rate,
        "skew": stats.skewness,
        "kurtosis": stats.kurtosis,
        "benchmark_symbol": "SPY",
        "benchmark_available": trial.benchmark_returns is not None,
        **result.performance.as_dict(),
    }
    return _CompactTrial(trial.spec, row, _periodic_sharpe(result.net_returns))


def _finalize_compact_metrics(
    compact: _CompactTrial,
    *,
    trial_sharpes: np.ndarray[Any, np.dtype[np.float64]],
    n_trials: int,
) -> dict[str, Any]:
    """Add PSR/DSR once the complete preregistered trial family is known."""

    row = dict(compact.row)
    sample_size = int(row["sample_size"])
    row["probabilistic_sharpe"] = probabilistic_sharpe_ratio(
        compact.periodic_sharpe,
        n_observations=sample_size,
        skewness=_finite_or(float(row["skew"]), 0.0),
        kurtosis=_finite_or(float(row["kurtosis"]), 3.0),
    )
    finite_sharpes = trial_sharpes[np.isfinite(trial_sharpes)]
    if finite_sharpes.size:
        row["deflated_sharpe_probability"] = deflated_sharpe_ratio(
            compact.periodic_sharpe,
            n_observations=sample_size,
            n_trials=n_trials,
            skewness=_finite_or(float(row["skew"]), 0.0),
            kurtosis=_finite_or(float(row["kurtosis"]), 3.0),
            trial_sharpes=finite_sharpes,
        )
    else:
        row["deflated_sharpe_probability"] = math.nan
    return row


def _control_trial(
    real: TrialRun,
    control: HypothesisSpec,
    *,
    cost_model: CostModel,
) -> TrialRun:
    """Execute one of the exactly two registered deterministic controls."""

    seed = int(control.parameters["control_seed"])
    if control.placebo_kind == "SHUFFLED_DATE":
        signal = real.signal
        returns = np.asarray(
            shuffled_date_returns(real.underlying_returns, seed=seed),
            dtype=float,
        )
    elif control.placebo_kind == "MATCHED_RANDOM":
        signal = np.asarray(matched_random_signal(real.signal, seed=seed), dtype=float)
        returns = real.underlying_returns
    else:
        raise ValueError(f"unsupported control {control.placebo_kind!r}")
    return _run_explicit(
        control,
        signal,
        returns,
        cost_model,
        adv_dollars=real.adv_dollars,
        asset_type=real.asset_type,
        benchmark_returns=real.benchmark_returns,
    )


def _registered_trial(
    prepared: PreparedResearch,
    compact: _CompactTrial,
    real_by_id: dict[str, HypothesisSpec],
    *,
    cost_model: CostModel,
) -> TrialRun:
    """Rebuild one stream on demand instead of retaining every trial."""

    if compact.spec.placebo_kind is None:
        return run_trial(prepared, compact.spec, cost_model=cost_model)
    parent_id = str(compact.spec.parameters.get("parent_id", ""))
    try:
        parent = real_by_id[parent_id]
    except KeyError as error:
        raise ValueError(f"control has unknown parent {parent_id!r}") from error
    real = run_trial(prepared, parent, cost_model=cost_model)
    return _control_trial(real, compact.spec, cost_model=cost_model)


def _bounded_family_p_values(
    matrix: np.memmap[Any, np.dtype[np.float64]],
    *,
    n_bootstrap: int,
    seed: int,
    workdir: Path,
    strategy_batch: int,
    bootstrap_batch: int,
    callbacks: tuple[ProgressCallback, ...],
    completed_trials: int,
    romano_wolf_alpha: float | None = None,
) -> _FamilyPValues:
    """Run family-wide tests without a strategies-by-reps resident tensor.

    Small problems use the pinned ``arch`` reference implementations directly.
    Large problems use the same stationary-bootstrap/recentering equations in
    model and replication chunks.  White uses least-favorable recentering;
    Hansen SPA applies the reference log-log relevance rule.
    """

    n_dates, n_models = matrix.shape
    if n_dates < 2 or n_models == 0:
        return _FamilyPValues(1.0, 1.0)
    reference_work = n_dates * n_models * n_bootstrap
    if reference_work <= 2_000_000:
        values = np.asarray(matrix, dtype=float)
        spa = hansen_spa(values, n_bootstrap=n_bootstrap, seed=seed)
        reality = white_reality_check(values, n_bootstrap=n_bootstrap, seed=seed)
        romano_wolf = (
            romano_wolf_stepdown(
                values,
                alpha=romano_wolf_alpha,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            if romano_wolf_alpha is not None
            else None
        )
        _emit_progress(
            callbacks,
            DiscoveryProgress("family_models", n_models, n_models, completed_trials),
        )
        return _FamilyPValues(
            spa.p_value,
            reality.p_value,
            "ARCH_REFERENCE",
            (romano_wolf.adjusted_p_values if romano_wolf is not None else None),
            romano_wolf.method if romano_wolf is not None else "NOT_REQUIRED",
        )

    means, variances = _family_moments(matrix, strategy_batch=strategy_batch)
    threshold = -np.sqrt(
        np.maximum(variances, 0.0)
        / n_dates
        * 2.0
        * math.log(max(math.log(n_dates), 1.0))
    )
    spa_centers = np.where(means >= threshold, means, 0.0)
    observed = float(means.max())
    spa_max = np.full(n_bootstrap, -math.inf, dtype=float)
    reality_max = np.full(n_bootstrap, -math.inf, dtype=float)
    romano_wolf_path = workdir / "romano-wolf-studentized.dat"
    romano_wolf_statistics: np.memmap[Any, np.dtype[np.float64]] | None = None
    standard_errors = np.sqrt(np.maximum(variances, 0.0) / n_dates)
    valid_standard_errors = np.isfinite(standard_errors) & (standard_errors > 0.0)
    if romano_wolf_alpha is not None:
        romano_wolf_statistics = np.memmap(
            romano_wolf_path,
            mode="w+",
            dtype=np.float64,
            shape=(n_bootstrap, n_models),
        )
    try:
        with _stationary_count_matrix(
            workdir,
            n_observations=n_dates,
            n_resamples=n_bootstrap,
            seed=seed,
            callbacks=callbacks,
            completed_trials=completed_trials,
        ) as counts:
            for start in range(0, n_models, strategy_batch):
                stop = min(start + strategy_batch, n_models)
                values = np.asarray(matrix[:, start:stop], dtype=float)
                for draw_start in range(0, n_bootstrap, bootstrap_batch):
                    draw_stop = min(draw_start + bootstrap_batch, n_bootstrap)
                    weights = np.asarray(counts[draw_start:draw_stop], dtype=float)
                    sampled = weights @ values / n_dates
                    spa_max[draw_start:draw_stop] = np.maximum(
                        spa_max[draw_start:draw_stop],
                        np.max(sampled - spa_centers[start:stop], axis=1),
                    )
                    reality_max[draw_start:draw_stop] = np.maximum(
                        reality_max[draw_start:draw_stop],
                        np.max(sampled - means[start:stop], axis=1),
                    )
                    if romano_wolf_statistics is not None:
                        romano_wolf_statistics[draw_start:draw_stop, start:stop] = (
                            np.divide(
                                sampled - means[start:stop],
                                standard_errors[start:stop],
                                out=np.zeros_like(sampled),
                                where=valid_standard_errors[start:stop],
                            )
                        )
                _emit_progress(
                    callbacks,
                    DiscoveryProgress(
                        "family_models", stop, n_models, completed_trials
                    ),
                )
        romano_wolf_adjusted: np.ndarray[Any, np.dtype[np.float64]] | None = None
        if romano_wolf_statistics is not None:
            romano_wolf_statistics.flush()
            observed_t = np.divide(
                means,
                standard_errors,
                out=np.where(
                    means > 0.0, math.inf, np.where(means < 0.0, -math.inf, 0.0)
                ),
                where=valid_standard_errors,
            )
            order = np.argsort(-observed_t, kind="stable")
            exceedances = np.zeros(n_models, dtype=np.int64)
            for draw_start in range(0, n_bootstrap, bootstrap_batch):
                draw_stop = min(draw_start + bootstrap_batch, n_bootstrap)
                ordered = np.asarray(
                    romano_wolf_statistics[draw_start:draw_stop, :], dtype=float
                )[:, order]
                maxima = np.maximum.accumulate(ordered[:, ::-1], axis=1)[:, ::-1]
                exceedances += np.count_nonzero(
                    maxima >= observed_t[order][None, :], axis=0
                )
            ordered_p = (1.0 + exceedances) / (n_bootstrap + 1.0)
            ordered_adjusted = np.maximum.accumulate(ordered_p).clip(0.0, 1.0)
            romano_wolf_adjusted = np.empty(n_models, dtype=float)
            romano_wolf_adjusted[order] = ordered_adjusted
    finally:
        _close_memmap(romano_wolf_statistics)
        romano_wolf_path.unlink(missing_ok=True)
    return _FamilyPValues(
        float(np.mean(spa_max > observed)),
        float(np.mean(reality_max > observed)),
        "BOUNDED_ARCH_EQUIVALENT",
        romano_wolf_adjusted,
        (
            "BOUNDED_STUDENTIZED_STATIONARY_MAX_T"
            if romano_wolf_alpha is not None
            else "NOT_REQUIRED"
        ),
    )


def _family_moments(
    matrix: np.memmap[Any, np.dtype[np.float64]],
    *,
    strategy_batch: int,
    average_block_length: float = 10.0,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """Compute arch-compatible stationary long-run variances by FFT chunks."""

    n_dates, n_models = matrix.shape
    means = np.empty(n_models, dtype=float)
    variances = np.empty(n_models, dtype=float)
    n_fft = 1 << (2 * n_dates - 1).bit_length()
    lags = np.arange(1, n_dates, dtype=float)
    continuation = 1.0 - 1.0 / average_block_length
    kappa = (1.0 - lags / n_dates) * continuation**lags
    kappa += (lags / n_dates) * continuation ** (n_dates - lags)
    for start in range(0, n_models, strategy_batch):
        stop = min(start + strategy_batch, n_models)
        values = np.asarray(matrix[:, start:stop], dtype=float)
        chunk_means = values.mean(axis=0)
        centered = values - chunk_means
        spectrum = np.fft.rfft(centered, n=n_fft, axis=0)
        autocovariance = np.fft.irfft(
            spectrum.conjugate() * spectrum,
            n=n_fft,
            axis=0,
        )[:n_dates]
        means[start:stop] = chunk_means
        variances[start:stop] = (
            autocovariance[0]
            + 2.0 * np.sum(kappa[:, None] * autocovariance[1:], axis=0)
        ) / n_dates
    return means, variances


@contextmanager
def _stationary_count_matrix(
    workdir: Path,
    *,
    n_observations: int,
    n_resamples: int,
    seed: int,
    callbacks: tuple[ProgressCallback, ...] = (),
    completed_trials: int = 0,
) -> Iterator[np.memmap[Any, np.dtype[Any]]]:
    """Materialize shared stationary draws as bounded integer date counts."""

    if n_observations < 1 or n_resamples < 1:
        raise ValueError("stationary draw dimensions must be positive")
    dtype: type[np.uint16] | type[np.uint32] = (
        np.uint16 if n_observations <= np.iinfo(np.uint16).max else np.uint32
    )
    path = workdir / f"stationary-{n_observations}-{n_resamples}-{seed}.dat"
    counts = np.memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=(n_resamples, n_observations),
    )
    try:
        # Draw generation has a canonical batching independent of the caller's
        # compute chunk, so memory tuning cannot change empirical results.
        for start in range(0, n_resamples, _CANONICAL_DRAW_BATCH):
            stop = min(start + _CANONICAL_DRAW_BATCH, n_resamples)
            draws = stationary_bootstrap_indices(
                n_observations,
                stop - start,
                average_block_length=10.0,
                seed=_batch_seed(seed, start),
            )
            for row, path_indices in enumerate(draws):
                counts[start + row] = np.bincount(
                    path_indices,
                    minlength=n_observations,
                )
            _emit_progress(
                callbacks,
                DiscoveryProgress("family_draws", stop, n_resamples, completed_trials),
            )
        counts.flush()
        yield counts
    finally:
        _close_memmap(counts)
        path.unlink(missing_ok=True)


def _apply_shared_bootstrap_intervals(
    prepared: PreparedResearch,
    cost_model: CostModel,
    compact: list[_CompactTrial],
    real_by_id: dict[str, HypothesisSpec],
    metrics: pd.DataFrame,
    *,
    plans: tuple[tuple[np.ndarray[Any, np.dtype[np.int64]], int], ...],
    seed: int,
    workdir: Path,
    bootstrap_batch: int,
    callbacks: tuple[ProgressCallback, ...],
) -> None:
    """Apply shared base/finalist draws while retaining one trial at a time."""

    total = sum(len(positions) for positions, _ in plans)
    completed = 0
    for positions, repetitions in plans:
        groups: dict[int, list[int]] = {}
        for raw_position in positions:
            position = int(raw_position)
            trial = _registered_trial(
                prepared,
                compact[position],
                real_by_id,
                cost_model=cost_model,
            )
            finite_count = int(np.isfinite(trial.result.net_returns).sum())
            if finite_count >= 2:
                groups.setdefault(finite_count, []).append(position)
            else:
                completed += 1
        for finite_count, grouped_positions in sorted(groups.items()):
            with _stationary_count_matrix(
                workdir,
                n_observations=finite_count,
                n_resamples=repetitions,
                seed=seed,
            ) as counts:
                for position in grouped_positions:
                    trial = _registered_trial(
                        prepared,
                        compact[position],
                        real_by_id,
                        cost_model=cost_model,
                    )
                    values = trial.result.net_returns
                    finite = values[np.isfinite(values)]
                    mean_bounds, sharpe_bounds = _bootstrap_bounds(
                        finite,
                        counts,
                        batch_size=bootstrap_batch,
                    )
                    metrics.loc[position, ["mean_ci_lower", "mean_ci_upper"]] = (
                        mean_bounds
                    )
                    metrics.loc[position, ["sharpe_ci_lower", "sharpe_ci_upper"]] = (
                        sharpe_bounds
                    )
                    completed += 1
                    _emit_progress(
                        callbacks,
                        DiscoveryProgress(
                            "confidence_intervals",
                            completed,
                            total,
                            len(compact),
                        ),
                    )


def _bootstrap_bounds(
    values: np.ndarray[Any, np.dtype[np.float64]],
    counts: np.memmap[Any, np.dtype[Any]],
    *,
    batch_size: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute mean/Sharpe percentiles from shared stationary date counts."""

    sample = np.asarray(values, dtype=float)
    n_observations = len(sample)
    if n_observations < 2 or counts.shape[1] != n_observations:
        raise ValueError("bootstrap values and shared counts are not aligned")
    means = np.empty(counts.shape[0], dtype=float)
    sharpes = np.empty(counts.shape[0], dtype=float)
    squared = sample**2
    for start in range(0, counts.shape[0], batch_size):
        stop = min(start + batch_size, counts.shape[0])
        weights = np.asarray(counts[start:stop], dtype=float)
        sums = weights @ sample
        sum_squares = weights @ squared
        means[start:stop] = sums / n_observations
        variance = (sum_squares - sums**2 / n_observations) / (n_observations - 1)
        deviation = np.sqrt(np.maximum(variance, 0.0))
        sharpes[start:stop] = np.divide(
            means[start:stop] * math.sqrt(252.0),
            deviation,
            out=np.full(stop - start, np.nan),
            where=deviation > 0.0,
        )
    finite_sharpes = sharpes[np.isfinite(sharpes)]
    if not finite_sharpes.size:
        sharpe_bounds = (math.nan, math.nan)
    else:
        sharpe_quantiles = np.quantile(finite_sharpes, [0.025, 0.975])
        sharpe_bounds = (float(sharpe_quantiles[0]), float(sharpe_quantiles[1]))
    mean_quantiles = np.quantile(means, [0.025, 0.975])
    return (
        (float(mean_quantiles[0]), float(mean_quantiles[1])),
        sharpe_bounds,
    )


def _batch_seed(seed: int, start: int) -> int:
    """Derive deterministic independent batch seeds without worker-order effects."""

    sequence = np.random.SeedSequence([seed, start])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _emit_progress(
    callbacks: tuple[ProgressCallback, ...], progress: DiscoveryProgress
) -> None:
    """Emit one immutable marker to each configured observer."""

    for callback in callbacks:
        callback(progress)


def _close_memmap(value: np.memmap[Any, np.dtype[Any]] | None) -> None:
    """Flush and close a NumPy memory map promptly, including on Windows."""

    if value is None:
        return
    value.flush()
    mapping = getattr(value, "_mmap", None)
    if mapping is not None:
        mapping.close()


def spec_from_dict(value: dict[str, Any]) -> HypothesisSpec:
    """Rehydrate a canonical JSON hypothesis declaration."""

    return HypothesisSpec(
        family=str(value["family"]),
        description=str(value["description"]),
        predicates=dict(value.get("predicates", {})),
        direction=Direction(value["direction"]),
        session=Session(value["session"]),
        holding_period=value["holding_period"],
        rationale=RationaleCategory(value.get("rationale", "none")),
        universe=str(value.get("universe", "sp500_current")),
        parameters=dict(value.get("parameters", {})),
        placebo_kind=value.get("placebo_kind"),
    )


def _calendar_trial_inputs(
    prepared: PreparedResearch, spec: HypothesisSpec
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    returns_frame = {
        Session.CLOSE_TO_CLOSE: prepared.close_returns,
        Session.OVERNIGHT: prepared.overnight_returns,
        Session.INTRADAY: prepared.intraday_returns,
    }[spec.session]
    symbols = list(returns_frame.columns)
    sector = spec.predicates.get("sector")
    if sector is not None:
        symbols = [
            symbol
            for symbol in symbols
            if prepared.sector_by_symbol.get(str(symbol)) == sector
        ]
    if not symbols:
        returns = np.full(len(prepared.dates), np.nan)
    else:
        returns = (
            returns_frame.loc[:, symbols].mean(axis=1, skipna=True).to_numpy(float)
        )
    mask = _calendar_predicate_mask(prepared, spec.predicates)
    holding = int(spec.holding_period) if isinstance(spec.holding_period, int) else 1
    # Calendar predicates describe the return's target session. The centralized
    # engine still applies its mandatory one-bar execution lag, so map the known-
    # in-advance target position back to the preceding decision row. This makes a
    # Monday predicate earn Monday's return rather than Tuesday's.
    target_position = np.convolve(mask.astype(float), np.ones(holding), mode="full")[
        : len(mask)
    ]
    direction = 1.0 if spec.direction is Direction.LONG else -1.0
    signal = np.zeros(len(mask), dtype=float)
    signal[:-1] = np.clip(target_position[1:], 0.0, 1.0) * direction
    return signal.astype(float), returns.astype(float)


def _calendar_predicate_mask(
    prepared: PreparedResearch, predicates: Mapping[str, str]
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Return the session mask selected by the declared calendar predicates."""

    mask = np.ones(len(prepared.dates), dtype=bool)
    weekday_names = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}
    for family, value in predicates.items():
        if family == "sector":
            continue
        if family == "weekday":
            selected = prepared.calendar["weekday"].to_numpy() == weekday_names[value]
        elif family == "month":
            selected = prepared.calendar["month"].to_numpy() == int(value)
        elif family == "turn_of_month":
            tom = prepared.calendar["turn_of_month"].to_numpy(dtype=bool)
            selected = tom if value == "TOM" else ~tom
        elif family == "holiday":
            column = "pre_holiday" if value == "PRE" else "post_holiday"
            selected = prepared.calendar[column].to_numpy(dtype=bool)
        elif family == "fomc":
            column = {
                "DAY_BEFORE": "fomc_event_day_before",
                "DAY_OF": "fomc_event_day_of",
                "EVENT_WEEK": "fomc_event_week",
            }[value]
            selected = prepared.calendar[column].to_numpy(dtype=bool)
        elif family == "opex":
            selected = prepared.calendar["opex_week"].to_numpy(dtype=bool)
        elif family in {"quarter_end", "month_end"}:
            flag = prepared.calendar[f"{family}_window"].to_numpy(dtype=bool)
            selected = flag if value == "WINDOW" else ~flag
        else:
            raise ValueError(f"unsupported predicate family {family}")
        mask &= selected
    return mask


def _spy_benchmark_returns(
    prepared: PreparedResearch,
    spec: HypothesisSpec,
) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    """Return the causally aligned SPY convention used for reporting only."""

    frame = {
        Session.CLOSE_TO_CLOSE: prepared.close_returns,
        Session.OVERNIGHT: prepared.overnight_returns,
        Session.INTRADAY: prepared.intraday_returns,
    }[spec.session]
    spy_column = next(
        (column for column in frame.columns if str(column).upper() == "SPY"),
        None,
    )
    if spy_column is None:
        return None
    return frame.loc[:, spy_column].to_numpy(dtype=float)


def _cross_sectional_feature(
    prepared: PreparedResearch, spec: HypothesisSpec
) -> pd.DataFrame:
    """Return the declared feature panel for one cross-sectional family.

    Extended families draw on volume and open prices; the ETF relative family
    is scored only on ETF columns so equities never enter its ranks.
    """

    if spec.family in {
        "momentum_12_1",
        "reversal_5d",
        "low_volatility",
        "high_52w_proximity",
    }:
        features = canonical_features(prepared.close)
        return {
            "momentum_12_1": features.momentum,
            "reversal_5d": features.reversal,
            "low_volatility": features.low_volatility,
            "high_52w_proximity": features.high_proximity,
        }[spec.family]
    if spec.family == "amihud_illiquidity":
        return amihud_illiquidity(
            prepared.close,
            prepared.volume,
            window=int(spec.parameters["window"]),
        )
    if spec.family == "max_lottery":
        return max_lottery(prepared.close, window=int(spec.parameters["window"]))
    if spec.family == "overnight_intraday_gap":
        return overnight_intraday_gap(
            prepared.open,
            prepared.close,
            window=int(spec.parameters["window"]),
        )
    if spec.family == "etf_relative_reversal":
        etf_columns = [
            column
            for column, asset_type in zip(
                prepared.close.columns, prepared.asset_types, strict=True
            )
            if asset_type == "etf"
        ]
        if not etf_columns:
            raise ValueError("etf_relative_reversal requires ETF columns")
        feature = short_term_reversal(
            prepared.close.loc[:, etf_columns],
            lookback=int(spec.parameters["lookback"]),
        )
        return feature.reindex(columns=prepared.close.columns)
    raise ValueError(f"unsupported cross-sectional family {spec.family}")


def _cross_sectional_trial_inputs(
    prepared: PreparedResearch, spec: HypothesisSpec
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    feature = _cross_sectional_feature(prepared, spec)
    weights = decile_weights(feature)
    gate_predicates = {
        key: value for key, value in spec.predicates.items() if key != "sector"
    }
    if gate_predicates:
        # Conditional combination candidate: the ranked entry is allowed only
        # when its FIRST earned session satisfies the declared calendar gate.
        # The exchange calendar is known in advance, so gating a decision row
        # by its own future fill session is causal. Gated candidates always
        # use the overlapping-cohort convention so the gate cannot silently
        # extend a holding period.
        gate = pd.Series(
            _calendar_predicate_mask(prepared, gate_predicates),
            index=weights.index,
        )
        entry_gate = gate.shift(
            -close_derived_execution_lag(spec.session), fill_value=False
        )
        weights = weights.where(entry_gate, 0.0)
        holding = (
            int(spec.holding_period) if isinstance(spec.holding_period, int) else 1
        )
        weights = pd.DataFrame(
            overlapping_cohort_targets(
                weights.to_numpy(dtype=float), holding_period=holding
            ),
            index=weights.index,
            columns=weights.columns,
        )
    elif spec.family == "momentum_12_1":
        periods = pd.Series(prepared.dates.to_period("M"), index=prepared.dates)
        rebalance = periods.ne(periods.shift(1))
        weights = weights.where(rebalance, np.nan).ffill().fillna(0.0)
    else:
        holding = (
            int(spec.holding_period) if isinstance(spec.holding_period, int) else 1
        )
        # Each daily rank starts an equal-sized cohort held for the declared
        # horizon. The portfolio is the average of its overlapping cohorts.
        weights = pd.DataFrame(
            overlapping_cohort_targets(
                weights.to_numpy(dtype=float), holding_period=holding
            ),
            index=weights.index,
            columns=weights.columns,
        )
    if spec.direction is Direction.SHORT:
        weights = -weights
    if spec.session is Session.CLOSE_TO_CLOSE:
        returns = prepared.close_returns
    elif spec.session is Session.OVERNIGHT:
        returns = prepared.overnight_returns
    else:
        returns = prepared.intraday_returns
    return weights.to_numpy(dtype=float), returns.to_numpy(dtype=float)


def _trial_execution_lag(spec: HypothesisSpec) -> int:
    """Map information timing and return convention to a row lag.

    Calendar schedules are known before the target session and their signals
    are deliberately materialized on the preceding decision row. All other
    currently registered families are close-derived: intraday can enter at the
    next open, while overnight and close-to-close must wait for the next close
    and can only earn the following return interval.
    """

    if spec.family == "calendar":
        return 1
    return close_derived_execution_lag(spec.session)


def _run_explicit(
    spec: HypothesisSpec,
    signal: np.ndarray[Any, np.dtype[np.float64]],
    returns: np.ndarray[Any, np.dtype[np.float64]],
    cost_model: CostModel,
    *,
    adv_dollars: float | np.ndarray[Any, np.dtype[np.float64]],
    asset_type: str | tuple[str, ...],
    benchmark_returns: np.ndarray[Any, np.dtype[np.float64]] | None,
) -> TrialRun:
    execution_lag = _trial_execution_lag(spec)
    gross, net, positions = vectorized_backtest(
        signal,
        returns,
        execution_lag=execution_lag,
        cost_model=cost_model,
        adv_dollars=adv_dollars,
        asset_type=asset_type,
    )
    holding = int(spec.holding_period) if isinstance(spec.holding_period, int) else 1
    statistics = summarize_returns(net, holding_period=holding)
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics = performance_metrics(
            net,
            positions=positions,
            benchmark=benchmark_returns,
        )
    result = BacktestResult(
        spec.hypothesis_id,
        gross,
        net,
        positions,
        statistics,
        metrics,
    )
    return TrialRun(
        spec,
        result,
        signal,
        returns,
        adv_dollars,
        asset_type,
        benchmark_returns,
        execution_lag,
    )


def _periodic_sharpe(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return math.nan
    deviation = float(finite.std(ddof=1))
    if deviation == 0.0:
        return math.copysign(math.inf, float(finite.mean())) if finite.mean() else 0.0
    return float(finite.mean() / deviation)


def _finite_or(value: float, fallback: float) -> float:
    return float(value) if math.isfinite(value) else fallback


def canonical_spec_payload(specs: tuple[HypothesisSpec, ...]) -> list[dict[str, Any]]:
    """Return deterministic JSON-ready declarations."""

    return [
        {
            **asdict(spec),
            "direction": spec.direction.value,
            "session": spec.session.value,
            "rationale": spec.rationale.value,
            "hypothesis_id": spec.hypothesis_id,
        }
        for spec in specs
    ]
