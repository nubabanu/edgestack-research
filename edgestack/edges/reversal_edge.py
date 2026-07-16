"""Promotion study for the exact five-name, half-gross reversal contract.

This module deliberately does not reuse the broad campaign's holdout runner.
It reconstructs the selected rule from canonical bars, runs missing family and
placebo safeguards, requires a real Zipline confirmation, and then uses the
global economic-window ledger before exposing any holdout return.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from importlib import metadata
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml

from edgestack.backtest.costs import CostAssumptions, CostModel
from edgestack.backtest.engine import (
    BacktestResult,
    aggregate_cross_sectional_returns,
    lag_positions,
    overlapping_cohort_targets,
    vectorized_backtest,
)
from edgestack.backtest.metrics import performance_metrics
from edgestack.backtest.zipline_adapter import (
    ZiplineCanonicalData,
    confirm_with_zipline,
)
from edgestack.config import ReversalResearchConfig
from edgestack.data.calendars import NYSECalendar
from edgestack.disclaimer import DISCLAIMER
from edgestack.edges.global_holdout import (
    GlobalHoldoutLedger,
    GlobalHoldoutRecord,
    global_scope_id,
)
from edgestack.models import Direction, GateResult, GateStatus, HypothesisSpec
from edgestack.pipeline.research import PreparedResearch, prepare_research
from edgestack.provenance import canonical_sha256, sha256_file, source_tree_sha256
from edgestack.reversal.portfolio import reversal_trial_specs, top_k_side_weights
from edgestack.stats.bootstrap import (
    stationary_bootstrap_ci,
    stationary_bootstrap_indices,
)
from edgestack.stats.deflated_sharpe import deflated_sharpe_ratio
from edgestack.stats.reality_check import hansen_spa, white_reality_check
from edgestack.stats.tests import hac_mean_test, summarize_returns
from edgestack.storage.catalog import Catalog
from edgestack.validation.decay import analyze_decay
from edgestack.validation.walkforward import expanding_walk_forward

HOLDOUT_PROGRAM_ID: Final = "EDGESTACK_US_EQUITY_RESEARCH_V1"
HOLDOUT_MARKET: Final = "XNYS_US_EQUITIES"
HOLDOUT_PROMOTION_CLASS: Final = "FINAL"


@dataclass(frozen=True, slots=True)
class ReversalComputation:
    """Exact vector state shared by validation, confirmation, and holdout."""

    prepared: PreparedResearch
    spec: HypothesisSpec
    score: pd.DataFrame
    entries: pd.DataFrame
    cohort_targets: np.ndarray[Any, np.dtype[np.float64]]
    positions: np.ndarray[Any, np.dtype[np.float64]]
    gross_returns: np.ndarray[Any, np.dtype[np.float64]]
    net_returns: np.ndarray[Any, np.dtype[np.float64]]
    benchmark_returns: np.ndarray[Any, np.dtype[np.float64]]
    adv_dollars: np.ndarray[Any, np.dtype[np.float64]]
    active: np.ndarray[Any, np.dtype[np.bool_]]


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reversal edge configuration must be a mapping")
    return cast(dict[str, Any], payload)


def _parse_date(value: object) -> date:
    return date.fromisoformat(str(value))


def _cost_model(config: Mapping[str, Any]) -> CostModel:
    costs = cast(Mapping[str, Any], config["costs"])
    return CostModel(
        CostAssumptions(
            portfolio_capital=float(costs["portfolio_capital_usd"]),
            commission_per_side=float(costs["commission_per_side_usd"]),
            equity_full_spread_bps=float(costs["equity_full_spread_bps"]),
            base_slippage_bps=float(costs["base_slippage_bps_per_fill"]),
            impact_coefficient_bps=float(costs["impact_coefficient_bps"]),
            max_impact_bps=float(costs["impact_cap_bps_per_fill"]),
            turnover_penalty_bps=float(
                costs["selection_penalty_bps_per_100pct_one_way_turnover"]
            ),
        )
    )


def _load_universe(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    frame = pq.read_table(path).to_pandas()
    if frame["symbol"].duplicated().any():
        raise ValueError("universe symbols must be unique")
    sectors = dict(
        zip(frame["symbol"].astype(str), frame["sector"].astype(str), strict=True)
    )
    asset_types = dict(
        zip(frame["symbol"].astype(str), frame["asset_type"].astype(str), strict=True)
    )
    return sectors, asset_types


def _filter_frozen_universe(
    bars: pd.DataFrame, asset_types: Mapping[str, str]
) -> pd.DataFrame:
    """Exclude any observation not named by the hash-frozen universe."""

    symbols = bars["symbol"].astype(str)
    filtered = bars.loc[symbols.isin(set(asset_types))].copy()
    if filtered.empty:
        raise ValueError("frozen universe has no matching canonical bars")
    if "asset_type" in filtered:
        expected = filtered["symbol"].astype(str).map(asset_types).str.lower()
        observed = filtered["asset_type"].astype(str).str.lower()
        if bool((expected != observed).any()):
            raise ValueError("canonical bars disagree with frozen universe asset types")
    return filtered


def _load_bars(
    path: Path,
    *,
    start: date,
    end_exclusive: date,
) -> pd.DataFrame:
    """Predicate-push the exact date interval before converting to pandas."""

    table = pq.read_table(
        path,
        filters=[
            ("session", ">=", datetime.combine(start, time.min)),
            ("session", "<", datetime.combine(end_exclusive, time.min)),
        ],
    )
    frame = table.to_pandas()
    if frame.empty:
        raise ValueError("canonical bar slice is empty")
    frame["session"] = (
        pd.to_datetime(frame["session"]).dt.tz_localize(None).dt.normalize()
    )
    minimum = cast(pd.Timestamp, frame["session"].min()).date()
    maximum = cast(pd.Timestamp, frame["session"].max()).date()
    if minimum < start or maximum >= end_exclusive:
        raise RuntimeError(
            "Parquet predicate exposed a date outside the authorized slice"
        )
    return cast(pd.DataFrame, frame)


def _selected_spec(config: Mapping[str, Any]) -> HypothesisSpec:
    strategy = cast(Mapping[str, Any], config["strategy"])
    research = ReversalResearchConfig(
        enabled=True,
        top_k=(3, 5, 10, 20, 50),
        variants=("raw", "sector_neutral", "market_sector_residual"),
        allow_survivorship_biased_diagnostic=True,
    )
    identifier = str(strategy["candidate_hypothesis_id"])
    matches = [
        spec
        for spec in reversal_trial_specs(research, point_in_time_universe=False)
        if spec.hypothesis_id == identifier
    ]
    if len(matches) != 1:
        raise RuntimeError("frozen reversal hypothesis ID is not in the declared grid")
    spec = matches[0]
    if (
        spec.direction is not Direction.LONG
        or spec.parameters.get("variant") != "raw"
        or int(spec.parameters.get("top_k", 0)) != int(strategy["top_k"])
    ):
        raise RuntimeError(
            "frozen hypothesis does not match the five-name long contract"
        )
    return spec


def _prepared(
    base: Path,
    config: Mapping[str, Any],
    *,
    start: date,
    end_exclusive: date,
) -> PreparedResearch:
    data = cast(Mapping[str, Any], config["data"])
    sectors, asset_types = _load_universe(base / str(data["universe_path"]))
    bars = _filter_frozen_universe(
        _load_bars(
            base / str(data["bars_path"]), start=start, end_exclusive=end_exclusive
        ),
        asset_types,
    )
    return prepare_research(
        bars,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end_exclusive - timedelta(days=1)),
        fomc_dates=pd.DatetimeIndex([]),
        sector_by_symbol=sectors,
    )


def _compute(
    prepared: PreparedResearch,
    config: Mapping[str, Any],
    *,
    cost_multiplier: float = 1.0,
    gross_exposure: float | None = None,
) -> ReversalComputation:
    strategy = cast(Mapping[str, Any], config["strategy"])
    exposure = (
        float(strategy["gross_exposure"])
        if gross_exposure is None
        else float(gross_exposure)
    )
    if not 0.0 < exposure <= 1.0:
        raise ValueError("gross exposure must be in (0, 1]")
    close = prepared.close
    equity = pd.Series(
        [kind == "equity" for kind in prepared.asset_types],
        index=close.columns,
        dtype=bool,
    )
    eligible = (
        pd.DataFrame(
            np.broadcast_to(equity.to_numpy(), close.shape),
            index=close.index,
            columns=close.columns,
        )
        & close.notna()
    )
    lookback = int(strategy["lookback_sessions"])
    score = -(close.div(close.shift(lookback)) - 1.0)
    entries = top_k_side_weights(
        score,
        top_k=int(strategy["top_k"]),
        direction=Direction.LONG,
        eligible=eligible,
    )
    cohorts = overlapping_cohort_targets(
        entries.to_numpy(dtype=float),
        holding_period=int(strategy["holding_sessions"]),
    )
    targets = np.asarray(cohorts * exposure, dtype=float)
    adv = close.mul(prepared.volume).rolling(20, min_periods=1).mean().shift(1)
    adv_values = np.nan_to_num(
        adv.to_numpy(dtype=float),
        nan=100_000_000.0,
        posinf=100_000_000.0,
        neginf=100_000_000.0,
    )
    desired_positions = lag_positions(targets, execution_lag=2)
    positions = _execute_with_fill_availability(
        desired_positions,
        close=prepared.close.to_numpy(dtype=float),
        volume=prepared.volume.to_numpy(dtype=float),
        gross_cap=exposure,
    )
    gross, net = _portfolio_stream(
        positions,
        asset_returns=prepared.close_returns.to_numpy(dtype=float),
        cost_model=_cost_model(config),
        asset_types=prepared.asset_types,
        adv_dollars=adv_values,
        cost_multiplier=cost_multiplier,
    )
    active = np.abs(positions).sum(axis=1) > 0.0
    gross = np.where(active, gross, np.nan)
    net = np.where(active, net, np.nan)
    spy = next(
        (column for column in prepared.close.columns if str(column).upper() == "SPY"),
        None,
    )
    if spy is None:
        raise ValueError("SPY is required for the frozen half-SPY benchmark")
    benchmark = prepared.close_returns[spy].to_numpy(dtype=float) * exposure
    benchmark = np.where(active, benchmark, np.nan)
    return ReversalComputation(
        prepared,
        _selected_spec(config),
        score,
        entries,
        targets,
        positions,
        gross,
        net,
        benchmark,
        adv_values,
        active,
    )


def _execute_with_fill_availability(
    desired_positions: np.ndarray[Any, np.dtype[np.float64]],
    *,
    close: np.ndarray[Any, np.dtype[np.float64]],
    volume: np.ndarray[Any, np.dtype[np.float64]],
    gross_cap: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Carry positions when the preceding close cannot supply a fill.

    Position row ``t`` earns the close-to-close return ending on ``t`` and is
    established at close ``t-1``. A missing/nonpositive close or zero volume at
    that fill session leaves the prior position unchanged. Locked exposure is
    reserved first and tradable targets are scaled only when needed to retain
    the preregistered portfolio gross cap.
    """

    desired = np.asarray(desired_positions, dtype=float)
    prices = np.asarray(close, dtype=float)
    volumes = np.asarray(volume, dtype=float)
    if (
        desired.ndim != 2
        or desired.shape != prices.shape
        or desired.shape != volumes.shape
    ):
        raise ValueError("desired positions, close, and volume must align")
    if not math.isfinite(gross_cap) or not 0.0 < gross_cap <= 1.0:
        raise ValueError("gross cap must be finite and in (0, 1]")
    executed = np.zeros_like(desired)
    for row in range(1, len(desired)):
        tradable = (
            np.isfinite(prices[row - 1])
            & (prices[row - 1] > 0.0)
            & np.isfinite(volumes[row - 1])
            & (volumes[row - 1] > 0.0)
        )
        locked = np.where(tradable, 0.0, executed[row - 1])
        requested = np.where(tradable, desired[row], 0.0)
        remaining = max(gross_cap - float(np.abs(locked).sum()), 0.0)
        requested_gross = float(np.abs(requested).sum())
        if requested_gross > remaining and requested_gross > 0.0:
            requested *= remaining / requested_gross
        executed[row] = locked + requested
    return executed


def _portfolio_stream(
    positions: np.ndarray[Any, np.dtype[np.float64]],
    *,
    asset_returns: np.ndarray[Any, np.dtype[np.float64]],
    cost_model: CostModel,
    asset_types: tuple[str, ...],
    adv_dollars: np.ndarray[Any, np.dtype[np.float64]],
    cost_multiplier: float,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    gross = aggregate_cross_sectional_returns(positions, asset_returns)
    all_missing = ~np.isfinite(asset_returns).any(axis=1)
    gross[all_missing] = np.nan
    costs = cost_model.portfolio_costs(
        positions,
        asset_type=asset_types,
        adv_dollars=adv_dollars,
        multiplier=cost_multiplier,
    )
    net = gross - costs
    net[~np.isfinite(gross)] = np.nan
    return gross, net


def _backtest_result(run: ReversalComputation) -> BacktestResult:
    statistics = summarize_returns(
        run.net_returns,
        holding_period=5,
        minimum_observations=100,
    )
    performance = performance_metrics(
        run.net_returns,
        positions=run.positions,
        benchmark=run.benchmark_returns,
    )
    return BacktestResult(
        run.spec.hypothesis_id,
        run.gross_returns,
        run.net_returns,
        run.positions,
        statistics,
        performance,
    )


def _finite_pair(
    left: np.ndarray[Any, np.dtype[np.float64]],
    right: np.ndarray[Any, np.dtype[np.float64]],
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    valid = np.isfinite(left) & np.isfinite(right)
    return left[valid], right[valid]


def _control_metrics(
    run: ReversalComputation,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    validation = cast(Mapping[str, Any], config["preholdout_validation"])
    costs = run.gross_returns - run.net_returns
    asset_returns = run.prepared.close_returns.to_numpy(dtype=float)
    seed_shuffle, seed_random = [int(value) for value in validation["placebo_seeds"]]
    permutation = np.random.default_rng(seed_shuffle).permutation(len(asset_returns))
    shuffled_gross = np.sum(
        np.where(
            np.isfinite(asset_returns[permutation]),
            run.positions * asset_returns[permutation],
            0.0,
        ),
        axis=1,
    )
    shuffled_net = np.where(run.active, shuffled_gross - costs, np.nan)

    equity = np.asarray(
        [kind == "equity" for kind in run.prepared.asset_types], dtype=bool
    )
    random_entries = np.zeros(run.entries.shape, dtype=float)
    rng = np.random.default_rng(seed_random)
    top_k = int(cast(Mapping[str, Any], config["strategy"])["top_k"])
    for row in range(len(random_entries)):
        choices = np.flatnonzero(
            equity & run.prepared.close.iloc[row].notna().to_numpy(dtype=bool)
        )
        if len(choices) >= top_k:
            selected = rng.choice(choices, size=top_k, replace=False)
            random_entries[row, selected] = 1.0 / top_k
    random_targets = overlapping_cohort_targets(
        random_entries,
        holding_period=int(
            cast(Mapping[str, Any], config["strategy"])["holding_sessions"]
        ),
    ) * float(cast(Mapping[str, Any], config["strategy"])["gross_exposure"])
    random_desired = lag_positions(random_targets, execution_lag=2)
    random_positions = _execute_with_fill_availability(
        random_desired,
        close=run.prepared.close.to_numpy(dtype=float),
        volume=run.prepared.volume.to_numpy(dtype=float),
        gross_cap=float(cast(Mapping[str, Any], config["strategy"])["gross_exposure"]),
    )
    _, random_net = _portfolio_stream(
        random_positions,
        asset_returns=asset_returns,
        cost_model=_cost_model(config),
        asset_types=run.prepared.asset_types,
        adv_dollars=run.adv_dollars,
        cost_multiplier=1.0,
    )
    random_active = np.abs(random_positions).sum(axis=1) > 0.0
    random_net = np.where(random_active, random_net, np.nan)

    def metrics(values: np.ndarray[Any, np.dtype[np.float64]]) -> dict[str, Any]:
        selected = values[np.isfinite(values)]
        summary = summarize_returns(selected, holding_period=5)
        periodic_sharpe = summary.annualized_sharpe / math.sqrt(252.0)
        dsr = deflated_sharpe_ratio(
            periodic_sharpe,
            n_observations=len(selected),
            n_trials=int(validation["conservative_global_trial_count"]),
            skewness=summary.skewness,
            kurtosis=summary.kurtosis,
        )
        provisional = bool(
            summary.mean > 0.0
            and summary.hac_t_stat > float(validation["directed_hac_t_minimum"])
            and dsr > float(validation["dsr_probability_minimum"])
        )
        return {
            "observations": len(selected),
            "mean": summary.mean,
            "hac_t": summary.hac_t_stat,
            "dsr_probability": dsr,
            "provisional_survivor": provisional,
        }

    return {
        "shuffled_date_returns": metrics(shuffled_net),
        "exposure_matched_random_signals": metrics(random_net),
    }


def _family_tests(base: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    data = cast(Mapping[str, Any], config["data"])
    validation = cast(Mapping[str, Any], config["preholdout_validation"])
    metrics = pd.read_parquet(base / str(data["parent_metrics_path"]))
    long_ids = (
        metrics.loc[metrics["direction"] == "LONG", "hypothesis_id"]
        .astype(str)
        .tolist()
    )
    streams = pd.read_parquet(
        base / str(data["parent_net_returns_path"]),
        columns=["session", *long_ids],
    ).set_index("session")
    aligned = streams.dropna(axis=0, how="any").to_numpy(dtype=float)
    white = white_reality_check(
        aligned,
        n_bootstrap=int(validation["reality_check_bootstrap_draws"]),
        seed=int(validation["family_test_seed"]),
    )
    spa = hansen_spa(
        aligned,
        n_bootstrap=int(validation["spa_bootstrap_draws"]),
        seed=int(validation["family_test_seed"]),
    )
    return {
        "family": "15_declared_LONG_reversal_alternatives",
        "observations": len(aligned),
        "models": len(long_ids),
        "white_reality_check": asdict(white),
        "hansen_spa": asdict(spa),
    }


def _parent_identity_and_match(
    base: Path,
    config: Mapping[str, Any],
    prepared: PreparedResearch,
) -> dict[str, Any]:
    data = cast(Mapping[str, Any], config["data"])
    candidate = str(
        cast(Mapping[str, Any], config["strategy"])["candidate_hypothesis_id"]
    )
    unscaled = _compute(prepared, config, gross_exposure=1.0)
    _, legacy_net, _ = vectorized_backtest(
        unscaled.cohort_targets,
        prepared.close_returns.to_numpy(dtype=float),
        execution_lag=2,
        cost_model=_cost_model(config),
        asset_type=prepared.asset_types,
        adv_dollars=unscaled.adv_dollars,
    )
    legacy_active = (
        np.abs(lag_positions(unscaled.cohort_targets, execution_lag=2)).sum(axis=1)
        > 0.0
    )
    legacy_net = np.where(legacy_active, legacy_net, np.nan)
    parent = pd.read_parquet(
        base / str(data["parent_net_returns_path"]),
        columns=["session", candidate],
    )
    dates_match = pd.DatetimeIndex(parent["session"]).equals(prepared.dates)
    values_match = bool(
        dates_match
        and np.allclose(
            parent[candidate].to_numpy(dtype=float),
            legacy_net,
            rtol=0.0,
            atol=1e-14,
            equal_nan=True,
        )
    )
    metric_frame = pd.read_parquet(base / str(data["parent_metrics_path"]))
    rows = metric_frame.loc[metric_frame["hypothesis_id"] == candidate]
    if len(rows) != 1:
        raise RuntimeError("parent metrics contain no unique selected candidate")
    row = rows.iloc[0].to_dict()
    return {
        "stream_reconstruction_match": values_match,
        "parent_metric": row,
        "artifact_sha256": {
            key: sha256_file(base / str(data[key]))
            for key in (
                "parent_metrics_path",
                "parent_specs_path",
                "parent_net_returns_path",
                "parent_gross_returns_path",
            )
        },
    }


def _confirmation(
    run: ReversalComputation,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    validation = cast(Mapping[str, Any], config["preholdout_validation"])
    confirmation_start = pd.Timestamp(str(validation["zipline_confirmation_start"]))
    eligible = np.flatnonzero(run.prepared.dates >= confirmation_start)
    if len(eligible) < 100:
        raise RuntimeError("Zipline-compatible confirmation window is too short")
    chunk_years = int(validation["zipline_chunk_years"])
    if chunk_years < 1:
        raise ValueError("Zipline confirmation chunk length must be positive")
    chunks: list[dict[str, Any]] = []
    chunk_start = confirmation_start
    final_date = run.prepared.dates[eligible[-1]]
    while chunk_start <= final_date:
        chunk_end = chunk_start + pd.DateOffset(years=chunk_years)
        selected = np.flatnonzero(
            (run.prepared.dates >= chunk_start) & (run.prepared.dates < chunk_end)
        )
        if len(selected) >= 100:
            chunks.append(_confirmation_chunk(run, config, selected))
        chunk_start = chunk_end
    if not chunks:
        raise RuntimeError("no eligible Zipline confirmation chunks")
    raw_passed = all(bool(chunk["passed"]) for chunk in chunks)
    passed = all(bool(chunk["normalized_pass"]) for chunk in chunks)
    exception_events = sum(
        int(chunk["normalized_exception_event_count"]) for chunk in chunks
    )
    return {
        "hypothesis_id": run.spec.hypothesis_id,
        "passed": passed,
        "raw_passed": raw_passed,
        "backend": "zipline-reloaded-3.1.1-chunked-in-memory-adjusted-ohlcv",
        "reason": (
            "all chunks agree within tolerance"
            if raw_passed
            else (
                "all chunks agree after frozen normalization of a documented "
                "backend calendar defect"
                if passed
                else "one or more chunks failed"
            )
        ),
        "trade_count": sum(int(chunk["trade_count"]) for chunk in chunks),
        "vector_trade_count": sum(int(chunk["vector_trade_count"]) for chunk in chunks),
        "timestamps_match": all(bool(chunk["timestamps_match"]) for chunk in chunks),
        "normalized_timestamps_match": all(
            bool(chunk["timestamps_match"])
            or int(chunk["normalized_exception_event_count"]) > 0
            for chunk in chunks
        ),
        "normalized_exception_event_count": exception_events,
        "maximum_difference_bps_per_trade": max(
            float(chunk["difference_bps_per_trade"]) for chunk in chunks
        ),
        "confirmation_start": chunks[0]["start"],
        "confirmation_end": chunks[-1]["end"],
        "confirmed_observations": sum(int(chunk["observations"]) for chunk in chunks),
        "reset_sessions_excluded": len(chunks),
        "chunks": chunks,
        "calendar_limitation": (
            "exchange_calendars 4.13.2 has 68 false pre-1970 XNYS sessions; "
            "the frozen 2000+ window matches canonical sessions, but its 2005-06-01 "
            "close is 16:00 ET instead of the documented 15:56 ET emergency close; "
            "raw and narrowly normalized results are both persisted"
        ),
        "numerical_policy": (
            "two-year chunks reset synthetic capital and exclude each first return; "
            "the frozen 100-million-dollar synthetic notional makes every ideal-weight "
            "rebalance executable at integer-share precision, while chunk resets prevent "
            "Zipline's 100-billion-share guard; this is not the paper portfolio size"
        ),
    }


def _normalize_known_zipline_calendar_exception(
    payload: dict[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize only fully matched, preregistered backend timestamp defects.

    Counts, securities, and numerical agreement still have to pass.  The raw
    Zipline verdict and mismatch events remain in the artifact, so this cannot
    convert an unknown discrepancy into a pass.
    """

    payload["raw_passed"] = bool(payload["passed"])
    payload["normalized_pass"] = bool(payload["passed"])
    payload["normalized_exception_event_count"] = 0
    payload["applied_calendar_exceptions"] = []
    if bool(payload["passed"]):
        return payload

    difference = payload.get("difference_bps_per_trade")
    tolerance = float(validation["zipline_tolerance_bps_per_trade"])
    if (
        bool(payload.get("timestamps_match", True))
        or not bool(payload.get("convention_supported", False))
        or int(payload.get("trade_count", -1))
        != int(payload.get("vector_trade_count", -2))
        or difference is None
        or not math.isfinite(float(difference))
        or float(difference) > tolerance
    ):
        return payload

    missing = Counter(
        (int(event[0]), str(event[1]))
        for event in cast(Sequence[Sequence[Any]], payload["missing_fill_events"])
    )
    extra = Counter(
        (int(event[0]), str(event[1]))
        for event in cast(Sequence[Sequence[Any]], payload["extra_fill_events"])
    )
    if not missing or not extra:
        return payload

    applied: list[dict[str, Any]] = []
    for raw_exception in cast(
        Sequence[Mapping[str, Any]],
        validation.get("zipline_known_calendar_exceptions", []),
    ):
        runtime_matches = bool(
            str(payload.get("backend")) == str(raw_exception["zipline_backend"])
            and metadata.version("zipline-reloaded")
            == str(raw_exception["zipline_version"])
            and f"pandas-market-calendars=={metadata.version('pandas-market-calendars')}"
            == str(raw_exception["canonical_calendar"])
            and f"exchange-calendars=={metadata.version('exchange-calendars')}"
            == str(raw_exception["backend_calendar"])
        )
        if (
            not runtime_matches
            or str(payload.get("start")) != str(raw_exception["chunk_start"])
            or str(payload.get("end")) != str(raw_exception["chunk_end"])
        ):
            continue
        expected = str(raw_exception["canonical_close_utc"])
        actual = str(raw_exception["backend_close_utc"])
        missing_sids = Counter(
            {
                sid: count
                for (sid, timestamp), count in missing.items()
                if timestamp == expected
            }
        )
        extra_sids = Counter(
            {
                sid: count
                for (sid, timestamp), count in extra.items()
                if timestamp == actual
            }
        )
        expected_count = int(raw_exception["expected_event_count"])
        if (
            missing_sids != extra_sids
            or sum(missing_sids.values()) != expected_count
            or sum(missing.values()) != expected_count
            or sum(extra.values()) != expected_count
        ):
            continue
        for sid, count in missing_sids.items():
            missing[(sid, expected)] -= count
            extra[(sid, actual)] -= count
        missing += Counter()
        extra += Counter()
        applied.append(
            {
                **dict(raw_exception),
                "matched_event_count": expected_count,
                "matched_local_asset_ids": sorted(missing_sids.elements()),
                "runtime_versions_verified": True,
            }
        )

    if missing or extra or not applied:
        return payload
    payload["normalized_pass"] = True
    payload["normalized_exception_event_count"] = sum(
        int(item["matched_event_count"]) for item in applied
    )
    payload["applied_calendar_exceptions"] = applied
    return payload


def _confirmation_chunk(
    run: ReversalComputation,
    config: Mapping[str, Any],
    selected: np.ndarray[Any, np.dtype[np.int64]],
) -> dict[str, Any]:
    """Confirm one bounded notional path through the real Zipline engine."""

    validation = cast(Mapping[str, Any], config["preholdout_validation"])
    positions = np.asarray(run.positions[selected], dtype=float).copy()
    # exchange_calendars 4.13.2 incorrectly labels 68 NYSE holidays as sessions
    # before 1970. The first selected row is a reset-only warm-up; all following
    # rows confirm original frozen targets without changing a signal.
    positions[0] = 0.0
    close_slice = run.prepared.close.iloc[selected]
    relevant = (np.abs(positions) > 0.0).any(axis=0) & close_slice.notna().any(
        axis=0
    ).to_numpy()
    if not np.any(relevant):
        raise RuntimeError("Zipline confirmation contains no targeted assets")
    relevant_columns = np.flatnonzero(relevant)
    positions = positions[:, relevant_columns]
    symbols = tuple(
        str(run.prepared.close.columns[index]) for index in relevant_columns
    )
    asset_returns = run.prepared.close_returns.iloc[
        selected, relevant_columns
    ].to_numpy(dtype=float)
    finite = np.isfinite(asset_returns)
    gross = np.sum(np.where(finite, positions * asset_returns, 0.0), axis=1)
    gross[~finite.any(axis=1)] = np.nan
    costs = _cost_model(config).portfolio_costs(
        positions,
        asset_type=tuple(run.prepared.asset_types[index] for index in relevant_columns),
        adv_dollars=run.adv_dollars[np.ix_(selected, relevant_columns)],
    )
    net = gross - costs
    net[~np.isfinite(gross)] = np.nan
    gross[0] = np.nan
    net[0] = np.nan
    benchmark = run.benchmark_returns[selected].copy()
    benchmark[0] = np.nan
    statistics = summarize_returns(net, holding_period=5)
    vector_result = BacktestResult(
        run.spec.hypothesis_id,
        gross,
        net,
        positions,
        statistics,
        performance_metrics(net, positions=positions, benchmark=benchmark),
    )
    result = confirm_with_zipline(
        run.spec,
        ZiplineCanonicalData(
            dates=run.prepared.dates[selected],
            symbols=symbols,
            open=run.prepared.open.iloc[selected, relevant_columns],
            high=run.prepared.high.iloc[selected, relevant_columns],
            low=run.prepared.low.iloc[selected, relevant_columns],
            close=close_slice.iloc[:, relevant_columns],
            volume=run.prepared.volume.iloc[selected, relevant_columns],
            target_positions=positions,
            asset_returns=asset_returns,
            cost_positions=positions,
            market_open=tuple(run.prepared.market_open[index] for index in selected),
            market_close=tuple(run.prepared.market_close[index] for index in selected),
            adv_dollars=run.adv_dollars[np.ix_(selected, relevant_columns)],
            asset_type=tuple(
                run.prepared.asset_types[index] for index in relevant_columns
            ),
        ),
        vector_result,
        _cost_model(config),
        tolerance_bps_per_trade=float(validation["zipline_tolerance_bps_per_trade"]),
        capital_base=float(validation["zipline_capital_base_usd"]),
    )
    payload = asdict(result)
    payload.update(
        {
            "start": run.prepared.dates[selected[0]].date().isoformat(),
            "end": run.prepared.dates[selected[-1]].date().isoformat(),
            "observations": len(selected) - 1,
        }
    )
    return _normalize_known_zipline_calendar_exception(payload, validation)


def _evaluate_preholdout(
    run: ReversalComputation,
    base: Path,
    config: Mapping[str, Any],
    *,
    run_zipline: bool,
) -> dict[str, Any]:
    validation = cast(Mapping[str, Any], config["preholdout_validation"])
    net = run.net_returns[np.isfinite(run.net_returns)]
    gross = run.gross_returns[np.isfinite(run.gross_returns)]
    benchmark_net, benchmark = _finite_pair(run.net_returns, run.benchmark_returns)
    excess = benchmark_net - benchmark
    summary = summarize_returns(net, holding_period=5)
    net_hac = hac_mean_test(net, holding_period=5, alternative="greater")
    excess_hac = hac_mean_test(excess, holding_period=5, alternative="greater")
    draws = stationary_bootstrap_indices(
        len(net),
        int(validation["bootstrap_draws"]),
        average_block_length=float(validation["stationary_average_block_sessions"]),
        seed=int(validation["bootstrap_seed"]),
    )
    net_ci = stationary_bootstrap_ci(
        net,
        statistic="mean",
        average_block_length=float(validation["stationary_average_block_sessions"]),
        indices=draws,
    )
    if len(excess) == len(net):
        excess_draws = draws
    else:
        excess_draws = stationary_bootstrap_indices(
            len(excess),
            int(validation["bootstrap_draws"]),
            average_block_length=float(validation["stationary_average_block_sessions"]),
            seed=int(validation["bootstrap_seed"]),
        )
    excess_ci = stationary_bootstrap_ci(
        excess,
        statistic="mean",
        average_block_length=float(validation["stationary_average_block_sessions"]),
        indices=excess_draws,
    )
    del draws, excess_draws
    periodic_sharpe = summary.annualized_sharpe / math.sqrt(252.0)
    global_dsr = deflated_sharpe_ratio(
        periodic_sharpe,
        n_observations=len(net),
        n_trials=int(validation["conservative_global_trial_count"]),
        skewness=summary.skewness,
        kurtosis=summary.kurtosis,
    )
    finite = np.isfinite(run.net_returns)
    dates = run.prepared.dates[finite]
    walk = expanding_walk_forward(
        run.net_returns[finite],
        dates,
        min_train_years=int(validation["walk_forward_min_train_years"]),
        test_years=int(validation["walk_forward_test_years"]),
        step_years=int(validation["walk_forward_step_years"]),
        holding_period=5,
        oos_t_threshold=float(validation["stitched_oos_hac_t_minimum"]),
        required_positive_fraction=float(
            validation["positive_test_year_fraction_minimum"]
        ),
    )
    decay = analyze_decay(
        run.net_returns[finite],
        dates,
        window_years=5,
        step_months=12,
        holding_period=5,
        minimum_observations=int(validation["minimum_daily_observations"]),
        stability_min=float(validation["stability_minimum"]),
    )
    sensitivities: dict[str, float] = {}
    for multiplier in cast(
        Sequence[float],
        cast(Mapping[str, Any], config["costs"])["sensitivity_multipliers"],
    ):
        scaled = _compute(
            run.prepared, config, cost_multiplier=float(multiplier)
        ).net_returns
        sensitivities[f"{float(multiplier):g}x"] = float(np.nanmean(scaled))
    parent = _parent_identity_and_match(base, config, run.prepared)
    parent_metric = cast(Mapping[str, Any], parent["parent_metric"])
    family = _family_tests(base, config)
    controls = _control_metrics(run, config)
    confirmation = (
        _confirmation(run, config)
        if run_zipline
        else {"passed": False, "reason": "SKIPPED_DIAGNOSTIC_RUN"}
    )
    p_limit = float(validation["family_p_value_maximum"])
    controls_pass = not any(
        bool(cast(Mapping[str, Any], item)["provisional_survivor"])
        for item in controls.values()
    )
    recent_ratio = decay.recent_to_prior_ratio
    checks = {
        "minimum_observations": len(net)
        >= int(validation["minimum_daily_observations"]),
        "directed_net_mean": float(net.mean()) > 0.0,
        "directed_hac_t": net_hac.t_stat > float(validation["directed_hac_t_minimum"]),
        "parent_global_bh": bool(parent_metric["bh_pass"]),
        "global_dsr": global_dsr > float(validation["dsr_probability_minimum"]),
        "bootstrap_net_lower_positive": net_ci.lower > 0.0,
        "benchmark_excess_positive": float(excess.mean()) > 0.0,
        "benchmark_excess_hac_t": excess_hac.t_stat
        > float(validation["benchmark_excess_hac_t_minimum"]),
        "walk_forward": walk.significant_oos and walk.majority_positive,
        "stability": decay.stability_score >= float(validation["stability_minimum"]),
        "recent_effect": recent_ratio is not None
        and recent_ratio >= float(validation["recent_to_prior_minimum"]),
        "side_pbo": float(parent_metric["side_pbo"])
        < float(validation["side_pbo_maximum"]),
        "four_x_cost": sensitivities["4x"] > 0.0,
        "white_reality_check": float(
            cast(Mapping[str, Any], family["white_reality_check"])["p_value"]
        )
        < p_limit,
        "hansen_spa": float(cast(Mapping[str, Any], family["hansen_spa"])["p_value"])
        < p_limit,
        "placebos": controls_pass,
        "parent_stream_reconstruction": bool(parent["stream_reconstruction_match"]),
        "zipline_confirmation": bool(confirmation["passed"]),
        "name_cap": float(np.nanmax(run.positions)) <= 0.100000000001,
        "gross_cap": float(np.nanmax(np.abs(run.positions).sum(axis=1)))
        <= float(cast(Mapping[str, Any], config["strategy"])["gross_exposure"]) + 1e-12,
    }
    return {
        "campaign_id": config["campaign_id"],
        "status": "PASS" if all(checks.values()) else "FAIL",
        "preholdout_pass": all(checks.values()),
        "bias_tier": "SURVIVORSHIP_BIASED",
        "selection_context": "POST_PREHOLDOUT_SELECTION_FROM_30_DECLARED_RULES",
        "observations": len(net),
        "start": dates[0].date().isoformat(),
        "end": dates[-1].date().isoformat(),
        "gross_mean": float(gross.mean()),
        "net_mean": float(net.mean()),
        "benchmark_excess_mean": float(excess.mean()),
        "net_hac": net_hac.as_dict(),
        "benchmark_excess_hac": excess_hac.as_dict(),
        "annualized_sharpe": summary.annualized_sharpe,
        "global_dsr_probability": global_dsr,
        "bootstrap_net_mean_95": asdict(net_ci),
        "bootstrap_benchmark_excess_mean_95": asdict(excess_ci),
        "walk_forward": {
            "stitched_oos_hac": walk.stitched_oos_test.as_dict(),
            "positive_test_fraction": walk.positive_fraction,
            "folds": len(walk.folds),
        },
        "decay": {
            "classification": decay.classification.value,
            "stability": decay.stability_score,
            "recent_to_prior": decay.recent_to_prior_ratio,
        },
        "cost_sensitivity_net_mean": sensitivities,
        "break_even_cost_multiplier": float(
            gross.mean() / max(gross.mean() - net.mean(), np.finfo(float).eps)
        ),
        "parent_evidence": parent,
        "family_tests": family,
        "placebo_controls": controls,
        "zipline_confirmation": confirmation,
        "checks": checks,
        "disclaimer": DISCLAIMER,
    }


def _write_json(
    path: Path, payload: Mapping[str, Any], *, durable: bool = False
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        if durable:
            os.fsync(handle.fileno())
    temporary.replace(path)
    return sha256_file(path)


def _artifact_root(base: Path, campaign_id: str) -> Path:
    return base / "artifacts" / "campaigns" / campaign_id


def run_preholdout(
    config_path: str | Path,
    *,
    root: str | Path = ".",
    run_zipline: bool = True,
) -> Path:
    """Rebuild the exact live contract using only preholdout rows."""

    base = Path(root).resolve()
    config_file = (base / config_path).resolve()
    config = _load_config(config_file)
    data = cast(Mapping[str, Any], config["data"])
    cutoff = _parse_date(data["preholdout_end_exclusive"])
    prepared = _prepared(
        base,
        config,
        start=_parse_date(data["research_start"]),
        end_exclusive=cutoff,
    )
    if prepared.dates[-1].date() >= cutoff:
        raise RuntimeError("preholdout computation exposed a holdout session")
    run = _compute(prepared, config)
    result = _evaluate_preholdout(run, base, config, run_zipline=run_zipline)
    input_identity = _preholdout_input_identity(base, config_file, config)
    result["config_sha256"] = input_identity["config_sha256"]
    result["preholdout_input_identity"] = input_identity
    artifact = _artifact_root(base, str(config["campaign_id"])) / "preholdout"
    path = artifact / "result.json"
    _write_json(path, result)
    latest = _latest_signal(run, config)
    _write_json(artifact / "latest_preholdout_signal.json", latest)
    print(DISCLAIMER)
    print(
        json.dumps(
            {
                "status": result["status"],
                "net_mean": result["net_mean"],
                "benchmark_excess_mean": result["benchmark_excess_mean"],
                "checks": result["checks"],
                "zipline": result["zipline_confirmation"],
            },
            indent=2,
            default=str,
        )
    )
    return path


def _holdout_scope_contract(config: Mapping[str, Any]) -> dict[str, str]:
    """Return the code-owned final scope and reject YAML attempts to rename it."""

    holdout = cast(Mapping[str, Any], config["holdout"])
    expected = {
        "program_id": HOLDOUT_PROGRAM_ID,
        "market": HOLDOUT_MARKET,
        "promotion_class": HOLDOUT_PROMOTION_CLASS,
    }
    observed = {key: str(holdout.get(key, "")) for key in expected}
    if observed != expected:
        raise RuntimeError("holdout economic-scope constants cannot be overridden")
    return expected


def _preholdout_input_identity(
    base: Path,
    config_file: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash every mutable input used to create preholdout promotion evidence."""

    data = cast(Mapping[str, Any], config["data"])
    manifest_path = base / str(data["data_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "data_snapshot_id": str(manifest["snapshot_id"]),
        "bars_sha256": sha256_file(base / str(data["bars_path"])),
        "universe_sha256": sha256_file(base / str(data["universe_path"])),
        "data_manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_file),
        "parent_artifact_sha256": {
            key: sha256_file(base / str(data[key]))
            for key in (
                "parent_metrics_path",
                "parent_specs_path",
                "parent_net_returns_path",
                "parent_gross_returns_path",
            )
        },
        "source_tree_sha256": source_tree_sha256(base),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "lock_sha256": sha256_file(base / "uv.lock"),
    }


def _freeze_identity(
    base: Path,
    config_file: Path,
    config: Mapping[str, Any],
    preholdout_path: Path,
) -> dict[str, Any]:
    data = cast(Mapping[str, Any], config["data"])
    inputs = _preholdout_input_identity(base, config_file, config)
    start = str(data["holdout_start"])
    end = str(data["holdout_end"])
    holdout = _holdout_scope_contract(config)
    scope = global_scope_id(
        program_id=holdout["program_id"],
        market=holdout["market"],
        promotion_class=holdout["promotion_class"],
        start=start,
        end=end,
    )
    return {
        "campaign_id": config["campaign_id"],
        "parent_campaign_id": config["parent_campaign_id"],
        "global_holdout_scope_id": scope,
        "holdout_program_id": holdout["program_id"],
        "holdout_market": holdout["market"],
        "promotion_class": holdout["promotion_class"],
        "holdout_start": start,
        "holdout_end": end,
        "data_snapshot_id": inputs["data_snapshot_id"],
        "bars_sha256": inputs["bars_sha256"],
        "universe_sha256": inputs["universe_sha256"],
        "data_manifest_sha256": inputs["data_manifest_sha256"],
        "config_sha256": inputs["config_sha256"],
        "preholdout_result_sha256": sha256_file(preholdout_path),
        "parent_artifact_sha256": inputs["parent_artifact_sha256"],
        "source_tree_sha256": inputs["source_tree_sha256"],
        "evaluator_sha256": inputs["evaluator_sha256"],
        "lock_sha256": inputs["lock_sha256"],
        "strategy_sha256": canonical_sha256(config["strategy"]),
        "selection_sha256": canonical_sha256(config["selection_rationale"]),
        "cost_sha256": canonical_sha256(config["costs"]),
        "threshold_sha256": canonical_sha256(config["preholdout_validation"]),
        "overlay_sha256": canonical_sha256({"enabled": []}),
        "bias_tier": "SURVIVORSHIP_BIASED",
    }


def _freeze_decision(
    identity: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Return every immutable model/policy field covered by ``freeze_id``."""

    strategy = cast(Mapping[str, Any], config["strategy"])
    return {
        **dict(identity),
        "edge_ids": [strategy["candidate_hypothesis_id"]],
        "stack": {
            "edge": "CURRENT_SP500_RAW_5D_REVERSAL_TOP5_HALF_GROSS",
            "weight": 1.0,
        },
        "enabled_overlays": [],
        "holdout_policy": dict(cast(Mapping[str, Any], config["holdout"])),
        "action_policy": dict(cast(Mapping[str, Any], config["action_policy"])),
        "second_holdout_evaluation": "FORBIDDEN",
        "disclaimer": DISCLAIMER,
    }


def freeze(config_path: str | Path, *, root: str | Path = ".") -> Path:
    """Freeze only a complete passing preholdout and register global scope."""

    base = Path(root).resolve()
    config_file = (base / config_path).resolve()
    config = _load_config(config_file)
    campaign_id = str(config["campaign_id"])
    artifact = _artifact_root(base, campaign_id)
    preholdout_path = artifact / "preholdout" / "result.json"
    preholdout = json.loads(preholdout_path.read_text(encoding="utf-8"))
    if preholdout.get("preholdout_pass") is not True:
        raise RuntimeError("cannot freeze: exact five-name contract failed preholdout")
    current_inputs = _preholdout_input_identity(base, config_file, config)
    if preholdout.get("preholdout_input_identity") != current_inputs:
        raise RuntimeError("cannot freeze: a preholdout input changed after validation")
    identity = _freeze_identity(base, config_file, config, preholdout_path)
    decision = _freeze_decision(identity, config)
    freeze_id = canonical_sha256(decision)
    freeze_path = artifact / "freeze" / "manifest.json"
    if freeze_path.exists():
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
        mismatches = [
            key for key, value in decision.items() if payload.get(key) != value
        ]
        expected_keys = set(decision) | {"freeze_id", "frozen_at"}
        if (
            mismatches
            or set(payload) != expected_keys
            or payload.get("freeze_id") != freeze_id
        ):
            raise RuntimeError("existing immutable freeze has another identity")
        freeze_sha = sha256_file(freeze_path)
    else:
        payload = {
            **decision,
            "freeze_id": freeze_id,
            "frozen_at": datetime.now(UTC).isoformat(),
        }
        freeze_sha = _write_json(freeze_path, payload)
    ledger = GlobalHoldoutLedger(base / "artifacts" / "edgestack.sqlite")
    registered = ledger.register(
        scope_id=str(identity["global_holdout_scope_id"]),
        program_id=str(identity["holdout_program_id"]),
        market=str(identity["holdout_market"]),
        promotion_class=str(identity["promotion_class"]),
        data_snapshot_id=str(identity["data_snapshot_id"]),
        start=str(identity["holdout_start"]),
        end=str(identity["holdout_end"]),
    )
    if registered.state != "UNSPENT" and registered.freeze_id != freeze_id:
        raise RuntimeError("global holdout scope belongs to another consumed freeze")
    catalog = Catalog(base / "artifacts" / "edgestack.sqlite")
    manifest = {
        "campaign_id": campaign_id,
        "parent_campaign_id": config["parent_campaign_id"],
        "freeze_id": freeze_id,
        "global_holdout_scope_id": identity["global_holdout_scope_id"],
        "bias_tier": "SURVIVORSHIP_BIASED",
    }
    existing = catalog.campaign(campaign_id)
    if existing is None:
        catalog.create_campaign(campaign_id, manifest)
    elif existing != manifest:
        raise RuntimeError("registered targeted campaign identity mismatch")
    catalog.record_gate(
        GateResult(
            campaign_id=campaign_id,
            phase="targeted_preholdout",
            status=GateStatus.PASS,
            checked_at=datetime.now(UTC),
            summary="exact five-name half-gross reversal contract passed frozen preholdout",
            evidence={"result_sha256": identity["preholdout_result_sha256"]},
        )
    )
    catalog.record_artifact(campaign_id, "targeted_freeze", freeze_sha, freeze_path)
    print(DISCLAIMER)
    print(
        json.dumps(
            {
                "freeze_id": freeze_id,
                "scope_id": identity["global_holdout_scope_id"],
                "manifest": str(freeze_path),
            },
            indent=2,
        )
    )
    return freeze_path


def _verified_freeze(
    base: Path, config_file: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    campaign_id = str(config["campaign_id"])
    path = _artifact_root(base, campaign_id) / "freeze" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    preholdout_path = _artifact_root(base, campaign_id) / "preholdout" / "result.json"
    identity = _freeze_identity(base, config_file, config, preholdout_path)
    expected = _freeze_decision(identity, config)
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    expected_keys = set(expected) | {"freeze_id", "frozen_at"}
    if mismatches:
        raise RuntimeError(
            f"freeze preflight failed before holdout access: {', '.join(mismatches)}"
        )
    if set(payload) != expected_keys:
        raise RuntimeError("freeze manifest contains unverified decision fields")
    if payload.get("freeze_id") != canonical_sha256(expected):
        raise RuntimeError("freeze ID is not canonical")
    freeze_sha = sha256_file(path)
    catalog = Catalog(base / "artifacts" / "edgestack.sqlite")
    if not catalog.artifact_registered(campaign_id, "targeted_freeze", freeze_sha):
        raise RuntimeError("freeze manifest SHA is not registered")
    return cast(dict[str, Any], payload)


def _latest_signal(
    run: ReversalComputation, config: Mapping[str, Any]
) -> dict[str, Any]:
    latest = run.prepared.dates[-1]
    equity = pd.Series(
        [kind == "equity" for kind in run.prepared.asset_types],
        index=run.prepared.close.columns,
        dtype=bool,
    )
    row = cast(pd.Series, run.score.loc[latest]).where(equity).dropna()
    row = row.sort_values(ascending=False, kind="stable")
    selected = row.head(int(cast(Mapping[str, Any], config["strategy"])["top_k"]))
    if len(selected) != int(cast(Mapping[str, Any], config["strategy"])["top_k"]):
        raise RuntimeError("latest signal has fewer than five eligible names")
    calendar = NYSECalendar()
    entry = calendar.next_session(latest)
    future = calendar.sessions(entry, entry + pd.Timedelta(days=20))
    earned = future[future > entry]
    exit_session = earned[
        int(cast(Mapping[str, Any], config["strategy"])["holding_sessions"]) - 1
    ]
    close = run.prepared.close
    previous = close.shift(
        int(cast(Mapping[str, Any], config["strategy"])["lookback_sessions"])
    )
    returns_5d = close.div(previous) - 1.0
    ranks = row.rank(method="min", pct=True)
    dsr_reliability = 1.0
    candidates: list[dict[str, Any]] = []
    for raw_symbol, raw_score in selected.items():
        symbol = str(raw_symbol)
        magnitude_percentile = float(ranks.get(raw_symbol, math.nan))
        trailing_return = float(
            cast(pd.Series, returns_5d.loc[latest]).get(raw_symbol, math.nan)
        )
        candidates.append(
            {
                "symbol": symbol,
                "rank": len(candidates) + 1,
                "trailing_5_session_return": trailing_return,
                "reversal_score": float(raw_score),
                "forecast_magnitude_percentile": magnitude_percentile,
                "confidence_ordinal": round(
                    100.0 * dsr_reliability * magnitude_percentile
                ),
                "new_account_target_weight": 0.10,
                "mature_new_cohort_weight": 0.02,
            }
        )
    return {
        "bias_tier": "SURVIVORSHIP_BIASED",
        "status": "SIGNAL_GENERATED_NOT_AN_ORDER",
        "signal_session": latest.date().isoformat(),
        "entry_session": entry.date().isoformat(),
        "entry_order": "MOC",
        "cohort_exit_session": exit_session.date().isoformat(),
        "exit_order": "MOC",
        "candidates": candidates,
        "shorts": [],
        "short_status": "DISABLED_FAILED_PREHOLDOUT_VALIDATION",
        "disclaimer": DISCLAIMER,
    }


def _evaluate_holdout(
    run: ReversalComputation,
    config: Mapping[str, Any],
    *,
    start: date,
    end: date,
) -> dict[str, Any]:
    dates = run.prepared.dates
    selected = (dates.date >= start) & (dates.date <= end)
    selected_dates = dates[selected]
    expected_dates = NYSECalendar().sessions(start, end)
    net_all = run.net_returns[selected]
    benchmark_all = run.benchmark_returns[selected]
    complete_returns = bool(
        np.isfinite(net_all).all() and np.isfinite(benchmark_all).all()
    )
    net = net_all
    benchmark = benchmark_all
    finite = np.isfinite(net) & np.isfinite(benchmark)
    net = net[finite]
    benchmark = benchmark[finite]
    if not len(net):
        raise RuntimeError("holdout contains no finite strategy observations")
    excess = net - benchmark
    terminal = float(np.prod(1.0 + net))
    net_hac = hac_mean_test(net, holding_period=5, alternative="greater")
    excess_hac = hac_mean_test(excess, holding_period=5, alternative="greater")
    name_cap = float(np.nanmax(run.positions[selected])) <= 0.100000000001
    gross_cap = float(np.nanmax(np.abs(run.positions[selected]).sum(axis=1))) <= (
        float(cast(Mapping[str, Any], config["strategy"])["gross_exposure"]) + 1e-12
    )
    exact_execution = bool(name_cap and gross_cap)
    checks = {
        "complete_session_coverage": selected_dates.equals(expected_dates),
        "complete_strategy_and_benchmark_returns": complete_returns,
        "net_mean_positive": float(net.mean()) > 0.0,
        "terminal_net_wealth_above_one": terminal > 1.0,
        "benchmark_excess_mean_positive": float(excess.mean()) > 0.0,
        "exact_execution_contract": exact_execution,
        "enabled_overlays_nonnegative": True,
    }
    return {
        "campaign_id": config["campaign_id"],
        "status": "PASS" if all(checks.values()) else "FAIL",
        "holdout_pass": all(checks.values()),
        "bias_tier": "SURVIVORSHIP_BIASED",
        "holdout_start": start.isoformat(),
        "holdout_end": end.isoformat(),
        "observations": len(net),
        "expected_sessions": len(expected_dates),
        "missing_return_sessions": int(len(net_all) - len(net)),
        "net_mean": float(net.mean()),
        "benchmark_mean": float(benchmark.mean()),
        "benchmark_excess_mean": float(excess.mean()),
        "net_hac_report_only": net_hac.as_dict(),
        "benchmark_excess_hac_report_only": excess_hac.as_dict(),
        "terminal_net_wealth": terminal,
        "terminal_benchmark_wealth": float(np.prod(1.0 + benchmark)),
        "checks": checks,
        "latest_frozen_snapshot_signal": _latest_signal(run, config),
        "enabled_overlays": [],
        "second_evaluation": "FORBIDDEN_REPLAY_ONLY",
        "disclaimer": DISCLAIMER,
    }


def _assert_record_matches_freeze(
    record: GlobalHoldoutRecord, freeze_payload: Mapping[str, Any]
) -> None:
    """Reject cross-campaign replay or consumption of another frozen model."""

    expected = (
        str(freeze_payload["holdout_program_id"]),
        str(freeze_payload["holdout_market"]),
        str(freeze_payload["promotion_class"]),
        str(freeze_payload["data_snapshot_id"]),
        str(freeze_payload["holdout_start"]),
        str(freeze_payload["holdout_end"]),
    )
    if (
        record.program_id,
        record.market,
        record.promotion_class,
        record.data_snapshot_id,
        record.start,
        record.end,
    ) != expected:
        raise RuntimeError("global holdout record metadata does not match the freeze")
    if record.state in {"CONSUMED", "SEALED"} and (
        record.freeze_id != str(freeze_payload["freeze_id"])
        or record.evaluator_sha256 != str(freeze_payload["evaluator_sha256"])
    ):
        raise RuntimeError("global holdout result belongs to another freeze/evaluator")


def _record_holdout_catalog(
    base: Path,
    *,
    campaign_id: str,
    scope_id: str,
    result_path: Path,
    result_sha: str,
    result: Mapping[str, Any],
) -> None:
    """Idempotently repair catalog evidence after the global result is sealed."""

    catalog = Catalog(base / "artifacts" / "edgestack.sqlite")
    catalog.record_artifact(campaign_id, "targeted_holdout", result_sha, result_path)
    passed = result.get("holdout_pass") is True
    status = str(result.get("status", "EVALUATION_ERROR"))
    summary = (
        "frozen five-name reversal passed the sole global holdout"
        if passed
        else f"frozen five-name reversal did not pass the sole global holdout ({status})"
    )
    catalog.record_gate(
        GateResult(
            campaign_id=campaign_id,
            phase="targeted_holdout",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            checked_at=datetime.now(UTC),
            summary=summary,
            evidence={"result_sha256": result_sha, "scope_id": scope_id},
        )
    )


def _replay_sealed_holdout(
    base: Path,
    *,
    campaign_id: str,
    freeze_path: Path,
    freeze_payload: Mapping[str, Any],
    record: GlobalHoldoutRecord,
) -> Path:
    """Replay signed stored evidence without re-reading bars or source hashes."""

    _assert_record_matches_freeze(record, freeze_payload)
    if not record.result_path or not record.result_sha256:
        raise RuntimeError("sealed global holdout lacks a result identity")
    path = Path(record.result_path)
    expected_path = _artifact_root(base, campaign_id) / "holdout" / "result.json"
    if path.resolve() != expected_path.resolve():
        raise RuntimeError("sealed global holdout points outside its campaign artifact")
    if sha256_file(path) != record.result_sha256:
        raise RuntimeError("sealed global holdout result was modified")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("freeze_id") != freeze_payload.get("freeze_id")
        or result.get("global_holdout_scope_id") != record.scope_id
        or result.get("campaign_id") != campaign_id
        or result.get("freeze_manifest_sha256") != sha256_file(freeze_path)
    ):
        raise RuntimeError("sealed result identity does not match its freeze manifest")
    _record_holdout_catalog(
        base,
        campaign_id=campaign_id,
        scope_id=record.scope_id,
        result_path=path,
        result_sha=record.result_sha256,
        result=cast(Mapping[str, Any], result),
    )
    print(DISCLAIMER)
    print(path.read_text(encoding="utf-8"))
    return path


def run_holdout(config_path: str | Path, *, root: str | Path = ".") -> Path:
    """Perform the sole global evaluation or replay its sealed result."""

    base = Path(root).resolve()
    config_file = (base / config_path).resolve()
    config = _load_config(config_file)
    campaign_id = str(config["campaign_id"])
    freeze_path = _artifact_root(base, campaign_id) / "freeze" / "manifest.json"
    freeze_payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    scope_id = str(freeze_payload["global_holdout_scope_id"])
    ledger = GlobalHoldoutLedger(base / "artifacts" / "edgestack.sqlite")
    record = ledger.get(scope_id)
    if record is None:
        raise RuntimeError("global holdout scope is not registered")
    if record.state == "SEALED":
        return _replay_sealed_holdout(
            base,
            campaign_id=campaign_id,
            freeze_path=freeze_path,
            freeze_payload=cast(Mapping[str, Any], freeze_payload),
            record=record,
        )
    if record.state == "CONSUMED":
        _assert_record_matches_freeze(record, freeze_payload)
        raise RuntimeError(
            "BURNED_NO_RESULT: global holdout was consumed and cannot rerun"
        )
    freeze_payload = _verified_freeze(base, config_file, config)
    _assert_record_matches_freeze(record, freeze_payload)
    catalog = Catalog(base / "artifacts" / "edgestack.sqlite")
    catalog.require_passed(campaign_id, ["targeted_preholdout"])
    ledger.consume(
        scope_id=scope_id,
        freeze_id=str(freeze_payload["freeze_id"]),
        evaluator_sha256=str(freeze_payload["evaluator_sha256"]),
    )
    result_path = _artifact_root(base, campaign_id) / "holdout" / "result.json"
    evaluation_error: Exception | None = None
    try:
        data = cast(Mapping[str, Any], config["data"])
        start = _parse_date(data["holdout_start"])
        end = _parse_date(data["holdout_end"])
        # Lookback state is loaded only after authorization.  It is not scored
        # and is required to carry cohorts across the holdout boundary.
        prepared = _prepared(
            base,
            config,
            start=start - timedelta(days=60),
            end_exclusive=end + timedelta(days=1),
        )
        run = _compute(prepared, config)
        result = _evaluate_holdout(run, config, start=start, end=end)
    except Exception as error:
        evaluation_error = error
        result = {
            "campaign_id": campaign_id,
            "status": "EVALUATION_ERROR",
            "holdout_pass": False,
            "bias_tier": "SURVIVORSHIP_BIASED",
            "error_type": type(error).__name__,
            "error": str(error),
            "checks": {"evaluation_completed": False},
            "disclaimer": DISCLAIMER,
        }
    result.update(
        {
            "freeze_id": freeze_payload["freeze_id"],
            "global_holdout_scope_id": scope_id,
            "freeze_manifest_sha256": sha256_file(freeze_path),
        }
    )
    result_sha = _write_json(result_path, result, durable=True)
    ledger.seal(
        scope_id=scope_id,
        freeze_id=str(freeze_payload["freeze_id"]),
        result_sha256=result_sha,
        result_path=result_path,
    )
    _record_holdout_catalog(
        base,
        campaign_id=campaign_id,
        scope_id=scope_id,
        result_path=result_path,
        result_sha=result_sha,
        result=result,
    )
    print(DISCLAIMER)
    print(json.dumps(result, indent=2, default=str))
    if evaluation_error is not None:
        raise RuntimeError(
            f"holdout evaluation failed and was sealed at {result_path}"
        ) from evaluation_error
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preholdout", "freeze", "holdout"))
    parser.add_argument("--config", default="configs/reversal-edge-v1.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--skip-zipline",
        action="store_true",
        help="diagnostic only; forces the preholdout gate to fail",
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "preholdout":
        run_preholdout(
            arguments.config,
            root=arguments.root,
            run_zipline=not arguments.skip_zipline,
        )
    elif arguments.command == "freeze":
        freeze(arguments.config, root=arguments.root)
    else:
        run_holdout(arguments.config, root=arguments.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
