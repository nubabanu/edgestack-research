"""Candidate validation, exhaustive verdict construction, and cost evidence."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from datetime import UTC
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from edgestack.backtest.confirm import (
    ConfirmationData,
    ConfirmationEngine,
    ConfirmationResult,
    zipline_available,
    zipline_backend_status,
)
from edgestack.backtest.costs import CostModel, break_even_cost_multiplier
from edgestack.config import EdgeStackConfig
from edgestack.evaluation.verdicts import VerdictInputs, classify_verdict
from edgestack.models import (
    DecayClass,
    EvidenceBundle,
    ExecutionStatus,
    HypothesisSpec,
    Verdict,
    VerdictRecord,
)
from edgestack.pipeline.research import PreparedResearch, run_trial
from edgestack.stats.bootstrap import stationary_bootstrap_ci
from edgestack.stats.multiple_testing import benjamini_hochberg
from edgestack.validation.cpcv import PBOResult, cpcv_pbo
from edgestack.validation.decay import DecayResult, analyze_decay, classify_decay
from edgestack.validation.regimes import (
    RegimeInteractionResult,
    causal_spy_ma200_regimes,
    trend_regime_interaction,
)
from edgestack.validation.walkforward import expanding_walk_forward


@dataclass(frozen=True, slots=True)
class ValidationBundle:
    """Full validation output including failed and unavailable declarations."""

    metrics: pd.DataFrame
    records: tuple[VerdictRecord, ...]
    validated_ids: tuple[str, ...]
    pbo: PBOResult
    placebo_fraction: float
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ConfirmationOutcome:
    """Structured persisted evidence from the required finalist backend."""

    passed: bool
    difference_bps: float
    backend: str
    executed: bool
    trade_count: int | None
    vector_trade_count: int | None
    timestamps_match: bool
    reason: str


def run_validation(
    prepared: PreparedResearch,
    config: EdgeStackConfig,
    specs: tuple[HypothesisSpec, ...],
    discovery_metrics: pd.DataFrame,
    net_streams: pd.DataFrame,
    gross_streams: pd.DataFrame,
) -> ValidationBundle:
    """Apply expanding OOS, CPCV, decay, cost, and confirmation gates."""

    del gross_streams  # gross evidence is rebuilt with the frozen cost model below
    by_id = {spec.hypothesis_id: spec for spec in specs}
    candidates = tuple(
        discovery_metrics.loc[
            discovery_metrics["discovery_survivor"].astype(bool)
            & discovery_metrics["placebo_kind"].isna(),
            "hypothesis_id",
        ].astype(str)
    )
    if len(candidates) >= 2:
        candidate_matrix = np.nan_to_num(
            net_streams.loc[:, list(candidates)].to_numpy(dtype=float), nan=0.0
        )
        pbo = cpcv_pbo(
            candidate_matrix,
            n_groups=config.validation.cpcv_groups,
            n_test_groups=config.validation.cpcv_test_groups,
            purge=config.validation.purge_sessions,
            embargo=config.validation.embargo_sessions,
        )
    else:
        pbo = PBOResult(
            None,
            np.array([], dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=float),
            0,
            False,
            "PBO needs at least two discovery survivors",
        )
    placebo_mask = discovery_metrics["placebo_kind"].notna()
    placebo_count = int(placebo_mask.sum())
    placebo_survivors = int(
        discovery_metrics.loc[placebo_mask, "discovery_survivor_pre_spa"]
        .astype(bool)
        .sum()
    )
    placebo_fraction = placebo_survivors / placebo_count if placebo_count else 0.0
    cost_model = CostModel(config.costs)
    rows = discovery_metrics.copy()
    for column, default in (
        ("oos_t", np.nan),
        ("oos_positive_fraction", np.nan),
        ("stability_score", np.nan),
        ("stability_same_sign_periods", 0),
        ("stability_eligible_periods", 0),
        ("fixed_stability_periods", 0),
        ("rolling_stability_periods", 0),
        ("recent_to_prior_ratio", np.nan),
        ("decay", DecayClass.INSUFFICIENT.value),
        ("trend_regime_available", False),
        ("trend_regime_source", ""),
        ("trend_regime_reason", "not tested"),
        ("regime_active", ""),
        ("regime_current", ""),
        ("regime_currently_active", False),
        ("regime_active_observations", 0),
        ("regime_inactive_observations", 0),
        ("regime_active_mean", np.nan),
        ("regime_inactive_mean", np.nan),
        ("regime_active_t", np.nan),
        ("regime_interaction_t", np.nan),
        ("regime_interaction_p", np.nan),
        ("regime_interaction_adjusted_p", np.nan),
        ("vix_regime_available", False),
        (
            "vix_regime_reason",
            "VIX and available_at are not part of PreparedResearch; no VIX evidence inferred",
        ),
        ("pbo", pbo.pbo if pbo.defined else np.nan),
        ("confirmation_difference_bps", np.nan),
        ("confirmation_pass", False),
        ("confirmation_executed", False),
        ("confirmation_backend", "not_run"),
        ("confirmation_trade_count", np.nan),
        ("confirmation_vector_trade_count", np.nan),
        ("confirmation_timestamps_match", False),
        ("confirmation_reason", "not run"),
        ("cost_sensitivity", "{}"),
        ("break_even_cost_multiplier", np.nan),
    ):
        rows[column] = default

    trend_regimes = causal_spy_ma200_regimes(prepared.close)
    decay_by_id: dict[str, DecayResult] = {}
    regime_by_id: dict[str, RegimeInteractionResult] = {}
    for hypothesis_id in candidates:
        spec = by_id[hypothesis_id]
        trial = run_trial(prepared, spec, cost_model=cost_model)
        stream = trial.result.net_returns
        holding = (
            int(spec.holding_period) if isinstance(spec.holding_period, int) else 1
        )
        walk = expanding_walk_forward(
            stream,
            prepared.dates,
            min_train_years=config.validation.min_train_years,
            test_years=config.validation.test_years,
            step_years=config.validation.step_years,
            holding_period=holding,
            oos_t_threshold=config.validation.oos_t,
            required_positive_fraction=config.validation.oos_positive_fraction,
        )
        decay = analyze_decay(
            stream,
            prepared.dates,
            window_years=config.validation.rolling_years,
            step_months=12,
            holding_period=holding,
            minimum_observations=config.grid.min_observations,
            stability_min=config.validation.stability_min,
        )
        regime = trend_regime_interaction(
            stream,
            trend_regimes,
            holding_period=holding,
            minimum_observations=config.grid.min_observations,
        )
        decay_by_id[hypothesis_id] = decay
        regime_by_id[hypothesis_id] = regime
        confirmation = _confirm_trial(prepared, trial, cost_model)
        asset_type, adv_dollars = _trial_cost_context(trial)
        sensitivity = cost_model.sensitivity(
            trial.result.gross_returns,
            trial.result.positions,
            multipliers=config.costs.sensitivity_multipliers,
            asset_type=asset_type,
            adv_dollars=adv_dollars,
        )
        sensitivity_means = {
            f"{scale:g}x": float(np.nanmean(values))
            for scale, values in sensitivity.items()
        }
        baseline_cost = float(
            np.nanmean(trial.result.gross_returns - trial.result.net_returns)
        )
        row = rows["hypothesis_id"] == hypothesis_id
        rows.loc[row, "oos_t"] = walk.stitched_oos_test.t_stat
        rows.loc[row, "oos_positive_fraction"] = walk.positive_fraction
        rows.loc[row, "stability_score"] = decay.stability_score
        rows.loc[row, "stability_same_sign_periods"] = decay.same_sign_periods
        rows.loc[row, "stability_eligible_periods"] = decay.eligible_periods
        rows.loc[row, "fixed_stability_periods"] = len(decay.fixed_points)
        rows.loc[row, "rolling_stability_periods"] = len(decay.points)
        rows.loc[row, "recent_to_prior_ratio"] = decay.recent_to_prior_ratio
        rows.loc[row, "decay"] = decay.classification.value
        rows.loc[row, "trend_regime_available"] = regime.available
        rows.loc[row, "trend_regime_source"] = regime.source
        rows.loc[row, "trend_regime_reason"] = regime.reason
        rows.loc[row, "regime_active"] = regime.active_regime or ""
        rows.loc[row, "regime_current"] = regime.current_regime or ""
        rows.loc[row, "regime_currently_active"] = regime.currently_active
        rows.loc[row, "regime_active_observations"] = regime.active_observations
        rows.loc[row, "regime_inactive_observations"] = regime.inactive_observations
        rows.loc[row, "regime_active_mean"] = regime.active_mean
        rows.loc[row, "regime_inactive_mean"] = regime.inactive_mean
        rows.loc[row, "regime_active_t"] = regime.active_t
        rows.loc[row, "regime_interaction_t"] = regime.interaction_t
        rows.loc[row, "regime_interaction_p"] = regime.p_value
        rows.loc[row, "confirmation_difference_bps"] = confirmation.difference_bps
        rows.loc[row, "confirmation_pass"] = confirmation.passed
        rows.loc[row, "confirmation_backend"] = confirmation.backend
        rows.loc[row, "confirmation_executed"] = confirmation.executed
        rows.loc[row, "confirmation_trade_count"] = confirmation.trade_count
        rows.loc[row, "confirmation_vector_trade_count"] = (
            confirmation.vector_trade_count
        )
        rows.loc[row, "confirmation_timestamps_match"] = confirmation.timestamps_match
        rows.loc[row, "confirmation_reason"] = confirmation.reason
        rows.loc[row, "cost_sensitivity"] = str(sensitivity_means)
        rows.loc[row, "break_even_cost_multiplier"] = break_even_cost_multiplier(
            float(np.nanmean(trial.result.gross_returns)), baseline_cost
        )
        # The finalist CI is recomputed at the frozen 10k/full profile count.
        mean_ci = stationary_bootstrap_ci(
            stream,
            statistic="mean",
            n_resamples=config.stats.finalist_bootstrap_reps,
            seed=config.stats.seed,
        )
        sharpe_ci = stationary_bootstrap_ci(
            stream,
            statistic="sharpe",
            n_resamples=config.stats.finalist_bootstrap_reps,
            seed=config.stats.seed,
        )
        rows.loc[row, ["mean_ci_lower", "mean_ci_upper"]] = [
            mean_ci.lower,
            mean_ci.upper,
        ]
        rows.loc[row, ["sharpe_ci_lower", "sharpe_ci_upper"]] = [
            sharpe_ci.lower,
            sharpe_ci.upper,
        ]

    # The interaction family is adjusted once across every eligible validation
    # interaction; no hypothesis can use an unadjusted p-value for promotion.
    eligible_regime_ids = tuple(
        hypothesis_id
        for hypothesis_id in candidates
        if regime_by_id[hypothesis_id].p_value is not None
    )
    if eligible_regime_ids:
        interaction_p_values = [
            cast(float, regime_by_id[hypothesis_id].p_value)
            for hypothesis_id in eligible_regime_ids
        ]
        regime_fdr = benjamini_hochberg(
            interaction_p_values,
            q=config.stats.fdr_q,
        )
        for hypothesis_id, adjusted_p in zip(
            eligible_regime_ids, regime_fdr.adjusted_p_values, strict=True
        ):
            regime = regime_by_id[hypothesis_id].with_adjusted_p(float(adjusted_p))
            base_decay = decay_by_id[hypothesis_id]
            decay = classify_decay(
                base_decay.points,
                fixed_points=base_decay.fixed_points,
                stability_min=config.validation.stability_min,
                regime_interaction_adjusted_p=regime.adjusted_p_value,
                active_regime_t=regime.active_t,
            )
            row = rows["hypothesis_id"] == hypothesis_id
            rows.loc[row, "decay"] = decay.classification.value
            rows.loc[row, "regime_interaction_adjusted_p"] = adjusted_p
            rows.loc[row, "regime_currently_active"] = regime.currently_active
            rows.loc[row, "trend_regime_reason"] = (
                "global BH-adjusted interaction available"
            )

    records = tuple(_records(rows, pbo))
    validated = tuple(
        record.hypothesis_id
        for record in records
        if record.verdict is Verdict.WORKS
        and record.execution_status is ExecutionStatus.TESTED
        and record.hypothesis_id in candidates
    )
    reasons: list[str] = []
    if placebo_fraction > config.stats.placebo_survival_max:
        reasons.append(
            f"provisional placebo survival {placebo_fraction:.3%} exceeds "
            f"{config.stats.placebo_survival_max:.3%}"
        )
    if pbo.defined and pbo.pbo is not None and pbo.pbo >= config.validation.pbo_max:
        reasons.append(
            f"candidate-set PBO {pbo.pbo:.3f} is not below {config.validation.pbo_max:.3f}"
        )
    if any(
        record.verdict is Verdict.WORKS
        for record in records
        if by_id.get(record.hypothesis_id) is not None
        and by_id[record.hypothesis_id].placebo_kind is not None
    ):
        reasons.append("a placebo received provisional WORKS")
    unavailable_confirmation = rows.loc[
        rows["hypothesis_id"].isin(candidates)
        & ~rows["confirmation_executed"].astype(bool)
    ]
    if not unavailable_confirmation.empty:
        reasons.append(
            "Zipline-reloaded finalist confirmation was not executed for all "
            "discovery survivors"
        )
    passed = not reasons
    return ValidationBundle(
        rows,
        records,
        validated,
        pbo,
        placebo_fraction,
        passed,
        tuple(reasons),
    )


def _records(metrics: pd.DataFrame, pbo: PBOResult) -> list[VerdictRecord]:
    records: list[VerdictRecord] = []
    for raw_row in metrics.to_dict(orient="records"):
        row: dict[str, Any] = {str(key): value for key, value in raw_row.items()}
        hypothesis_id = str(row["hypothesis_id"])
        evidence = _evidence_from_row(row, pbo)
        is_placebo = bool(row.get("placebo_kind"))
        confirmation_unavailable = (
            bool(row.get("discovery_survivor", False))
            and not is_placebo
            and not bool(row.get("confirmation_executed", False))
        )
        if bool(row.get("empty_signal", False)):
            status = ExecutionStatus.INVALID
        elif confirmation_unavailable:
            status = ExecutionStatus.DATA_UNAVAILABLE
        elif int(row["sample_size"]) >= 100:
            status = ExecutionStatus.TESTED
        else:
            status = ExecutionStatus.UNDERPOWERED
        decay = DecayClass(str(row.get("decay", DecayClass.INSUFFICIENT.value)))
        gates = VerdictInputs(
            bh_pass=bool(row.get("bh_pass", False)),
            deflated_sharpe_pass=bool(row.get("dsr_pass", False)),
            spa_pass=bool(row.get("spa_pass", False)),
            net_cost_pass=float(row["net_mean"]) > 0.0,
            event_confirmation_pass=bool(row.get("confirmation_pass", False)),
            regime_currently_active=bool(row.get("regime_currently_active", False)),
            holdout_opened=False,
            is_placebo=is_placebo,
        )
        records.append(
            classify_verdict(
                hypothesis_id,
                evidence,
                gates,
                execution_status=status,
                decay=decay,
            )
        )
    records.append(
        classify_verdict(
            "pead-data-unavailable",
            None,
            VerdictInputs(False, False, False, False, False),
            execution_status=ExecutionStatus.DATA_UNAVAILABLE,
            decay=DecayClass.INSUFFICIENT,
        )
    )
    records.append(
        classify_verdict(
            "recent-intraday-exploratory",
            None,
            VerdictInputs(False, False, False, False, False),
            execution_status=ExecutionStatus.DATA_UNAVAILABLE,
            decay=DecayClass.INSUFFICIENT,
        )
    )
    return records


def _evidence_from_row(row: dict[str, Any], pbo: PBOResult) -> EvidenceBundle:
    annotations = {
        "bh_adjusted_p": row.get("bh_adjusted_p"),
        "hac_lags": row.get("hac_lags"),
        "cost_sensitivity": row.get("cost_sensitivity", "{}"),
        "break_even_cost_multiplier": row.get("break_even_cost_multiplier"),
        "confirmation_backend": row.get("confirmation_backend", "not_run"),
        "confirmation_executed": bool(row.get("confirmation_executed", False)),
        "confirmation_pass": bool(row.get("confirmation_pass", False)),
        "confirmation_trade_count": _optional_int(row.get("confirmation_trade_count")),
        "confirmation_vector_trade_count": _optional_int(
            row.get("confirmation_vector_trade_count")
        ),
        "confirmation_timestamps_match": bool(
            row.get("confirmation_timestamps_match", False)
        ),
        "confirmation_reason": row.get("confirmation_reason", "not run"),
        "zipline_reloaded_available": zipline_available(),
        "stability_same_sign_periods": int(row.get("stability_same_sign_periods", 0)),
        "stability_eligible_periods": int(row.get("stability_eligible_periods", 0)),
        "fixed_stability_periods": int(row.get("fixed_stability_periods", 0)),
        "rolling_stability_periods": int(row.get("rolling_stability_periods", 0)),
        "recent_to_prior_ratio": _optional_float(row.get("recent_to_prior_ratio")),
        "trend_regime_available": bool(row.get("trend_regime_available", False)),
        "trend_regime_source": row.get("trend_regime_source", ""),
        "trend_regime_reason": row.get("trend_regime_reason", "not tested"),
        "regime_active": row.get("regime_active") or None,
        "regime_current": row.get("regime_current") or None,
        "regime_currently_active": bool(row.get("regime_currently_active", False)),
        "regime_active_observations": int(row.get("regime_active_observations", 0)),
        "regime_inactive_observations": int(row.get("regime_inactive_observations", 0)),
        "regime_active_mean": _optional_float(row.get("regime_active_mean")),
        "regime_inactive_mean": _optional_float(row.get("regime_inactive_mean")),
        "regime_active_t": _optional_float(row.get("regime_active_t")),
        "regime_interaction_t": _optional_float(row.get("regime_interaction_t")),
        "regime_interaction_p": _optional_float(row.get("regime_interaction_p")),
        "regime_interaction_adjusted_p": _optional_float(
            row.get("regime_interaction_adjusted_p")
        ),
        "vix_regime_available": bool(row.get("vix_regime_available", False)),
        "vix_regime_reason": row.get(
            "vix_regime_reason",
            "VIX and available_at are not part of PreparedResearch",
        ),
        "placebo_kind": row.get("placebo_kind"),
        "parent_id": row.get("parent_id"),
    }
    pbo_value = pbo.pbo if pbo.defined else None
    return EvidenceBundle(
        hypothesis_id=str(row["hypothesis_id"]),
        sample_size=int(row["sample_size"]),
        gross_mean=float(row["gross_mean"]),
        net_mean=float(row["net_mean"]),
        hac_t=float(row["hac_t"]),
        p_value=float(row["p_value"]),
        sharpe=float(row["sharpe"]),
        probabilistic_sharpe=float(row["probabilistic_sharpe"]),
        deflated_sharpe_probability=float(row["deflated_sharpe_probability"]),
        hit_rate=float(row["hit_rate"]),
        max_drawdown=float(row["max_drawdown"]),
        turnover=float(row["turnover"]),
        exposure=float(row["exposure"]),
        skew=float(row["skew"]),
        kurtosis=float(row["kurtosis"]),
        mean_ci=(float(row["mean_ci_lower"]), float(row["mean_ci_upper"])),
        sharpe_ci=(
            float(row.get("sharpe_ci_lower", math.nan)),
            float(row.get("sharpe_ci_upper", math.nan)),
        ),
        oos_t=_optional_float(row.get("oos_t")),
        oos_positive_fraction=_optional_float(row.get("oos_positive_fraction")),
        stability_score=_optional_float(row.get("stability_score")),
        pbo=pbo_value,
        confirmation_difference_bps=_optional_float(
            row.get("confirmation_difference_bps")
        ),
        annotations=annotations,
    )


def _confirm_trial(
    prepared: PreparedResearch,
    trial: Any,
    cost_model: CostModel,
) -> _ConfirmationOutcome:
    result = trial.result
    asset_type, adv_dollars = _trial_cost_context(trial)
    zipline = zipline_backend_status()
    version = zipline.version or "unavailable"
    zipline_label = f"zipline_reloaded_{version}_not_executed: {zipline.reason}"
    if zipline.executable:
        try:
            from edgestack.backtest.zipline_adapter import confirm_with_zipline

            confirmation = confirm_with_zipline(
                trial.spec,
                _zipline_canonical_data(prepared, trial),
                result,
                cost_model,
                tolerance_bps_per_trade=1.0,
            )
        except Exception as error:
            return _ConfirmationOutcome(
                False,
                math.inf,
                f"zipline-reloaded-{version}-runtime-failure",
                False,
                None,
                None,
                False,
                f"{type(error).__name__}: {error}",
            )
        return _outcome(confirmation, executed=True)
    if trial.signal.ndim == 1:
        if not isinstance(asset_type, str):
            raise ValueError("scalar finalist confirmation requires one asset type")
        timestamps = tuple(
            pd.Timestamp(value).tz_localize(UTC).to_pydatetime()
            for value in prepared.dates
        )
        confirmation = ConfirmationEngine(tolerance_bps_per_trade=1.0).confirm(
            trial.spec,
            ConfirmationData(
                trial.signal,
                trial.underlying_returns,
                timestamps,
                adv_dollars=adv_dollars,
                asset_type=asset_type,
            ),
            result,
            cost_model,
        )
        return _ConfirmationOutcome(
            False,
            float(confirmation.difference_bps_per_trade or 0.0),
            f"{confirmation.backend}+{zipline_label}",
            False,
            confirmation.trade_count,
            confirmation.vector_trade_count,
            confirmation.timestamps_match,
            zipline.reason,
        )
    # Deliberately sequential cross-sectional arithmetic, independent from the
    # vectorized aggregation path. Costs are still delegated to the frozen model.
    signal = np.nan_to_num(trial.signal, nan=0.0)
    returns = trial.underlying_returns
    positions = np.zeros_like(signal)
    positions[1:] = signal[:-1]
    gross = np.nansum(np.where(np.isfinite(returns), positions * returns, 0.0), axis=1)
    gross[~np.isfinite(returns).any(axis=1)] = np.nan
    net = gross - cost_model.portfolio_costs(
        positions,
        asset_type=asset_type,
        adv_dollars=adv_dollars,
    )
    difference = abs(float(np.nanmean(net)) - float(np.nanmean(result.net_returns)))
    trade_count = max(int(np.count_nonzero(np.diff(positions, axis=0))), 1)
    difference_bps = difference * 10_000 * len(net) / trade_count
    return _ConfirmationOutcome(
        False,
        difference_bps,
        f"independent_cross_sectional_loop+{zipline_label}",
        False,
        trade_count,
        trade_count,
        True,
        zipline.reason,
    )


def _zipline_canonical_data(prepared: PreparedResearch, trial: Any) -> Any:
    """Build exact canonical OHLCV/target inputs without synthetic prices."""

    from edgestack.backtest.zipline_adapter import ZiplineCanonicalData

    positions = np.asarray(trial.result.positions, dtype=float)
    per_asset_returns: np.ndarray[Any, np.dtype[np.float64]]
    target: np.ndarray[Any, np.dtype[np.float64]]
    cost_positions: np.ndarray[Any, np.dtype[np.float64]]
    returns_frame = {
        "close_to_close": prepared.close_returns,
        "overnight": prepared.overnight_returns,
        "intraday": prepared.intraday_returns,
    }[trial.spec.session.value]
    if positions.ndim == 1:
        sector = trial.spec.predicates.get("sector")
        symbols = tuple(
            str(symbol)
            for symbol in prepared.close.columns
            if sector is None or prepared.sector_by_symbol.get(str(symbol)) == sector
        )
        if not symbols:
            raise ValueError("scalar finalist has no canonical assets")
        per_asset_returns = np.asarray(
            returns_frame.loc[:, list(symbols)].to_numpy(float), dtype=float
        )
        valid = np.isfinite(per_asset_returns)
        counts = valid.sum(axis=1)
        target = np.zeros_like(per_asset_returns)
        nonempty = counts > 0
        target[nonempty] = np.where(
            valid[nonempty],
            positions[nonempty, None] / counts[nonempty, None],
            0.0,
        )
        cost_positions = positions
    elif positions.ndim == 2:
        symbols = tuple(str(symbol) for symbol in prepared.close.columns)
        per_asset_returns = np.asarray(trial.underlying_returns, dtype=float)
        target = positions
        cost_positions = positions
    else:
        raise ValueError("finalist positions must be one- or two-dimensional")
    columns = list(symbols)
    return ZiplineCanonicalData(
        dates=prepared.dates,
        symbols=symbols,
        open=prepared.open.loc[:, columns],
        high=prepared.high.loc[:, columns],
        low=prepared.low.loc[:, columns],
        close=prepared.close.loc[:, columns],
        volume=prepared.volume.loc[:, columns],
        target_positions=np.asarray(target, dtype=float),
        asset_returns=np.asarray(per_asset_returns, dtype=float),
        cost_positions=np.asarray(cost_positions, dtype=float),
        market_open=prepared.market_open,
        market_close=prepared.market_close,
        adv_dollars=trial.adv_dollars,
        asset_type=trial.asset_type,
    )


def _outcome(result: ConfirmationResult, *, executed: bool) -> _ConfirmationOutcome:
    """Translate the public backend result to persisted validation evidence."""

    return _ConfirmationOutcome(
        result.passed,
        float(result.difference_bps_per_trade or 0.0),
        result.backend,
        executed,
        result.trade_count,
        result.vector_trade_count,
        result.timestamps_match,
        result.reason,
    )


def _trial_cost_context(
    trial: Any,
) -> tuple[
    Literal["equity", "etf"] | tuple[str, ...],
    float | np.ndarray[Any, np.dtype[np.float64]],
]:
    """Read the exact frozen execution context attached to a trial."""

    raw_asset_type = getattr(trial, "asset_type", "equity")
    if isinstance(raw_asset_type, str):
        normalized = raw_asset_type.lower()
        if normalized not in {"equity", "etf"}:
            raise ValueError(f"unsupported trial asset type {normalized!r}")
        asset_type: Literal["equity", "etf"] | tuple[str, ...] = cast(
            Literal["equity", "etf"], normalized
        )
    else:
        asset_type = tuple(str(value).lower() for value in raw_asset_type)
        if not asset_type or any(
            value not in {"equity", "etf"} for value in asset_type
        ):
            raise ValueError("trial asset types must contain only equity/etf")
    raw_adv = np.asarray(getattr(trial, "adv_dollars", 100_000_000.0), dtype=float)
    if raw_adv.ndim == 0:
        adv_dollars: float | np.ndarray[Any, np.dtype[np.float64]] = float(raw_adv)
    else:
        adv_dollars = raw_adv
    return asset_type, adv_dollars


def _optional_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _optional_int(value: Any) -> int | None:
    converted = _optional_float(value)
    return int(converted) if converted is not None else None


def final_records(
    provisional: tuple[VerdictRecord, ...],
    holdout_means: dict[str, float],
    *,
    evaluated_ids: set[str],
    holdout_cis: dict[str, tuple[float, float]] | None = None,
) -> tuple[VerdictRecord, ...]:
    """Apply the one-use holdout evidence without rebuilding any model input."""

    output: list[VerdictRecord] = []
    for record in provisional:
        evidence = record.evidence
        if evidence is None:
            output.append(replace(record, provisional=False))
            continue
        holdout = holdout_means.get(record.hypothesis_id)
        annotations = dict(evidence.annotations)
        if holdout_cis and record.hypothesis_id in holdout_cis:
            annotations["holdout_mean_ci"] = holdout_cis[record.hypothesis_id]
        updated = replace(
            evidence,
            holdout_mean=holdout,
            annotations=annotations,
        )
        if record.hypothesis_id not in evaluated_ids:
            output.append(replace(record, evidence=updated, provisional=False))
            continue
        gates = VerdictInputs(
            bh_pass=record.verdict in {Verdict.WORKS, Verdict.WEAK},
            deflated_sharpe_pass=record.verdict in {Verdict.WORKS, Verdict.WEAK},
            spa_pass=record.verdict in {Verdict.WORKS, Verdict.WEAK},
            net_cost_pass=updated.net_mean > 0.0,
            event_confirmation_pass=(
                bool(updated.annotations.get("confirmation_executed", False))
                and bool(updated.annotations.get("confirmation_pass", False))
                and updated.confirmation_difference_bps is not None
                and updated.confirmation_difference_bps <= 1.0
            ),
            regime_currently_active=bool(
                updated.annotations.get("regime_currently_active", False)
            ),
            holdout_opened=True,
        )
        output.append(
            classify_verdict(
                record.hypothesis_id,
                updated,
                gates,
                execution_status=record.execution_status,
                decay=record.decay,
                bias_tier=record.bias_tier,
            )
        )
    return tuple(output)


def serialize_records(records: tuple[VerdictRecord, ...]) -> list[dict[str, Any]]:
    """Serialize immutable records for exact report replay."""

    return [asdict(record) for record in records]


def records_from_payload(payload: list[dict[str, Any]]) -> tuple[VerdictRecord, ...]:
    """Rehydrate records persisted for exact provisional/final report replay."""

    output: list[VerdictRecord] = []
    for item in payload:
        evidence_payload = item.get("evidence")
        evidence = (
            EvidenceBundle(**evidence_payload)
            if isinstance(evidence_payload, dict)
            else None
        )
        verdict_value = item.get("verdict")
        output.append(
            VerdictRecord(
                hypothesis_id=str(item["hypothesis_id"]),
                execution_status=ExecutionStatus(item["execution_status"]),
                verdict=Verdict(verdict_value) if verdict_value else None,
                decay=DecayClass(item["decay"]),
                reasons=tuple(item.get("reasons", ())),
                evidence=evidence,
                provisional=bool(item["provisional"]),
                bias_tier=str(item.get("bias_tier", "SURVIVORSHIP_BIASED")),
            )
        )
    return tuple(output)
