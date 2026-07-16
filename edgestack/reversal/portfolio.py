"""Preregistered top-K reversal portfolios evaluated as one search family."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from edgestack.backtest.costs import DEFAULT_ADV_FALLBACK_DOLLARS, CostModel
from edgestack.backtest.engine import overlapping_cohort_targets, vectorized_backtest
from edgestack.backtest.metrics import performance_metrics
from edgestack.config import ReversalResearchConfig
from edgestack.features.reversal import ReversalVariant, reversal_signal_set
from edgestack.models import Direction, HypothesisSpec, RationaleCategory, Session
from edgestack.stats.deflated_sharpe import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from edgestack.stats.multiple_testing import bonferroni, discovery_gauntlet
from edgestack.stats.tests import hac_mean_test, summarize_returns
from edgestack.validation.cpcv import PBOResult, cpcv_pbo
from edgestack.validation.decay import analyze_decay
from edgestack.validation.walkforward import expanding_walk_forward

if TYPE_CHECKING:
    from edgestack.pipeline.research import PreparedResearch


@dataclass(frozen=True, slots=True)
class ReversalGridResult:
    """Complete breadth/variant family with aligned evidence and PBO."""

    dates: pd.DatetimeIndex
    specs: tuple[HypothesisSpec, ...]
    metrics: pd.DataFrame
    net_returns: pd.DataFrame
    gross_returns: pd.DataFrame
    pbo: PBOResult
    pbo_by_side: dict[str, PBOResult]
    bias_tier: str
    trial_count: int


def reversal_trial_specs(
    config: ReversalResearchConfig,
    *,
    point_in_time_universe: bool,
) -> tuple[HypothesisSpec, ...]:
    """Declare every breadth, variant, and side before any result is observed."""

    universe = "sp500_point_in_time" if point_in_time_universe else "sp500_current"
    specs: list[HypothesisSpec] = []
    for variant in config.variants:
        for top_k in config.top_k:
            for direction in (Direction.LONG, Direction.SHORT):
                specs.append(
                    HypothesisSpec(
                        family=f"reversal_{variant}_topk",
                        description=(
                            f"{direction.value} top-{top_k} {variant} reversal"
                        ),
                        predicates={},
                        direction=direction,
                        session=Session.CLOSE_TO_CLOSE,
                        holding_period=config.holding_sessions,
                        rationale=RationaleCategory.MICROSTRUCTURE,
                        universe=universe,
                        parameters={
                            "variant": variant,
                            "top_k": top_k,
                            "lookback": config.lookback_sessions,
                            "holding": config.holding_sessions,
                            "beta_window": config.beta_window,
                            "beta_min_observations": config.beta_min_observations,
                            "residual_vol_window": config.residual_vol_window,
                            "side_specific": True,
                        },
                    )
                )
    identifiers = [spec.hypothesis_id for spec in specs]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("reversal trial identity collision")
    return tuple(specs)


def top_k_side_weights(
    signal: pd.DataFrame,
    *,
    top_k: int,
    direction: Direction,
    eligible: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build an equal-weight long-only or short-only extreme-tail portfolio."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    values = signal.astype(float)
    if eligible is None:
        allowed = values.notna()
    else:
        aligned = eligible.reindex(index=values.index, columns=values.columns)
        if aligned.isna().any(axis=None):
            raise ValueError("eligible mask must cover every signal cell")
        allowed = aligned.astype(bool) & values.notna()
    ranked = values.where(allowed).rank(
        axis=1,
        method="first",
        ascending=direction is Direction.SHORT,
    )
    selected = ranked.le(top_k)
    available = allowed.sum(axis=1)
    selected = selected.where(available >= top_k, False)
    counts = selected.sum(axis=1).replace(0, np.nan)
    magnitude = selected.div(counts, axis=0).fillna(0.0)
    return magnitude if direction is Direction.LONG else -magnitude


def _unannualized_sharpe(annualized: float) -> float:
    return annualized / math.sqrt(252.0) if math.isfinite(annualized) else annualized


def run_reversal_grid(
    prepared: PreparedResearch,
    config: ReversalResearchConfig,
    *,
    cost_model: CostModel,
    membership: pd.DataFrame | None = None,
    minimum_observations: int = 100,
    directed_t_threshold: float = 3.0,
    fdr_q: float = 0.05,
    dsr_threshold: float = 0.95,
    cpcv_groups: int = 6,
    cpcv_test_groups: int = 2,
    purge: int = 21,
    embargo: int = 21,
    min_train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
    oos_t_threshold: float = 2.0,
    required_positive_fraction: float = 0.5,
    rolling_years: int = 5,
    stability_min: float = 0.75,
    cost_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> ReversalGridResult:
    """Evaluate the complete top-K family without selecting a preferred K.

    A current-membership run is permitted only when explicitly labeled as a
    diagnostic.  This prevents the convenient local dataset from silently
    becoming promotion-grade point-in-time evidence.
    """

    if not config.enabled:
        raise ValueError("reversal research protocol is not enabled")
    if not cost_multipliers or any(value < 0.0 for value in cost_multipliers):
        raise ValueError("cost multipliers must be non-empty and non-negative")
    if (
        membership is None
        and config.require_point_in_time_universe
        and not config.allow_survivorship_biased_diagnostic
    ):
        raise ValueError(
            "point-in-time membership is required; enable only the explicitly "
            "survivorship-biased diagnostic override to inspect current members"
        )
    close = prepared.close
    if membership is not None:
        membership_mask = membership.reindex(index=close.index, columns=close.columns)
        if membership_mask.isna().any(axis=None):
            raise ValueError(
                "point-in-time membership does not cover the research panel"
            )
        membership_mask = membership_mask.astype(bool)
    else:
        membership_mask = None
    equity_columns = pd.Series(
        [asset_type == "equity" for asset_type in prepared.asset_types],
        index=close.columns,
        dtype=bool,
    )
    equity_eligible = (
        pd.DataFrame(
            np.broadcast_to(equity_columns.to_numpy(), close.shape),
            index=close.index,
            columns=close.columns,
        )
        & close.notna()
    )
    if membership_mask is not None:
        equity_eligible &= membership_mask
    spy = next(
        (column for column in close.columns if str(column).upper() == "SPY"), None
    )
    market_returns = (
        prepared.close_returns[spy]
        if spy is not None
        else prepared.close_returns.where(equity_eligible).mean(axis=1)
    )
    signal_set = reversal_signal_set(
        close,
        prepared.sector_by_symbol,
        market_returns=market_returns,
        membership=membership_mask,
        lookback=config.lookback_sessions,
        beta_window=config.beta_window,
        beta_min_observations=config.beta_min_observations,
        residual_vol_window=config.residual_vol_window,
    )
    point_in_time = membership_mask is not None
    specs = reversal_trial_specs(config, point_in_time_universe=point_in_time)
    adv = prepared.close.mul(prepared.volume).rolling(20, min_periods=1).mean().shift(1)
    adv_values = np.nan_to_num(
        adv.to_numpy(dtype=float),
        nan=DEFAULT_ADV_FALLBACK_DOLLARS,
        posinf=DEFAULT_ADV_FALLBACK_DOLLARS,
    )
    spread_matrix: np.ndarray[Any, np.dtype[np.float64]] | None = None
    if config.spread_source == "MEASURED_HL_FLOOR_V2":
        from edgestack.data.spreads import (
            floored_spread_matrix,
            monthly_median_spread_bps,
        )

        baseline = pd.Series(
            [
                (
                    cost_model.assumptions.etf_full_spread_bps
                    if asset_type == "etf"
                    else cost_model.assumptions.equity_full_spread_bps
                )
                for asset_type in prepared.asset_types
            ],
            index=close.columns,
            dtype=float,
        )
        spread_matrix = floored_spread_matrix(
            monthly_median_spread_bps(prepared.high, prepared.low, close),
            pd.DatetimeIndex(prepared.dates),
            baseline_bps=baseline,
        ).to_numpy(dtype=float)
    asset_returns = prepared.close_returns.to_numpy(dtype=float)
    benchmark = market_returns.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    net_by_id: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
    gross_by_id: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
    sharpes: list[float] = []
    temporary: list[tuple[HypothesisSpec, Any, Any, float]] = []
    for spec in specs:
        variant = str(spec.parameters["variant"])
        top_k = int(spec.parameters["top_k"])
        weights = top_k_side_weights(
            signal_set.signal(cast(ReversalVariant, variant)),
            top_k=top_k,
            direction=spec.direction,
            eligible=equity_eligible,
        )
        cohorts = overlapping_cohort_targets(
            weights.to_numpy(dtype=float), holding_period=config.holding_sessions
        )
        gross, net, positions = vectorized_backtest(
            cohorts,
            asset_returns,
            execution_lag=2,
            cost_model=cost_model,
            asset_type=prepared.asset_types,
            adv_dollars=adv_values,
            full_spread_bps=spread_matrix,
        )
        active = np.abs(positions).sum(axis=1) > 0.0
        gross = np.where(active, gross, np.nan)
        net = np.where(active, net, np.nan)
        statistics = summarize_returns(
            net,
            holding_period=config.holding_sessions,
            minimum_observations=minimum_observations,
        )
        directed_test = hac_mean_test(
            net,
            holding_period=config.holding_sessions,
            alternative="greater",
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            performance = performance_metrics(
                net, positions=positions, benchmark=benchmark
            )
        sharpes.append(_unannualized_sharpe(statistics.annualized_sharpe))
        temporary.append((spec, statistics, performance, directed_test.p_value))
        net_by_id[spec.hypothesis_id] = net
        gross_by_id[spec.hypothesis_id] = gross
    trial_sharpes = np.asarray(sharpes, dtype=float)
    side_indices = {
        direction: np.asarray(
            [
                index
                for index, (spec, _, _, _) in enumerate(temporary)
                if spec.direction is direction
            ],
            dtype=np.int64,
        )
        for direction in (Direction.LONG, Direction.SHORT)
    }
    for index, (spec, statistics, performance, directed_p_value) in enumerate(
        temporary
    ):
        periodic_sharpe = trial_sharpes[index]
        side_trial_sharpes = trial_sharpes[side_indices[spec.direction]]
        side_trial_sharpes = side_trial_sharpes[np.isfinite(side_trial_sharpes)]
        psr = probabilistic_sharpe_ratio(
            periodic_sharpe,
            n_observations=statistics.n_observations,
            skewness=statistics.skewness,
            kurtosis=statistics.kurtosis,
        )
        dsr = deflated_sharpe_ratio(
            periodic_sharpe,
            n_observations=statistics.n_observations,
            n_trials=len(side_indices[spec.direction]),
            skewness=statistics.skewness,
            kurtosis=statistics.kurtosis,
            trial_sharpes=side_trial_sharpes,
        )
        rows.append(
            {
                "hypothesis_id": spec.hypothesis_id,
                "variant": spec.parameters["variant"],
                "top_k": spec.parameters["top_k"],
                "direction": spec.direction.value,
                "sample_size": statistics.n_observations,
                "net_mean": statistics.mean,
                "hac_t": statistics.hac_t_stat,
                "p_value": directed_p_value,
                "p_value_alternative": "greater",
                "hac_lags": statistics.hac_lags,
                "sharpe": statistics.annualized_sharpe,
                "probabilistic_sharpe": psr,
                "deflated_sharpe_probability": dsr,
                "dsr_trial_count": len(side_indices[spec.direction]),
                "hit_rate": statistics.hit_rate,
                "max_drawdown": performance.max_drawdown,
                "turnover": performance.turnover,
                "point_in_time_universe": point_in_time,
                "bias_tier": (
                    "POINT_IN_TIME" if point_in_time else "SURVIVORSHIP_BIASED"
                ),
                "spread_source": config.spread_source,
            }
        )
    metrics = pd.DataFrame(rows)
    p_values = pd.to_numeric(metrics["p_value"], errors="coerce").fillna(1.0)
    gauntlet = discovery_gauntlet(
        sample_sizes=metrics["sample_size"].to_numpy(dtype=np.int64),
        directed_means=metrics["net_mean"].to_numpy(dtype=float),
        t_statistics=metrics["hac_t"].to_numpy(dtype=float),
        p_values=p_values.to_numpy(dtype=float),
        dsr_probabilities=metrics["deflated_sharpe_probability"].to_numpy(dtype=float),
        minimum_observations=minimum_observations,
        t_threshold=directed_t_threshold,
        fdr_q=fdr_q,
        dsr_probability=dsr_threshold,
    )
    eligible_p_values = np.where(
        gauntlet.minimum_sample & gauntlet.directed_positive & gauntlet.t_gate,
        p_values.to_numpy(dtype=float),
        1.0,
    )
    familywise = bonferroni(eligible_p_values, alpha=fdr_q)
    metrics["minimum_sample_pass"] = gauntlet.minimum_sample
    metrics["directed_positive"] = gauntlet.directed_positive
    metrics["directed_t_pass"] = gauntlet.t_gate
    metrics["bh_adjusted_p"] = gauntlet.adjusted_p_values
    metrics["bh_pass"] = gauntlet.fdr_gate
    metrics["bonferroni_adjusted_p"] = familywise.adjusted_p_values
    metrics["bonferroni_pass"] = familywise.reject
    metrics["dsr_pass"] = gauntlet.dsr_gate
    metrics["passes_discovery"] = gauntlet.survivors
    net_frame = pd.DataFrame(net_by_id, index=prepared.dates)
    gross_frame = pd.DataFrame(gross_by_id, index=prepared.dates)
    cost_rows: dict[str, dict[str, float]] = {}
    for spec in specs:
        identifier = spec.hypothesis_id
        gross_values = gross_frame[identifier].to_numpy(dtype=float)
        net_values = net_frame[identifier].to_numpy(dtype=float)
        gross_mean = float(np.nanmean(gross_values))
        baseline_cost = float(np.nanmean(gross_values - net_values))
        cost_evidence = {
            "gross_mean": gross_mean,
            "baseline_cost_bps_per_day": baseline_cost * 10_000.0,
            "break_even_cost_multiplier": (
                gross_mean / baseline_cost
                if baseline_cost > 0.0 and gross_mean > 0.0
                else math.nan
            ),
        }
        for multiplier in cost_multipliers:
            label = f"{multiplier:g}".replace(".", "_")
            cost_evidence[f"net_mean_cost_{label}x"] = (
                gross_mean - multiplier * baseline_cost
            )
        cost_rows[identifier] = cost_evidence
    metrics = metrics.join(
        pd.DataFrame.from_dict(cost_rows, orient="index"), on="hypothesis_id"
    )
    validation_rows: dict[str, dict[str, Any]] = {}
    for spec in specs:
        stream = net_frame[spec.hypothesis_id].to_numpy(dtype=float)
        finite = np.isfinite(stream)
        if np.count_nonzero(finite) < minimum_observations:
            validation_rows[spec.hypothesis_id] = {
                "oos_t": math.nan,
                "oos_positive_fraction": math.nan,
                "oos_fold_count": 0,
                "oos_pass": False,
                "stability_score": math.nan,
                "decay": "INSUFFICIENT",
                "recent_to_prior_ratio": math.nan,
                "decay_pass": False,
            }
            continue
        selected_returns = stream[finite]
        selected_dates = prepared.dates[finite]
        walk = expanding_walk_forward(
            selected_returns,
            selected_dates,
            min_train_years=min_train_years,
            test_years=test_years,
            step_years=step_years,
            holding_period=config.holding_sessions,
            oos_t_threshold=oos_t_threshold,
            required_positive_fraction=required_positive_fraction,
        )
        decay = analyze_decay(
            selected_returns,
            selected_dates,
            window_years=rolling_years,
            step_months=12,
            holding_period=config.holding_sessions,
            minimum_observations=minimum_observations,
            stability_min=stability_min,
        )
        validation_rows[spec.hypothesis_id] = {
            "oos_t": walk.stitched_oos_test.t_stat,
            "oos_positive_fraction": walk.positive_fraction,
            "oos_fold_count": len(walk.folds),
            "oos_pass": walk.significant_oos and walk.majority_positive,
            "stability_score": decay.stability_score,
            "decay": decay.classification.value,
            "recent_to_prior_ratio": decay.recent_to_prior_ratio,
            "decay_pass": decay.classification.value == "STABLE",
        }
    validation = pd.DataFrame.from_dict(validation_rows, orient="index")
    metrics = metrics.join(validation, on="hypothesis_id")

    def side_pbo(direction: Direction) -> PBOResult:
        identifiers = [
            spec.hypothesis_id for spec in specs if spec.direction is direction
        ]
        aligned = net_frame.loc[:, identifiers].dropna(axis=0, how="any")
        if len(aligned) >= cpcv_groups:
            return cpcv_pbo(
                aligned.to_numpy(dtype=float),
                n_groups=cpcv_groups,
                n_test_groups=cpcv_test_groups,
                purge=purge,
                embargo=embargo,
            )
        return PBOResult(
            None,
            np.array([], dtype=float),
            np.array([], dtype=np.int64),
            np.array([], dtype=float),
            0,
            False,
            "insufficient common observations for side-specific CPCV/PBO",
        )

    pbo_by_side = {
        direction.value: side_pbo(direction)
        for direction in (Direction.LONG, Direction.SHORT)
    }
    defined_pbo = [
        result.pbo
        for result in pbo_by_side.values()
        if result.defined and result.pbo is not None
    ]
    if defined_pbo:
        worst_side = max(
            pbo_by_side.values(),
            key=lambda result: result.pbo if result.pbo is not None else -1.0,
        )
        pbo = worst_side
    else:
        pbo = PBOResult(
            None,
            np.array([], dtype=float),
            np.array([], dtype=np.int64),
            np.array([], dtype=float),
            0,
            False,
            "side-specific CPCV/PBO was unavailable",
        )
    metrics["side_pbo"] = metrics["direction"].map(
        lambda side: pbo_by_side[str(side)].pbo
    )
    metrics["candidate_family_pbo"] = pbo.pbo
    metrics["pbo_pass"] = metrics["side_pbo"].map(
        lambda value: value is not None and math.isfinite(float(value)) and value < 0.20
    )
    metrics["passes_rule_validation"] = (
        metrics["passes_discovery"]
        & metrics["oos_pass"]
        & metrics["decay_pass"]
        & metrics["pbo_pass"]
    )
    metrics["promotion_eligible"] = False
    metrics["promotion_blocker"] = (
        "timestamped entry/event/borrow evidence and untouched holdout remain required"
        if point_in_time
        else "current-membership survivorship bias; point-in-time rerun required"
    )
    return ReversalGridResult(
        prepared.dates,
        specs,
        metrics,
        net_frame,
        gross_frame,
        pbo,
        pbo_by_side,
        "POINT_IN_TIME" if point_in_time else "SURVIVORSHIP_BIASED",
        len(specs),
    )
