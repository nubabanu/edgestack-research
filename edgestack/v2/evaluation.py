"""Selection-aware statistical gauntlet for aligned V2 trial families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from edgestack.stats.bootstrap import (
    stationary_bootstrap_ci,
    stationary_bootstrap_indices,
)
from edgestack.stats.deflated_sharpe import deflated_sharpe_ratio
from edgestack.stats.reality_check import hansen_spa, white_reality_check
from edgestack.stats.tests import HACTestResult, hac_mean_test
from edgestack.v2.research import Horizon
from edgestack.validation.cpcv import PBOResult, cpcv_pbo
from edgestack.validation.walkforward import WalkForwardResult, expanding_walk_forward


@dataclass(frozen=True, slots=True)
class V2StrategyEvidence:
    """Horizon-aware inference for one declared strategy column."""

    hac: HACTestResult
    bootstrap_mean_ci: tuple[float, float]
    deflated_sharpe_probability: float
    walk_forward: WalkForwardResult


@dataclass(frozen=True, slots=True)
class V2FamilyEvidence:
    """Family-level multiplicity, CPCV/PBO, SPA, and Reality Check evidence."""

    strategies: tuple[V2StrategyEvidence, ...]
    pbo: PBOResult
    spa_p_value: float
    reality_check_p_value: float
    bootstrap_draws: int
    trial_count: int


def evaluate_trial_family(
    returns: NDArray[np.float64],
    dates: pd.DatetimeIndex,
    *,
    horizon: Horizon,
    total_declared_trials: int,
    bootstrap_draws: int = 2_000,
    family_test_draws: int = 10_000,
    seed: int = 20250301,
) -> V2FamilyEvidence:
    """Evaluate date-level portfolio returns with shared dependence-preserving draws."""

    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(dates):
        raise ValueError("returns must be an aligned date-by-strategy matrix")
    if matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError("a finite multi-strategy candidate family is required")
    shared = stationary_bootstrap_indices(
        matrix.shape[0],
        bootstrap_draws,
        average_block_length=max(10.0, float(horizon.sessions)),
        seed=seed,
    )
    strategy_evidence: list[V2StrategyEvidence] = []
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        standard_deviation = float(values.std(ddof=1))
        sharpe = (
            float(values.mean() / standard_deviation) if standard_deviation else 0.0
        )
        centered = values - values.mean()
        moment_scale = float(np.sqrt(np.mean(centered**2)))
        skewness = (
            float(np.mean(centered**3) / moment_scale**3) if moment_scale else 0.0
        )
        kurtosis = (
            float(np.mean(centered**4) / moment_scale**4) if moment_scale else 3.0
        )
        interval = stationary_bootstrap_ci(
            values,
            statistic="mean",
            indices=shared,
            average_block_length=max(10.0, float(horizon.sessions)),
        )
        strategy_evidence.append(
            V2StrategyEvidence(
                hac=hac_mean_test(
                    values, holding_period=horizon.sessions, alternative="greater"
                ),
                bootstrap_mean_ci=(interval.lower, interval.upper),
                deflated_sharpe_probability=deflated_sharpe_ratio(
                    sharpe,
                    n_observations=len(values),
                    n_trials=total_declared_trials,
                    skewness=skewness,
                    kurtosis=kurtosis,
                ),
                walk_forward=expanding_walk_forward(
                    values,
                    dates,
                    holding_period=horizon.sessions,
                ),
            )
        )
    pbo = cpcv_pbo(
        matrix,
        n_groups=6,
        n_test_groups=2,
        purge=horizon.sessions,
        embargo=horizon.sessions,
    )
    spa = hansen_spa(
        matrix,
        n_bootstrap=family_test_draws,
        average_block_length=max(10.0, float(horizon.sessions)),
        seed=seed,
    )
    reality = white_reality_check(
        matrix,
        n_bootstrap=family_test_draws,
        average_block_length=max(10.0, float(horizon.sessions)),
        seed=seed,
    )
    return V2FamilyEvidence(
        tuple(strategy_evidence),
        pbo,
        spa.p_value,
        reality.p_value,
        bootstrap_draws,
        total_declared_trials,
    )
