"""White Reality Check and Hansen SPA stationary-bootstrap tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from arch.bootstrap import SPA, RealityCheck

from edgestack.stats._types import FloatArray


def _differentials(values: FloatArray, benchmark: FloatArray | None) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("strategy returns must be (dates, strategies)")
    if benchmark is not None:
        reference = np.asarray(benchmark, dtype=float)
        if reference.ndim != 1 or len(reference) != matrix.shape[0]:
            raise ValueError("benchmark must have one value per date")
        matrix = matrix - reference[:, None]
    if np.any(~np.isfinite(matrix)):
        raise ValueError("reality-check inputs must be finite and aligned")
    return matrix


@dataclass(frozen=True, slots=True)
class RealityCheckResult:
    """Family-level data-snooping test result."""

    statistic: float
    p_value: float
    best_strategy: int
    mean_differentials: FloatArray
    n_bootstrap: int
    method: str


def white_reality_check(
    strategy_returns: FloatArray,
    benchmark_returns: FloatArray | None = None,
    *,
    n_bootstrap: int = 10_000,
    average_block_length: float = 10.0,
    seed: int = 0,
) -> RealityCheckResult:
    """Test whether the best rule beats the benchmark after specification search."""

    differential = _differentials(strategy_returns, benchmark_returns)
    n = differential.shape[0]
    means = differential.mean(axis=0)
    observed = math.sqrt(n) * max(float(means.max()), 0.0)
    # arch uses a loss convention (smaller is better), so positive strategy
    # differential is represented as negative model loss against zero loss.
    reference = RealityCheck(
        np.zeros(n, dtype=float),
        -differential,
        block_size=max(1, round(average_block_length)),
        reps=n_bootstrap,
        bootstrap="stationary",
        studentize=False,
        seed=seed,
    )
    reference.compute()
    p_value = float(reference.pvalues["consistent"])
    return RealityCheckResult(
        observed,
        p_value,
        int(np.argmax(means)),
        means,
        n_bootstrap,
        "ARCH_WHITE_REALITY_CHECK",
    )


def hansen_spa(
    strategy_returns: FloatArray,
    benchmark_returns: FloatArray | None = None,
    *,
    n_bootstrap: int = 10_000,
    average_block_length: float = 10.0,
    seed: int = 0,
) -> RealityCheckResult:
    """Run Hansen's studentized Superior Predictive Ability test.

    Poor alternatives are consistently recentered out of the null, avoiding the
    power loss of White's least-favorable centering while retaining a bootstrap
    family-level p-value.
    """

    differential = _differentials(strategy_returns, benchmark_returns)
    n, n_models = differential.shape
    means = differential.mean(axis=0)
    standard_errors = differential.std(axis=0, ddof=1) / math.sqrt(n)
    valid = standard_errors > 0.0
    observed_t = np.full(n_models, -math.inf, dtype=float)
    observed_t[valid] = means[valid] / standard_errors[valid]
    observed = max(float(observed_t.max()), 0.0)
    reference = SPA(
        np.zeros(n, dtype=float),
        -differential,
        block_size=max(1, round(average_block_length)),
        reps=n_bootstrap,
        bootstrap="stationary",
        studentize=True,
        seed=seed,
    )
    reference.compute()
    p_value = float(reference.pvalues["consistent"])
    return RealityCheckResult(
        observed,
        p_value,
        int(np.argmax(means)),
        means,
        n_bootstrap,
        "ARCH_HANSEN_SPA",
    )


spa_test = hansen_spa
reality_check = white_reality_check
