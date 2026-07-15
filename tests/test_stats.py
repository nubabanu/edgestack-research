from __future__ import annotations

import numpy as np

from edgestack.stats.bootstrap import (
    stationary_bootstrap_ci,
    stationary_bootstrap_indices,
)
from edgestack.stats.deflated_sharpe import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from edgestack.stats.multiple_testing import benjamini_hochberg, bonferroni
from edgestack.stats.reality_check import hansen_spa, white_reality_check
from edgestack.stats.tests import hac_mean_test


def test_hac_iid_lag_zero_matches_mean_standard_error() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.00, 0.02, -0.01])
    result = hac_mean_test(returns, lags=0)
    # HAC uses the population autocovariance; this is the sandwich convention.
    expected_se = np.sqrt(np.mean((returns - returns.mean()) ** 2) / len(returns))
    np.testing.assert_allclose(result.standard_error, expected_se)
    np.testing.assert_allclose(result.t_stat, returns.mean() / expected_se)


def test_bh_and_bonferroni_known_example() -> None:
    p_values = np.array([0.001, 0.01, 0.03, 0.20])
    bh = benjamini_hochberg(p_values, q=0.05)
    assert bh.reject.tolist() == [True, True, True, False]
    np.testing.assert_allclose(bh.adjusted_p_values, [0.004, 0.02, 0.04, 0.20])
    family = bonferroni(p_values, alpha=0.05)
    assert family.reject.tolist() == [True, True, False, False]


def test_stationary_bootstrap_is_deterministic_and_brackets_mean() -> None:
    values = np.linspace(-0.01, 0.02, 100)
    first = stationary_bootstrap_indices(100, 50, seed=9)
    second = stationary_bootstrap_indices(100, 50, seed=9)
    np.testing.assert_array_equal(first, second)
    ci = stationary_bootstrap_ci(values, n_resamples=500, seed=4)
    assert ci.lower < values.mean() < ci.upper


def test_psr_and_dsr_penalize_trials() -> None:
    psr = probabilistic_sharpe_ratio(0.20, n_observations=252)
    dsr_one = deflated_sharpe_ratio(0.20, n_observations=252, n_trials=1)
    dsr_many = deflated_sharpe_ratio(0.20, n_observations=252, n_trials=1_000)
    assert psr == dsr_one
    assert dsr_many < dsr_one


def test_reality_checks_detect_strong_strategy() -> None:
    rng = np.random.default_rng(12)
    returns = rng.normal(0.0, 0.01, size=(500, 3))
    returns[:, 0] += 0.003
    white = white_reality_check(returns, n_bootstrap=300, seed=2)
    spa = hansen_spa(returns, n_bootstrap=300, seed=2)
    assert white.best_strategy == 0
    assert white.p_value < 0.05
    assert spa.p_value < 0.05
