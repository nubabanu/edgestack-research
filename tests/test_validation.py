from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.models import DecayClass
from edgestack.validation.cpcv import (
    combinatorial_purged_splits,
    probability_backtest_overfitting,
)
from edgestack.validation.decay import DecayPoint, classify_decay, fixed_decay
from edgestack.validation.lookahead import assert_prefix_invariant, shift_and_collapse
from edgestack.validation.regimes import (
    causal_realized_vol_terciles,
    CausalTrendRegimes,
    causal_spy_ma200_regimes,
    trend_regime_interaction,
)
from edgestack.validation.replication import (
    ReplicationCheck,
    ReplicationSuiteResult,
    replicate_short_term_reversal,
)
from edgestack.validation.walkforward import expanding_window_splits


def test_walkforward_is_strictly_forward_and_expanding() -> None:
    dates = pd.bdate_range("2000-01-03", "2012-12-31")
    folds = expanding_window_splits(dates, min_train_years=5)
    assert folds
    for fold in folds:
        assert fold.train_indices.max() < fold.test_indices.min()
        assert fold.train_end < fold.test_start
    assert len(folds[-1].train_indices) > len(folds[0].train_indices)


def test_cpcv_purges_before_and_embargoes_after_test() -> None:
    splits = combinatorial_purged_splits(
        60, n_groups=6, n_test_groups=1, purge=2, embargo=3
    )
    first = splits[0]
    assert first.test_indices.tolist() == list(range(10))
    assert not set(range(10, 13)) & set(first.train_indices)
    middle = splits[2]
    assert not set(range(18, 20)) & set(middle.train_indices)
    assert not set(range(30, 33)) & set(middle.train_indices)


def test_pbo_marks_consistently_reversed_selection_as_overfit() -> None:
    train = np.array([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [2.0, 1.0, 3.0]])
    test = -train
    result = probability_backtest_overfitting(train, test)
    assert result.defined
    assert result.pbo == 1.0


def test_decay_dead_precedes_old_full_sample_strength() -> None:
    dates = pd.date_range("2005", periods=5, freq="YE")
    points = [
        DecayPoint(dates[0], dates[0], 1000, 0.0020, 4.0),
        DecayPoint(dates[1], dates[1], 1000, 0.0018, 3.5),
        DecayPoint(dates[2], dates[2], 1000, 0.0015, 3.2),
        DecayPoint(dates[3], dates[3], 1000, 0.0002, 0.5),
        DecayPoint(dates[4], dates[4], 1000, 0.0001, 0.3),
    ]
    result = classify_decay(points)
    assert result.classification is DecayClass.DEAD
    assert result.death_date == dates[3]


def test_fixed_decay_uses_complete_nonoverlapping_windows() -> None:
    dates = pd.bdate_range("2010-01-01", "2019-12-31")
    returns = np.where(dates < pd.Timestamp("2015-01-01"), 0.001, 0.002)
    points = fixed_decay(returns, dates, minimum_observations=100)
    assert len(points) == 2
    assert np.isclose(points[0].mean, 0.001)
    assert np.isclose(points[1].mean, 0.002)
    assert points[0].window_start == pd.Timestamp("2010-01-01")
    assert points[1].window_start == pd.Timestamp("2015-01-01")


def test_fixed_and_rolling_periods_both_enter_stability_fraction() -> None:
    dates = pd.date_range("2001", periods=4, freq="YE")
    rolling = [
        DecayPoint(date, date, 100, mean, 2.0)
        for date, mean in zip(dates, (1.0, 1.0, 1.0, 0.8), strict=True)
    ]
    fixed = [
        DecayPoint(dates[0], dates[0], 100, 1.0, 2.0),
        DecayPoint(dates[1], dates[1], 100, -1.0, -2.0),
    ]
    result = classify_decay(rolling, fixed_points=fixed)
    assert result.stability_score == 5 / 6
    assert result.same_sign_periods == 5
    assert result.eligible_periods == 6
    assert result.classification is DecayClass.STABLE

    unstable = classify_decay(
        rolling,
        fixed_points=[
            DecayPoint(dates[0], dates[0], 100, -1.0, -2.0),
            DecayPoint(dates[1], dates[1], 100, -1.0, -2.0),
        ],
    )
    assert unstable.stability_score == 4 / 6
    assert unstable.classification is DecayClass.INSUFFICIENT


def test_recent_effect_is_compared_with_prior_rolling_median() -> None:
    dates = pd.date_range("2001", periods=3, freq="YE")
    points = [
        DecayPoint(dates[0], dates[0], 100, 0.002, 2.0),
        DecayPoint(dates[1], dates[1], 100, 0.004, 2.0),
        DecayPoint(dates[2], dates[2], 100, 0.001, 2.0),
    ]
    result = classify_decay(points)
    assert result.recent_to_prior_ratio is not None
    assert np.isclose(result.recent_to_prior_ratio, 1 / 3)
    assert result.classification is DecayClass.DECAYING


def test_regime_dependent_requires_strict_adjusted_p_and_active_t() -> None:
    dates = pd.date_range("2001", periods=3, freq="YE")
    points = [DecayPoint(date, date, 100, 0.001, 2.5) for date in dates]
    passing = classify_decay(
        points,
        regime_interaction_adjusted_p=0.049,
        active_regime_t=2.01,
    )
    assert passing.classification is DecayClass.REGIME_DEPENDENT
    at_boundary = classify_decay(
        points,
        regime_interaction_adjusted_p=0.05,
        active_regime_t=2.0,
    )
    assert at_boundary.classification is DecayClass.STABLE


def test_ma200_regime_labels_are_lagged_one_session() -> None:
    dates = pd.bdate_range("2020-01-01", periods=205)
    close = pd.DataFrame({"SPY": np.arange(1.0, 206.0)}, index=dates)
    regimes = causal_spy_ma200_regimes(close)
    assert regimes.available
    assert regimes.labels.iloc[199] == "UNKNOWN"
    assert regimes.labels.iloc[200] == "UP"
    assert regimes.current_regime == "UP"


def test_vol_tercile_labels_are_causal_and_use_preboundary_breakpoints() -> None:
    dates = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(7)
    calm = rng.normal(0.0, 0.001, 200)
    stormy = rng.normal(0.0, 0.05, 200)
    prices = pd.Series(100.0 * np.cumprod(1.0 + np.r_[calm, stormy]), index=dates)
    labels = causal_realized_vol_terciles(
        prices, breakpoint_end=pd.Timestamp(dates[300])
    )
    assert set(labels.unique()) <= {"UNKNOWN", "VOL_LOW", "VOL_MID", "VOL_HIGH"}
    # The first window+1 sessions cannot have a trailing-vol label.
    assert (labels.iloc[:22] == "UNKNOWN").all()
    # Post-boundary stormy sessions land in the top tercile of the
    # pre-boundary distribution.
    assert (labels.iloc[330:] == "VOL_HIGH").all()


def test_vol_terciles_with_thin_reference_stay_unknown() -> None:
    dates = pd.bdate_range("2020-01-01", periods=30)
    prices = pd.Series(np.linspace(100.0, 110.0, 30), index=dates)
    labels = causal_realized_vol_terciles(prices)
    assert (labels == "UNKNOWN").all()


def test_holdout_regime_stratification_is_report_only_and_complete() -> None:
    from edgestack.pipeline.runner import _holdout_regime_stratification

    dates = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(11)
    close = pd.DataFrame(
        {"SPY": 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, 400))},
        index=dates,
    )
    holdout_dates = dates[300:]
    stream = pd.Series(rng.normal(0.0002, 0.005, len(holdout_dates)), index=holdout_dates)
    payload = _holdout_regime_stratification(
        {"edge": stream},
        stream.rename("composite"),
        close,
        holdout_start=pd.Timestamp(holdout_dates[0]),
    )
    assert payload["policy"] == "REPORT_ONLY_NO_GATE_EFFECT"
    assert set(payload["streams"]) == {"edge", "composite"}
    trend = payload["streams"]["edge"]["trend"]
    volatility = payload["streams"]["edge"]["volatility"]
    assert sum(entry["n"] for entry in trend.values()) == len(holdout_dates)
    assert sum(entry["n"] for entry in volatility.values()) == len(holdout_dates)
    for entry in list(trend.values()) + list(volatility.values()):
        assert set(entry) == {"n", "mean", "hac_t"}


def test_hac_regime_interaction_recovers_hand_calculated_means() -> None:
    dates = pd.bdate_range("2020-01-01", periods=120)
    labels = pd.Series(["UP"] * 60 + ["DOWN"] * 60, index=dates)
    returns = np.r_[np.tile([0.02, 0.01], 30), np.tile([-0.01, 0.0], 30)]
    regimes = CausalTrendRegimes(labels, "UP", True, "fixture", "available")
    result = trend_regime_interaction(returns, regimes)
    assert result.available
    assert result.active_regime == "UP"
    assert result.active_mean is not None and np.isclose(result.active_mean, 0.015)
    assert result.inactive_mean is not None and np.isclose(result.inactive_mean, -0.005)
    assert result.active_t is not None and result.active_t > 2.0
    assert result.p_value is not None and result.p_value < 0.05
    assert result.currently_active
    adjusted = result.with_adjusted_p(0.04)
    assert adjusted.adjusted_p_value == 0.04


def test_future_mutation_invariance_catches_leaky_feature() -> None:
    values = np.arange(20.0)

    def mutate(data: object, split: int) -> np.ndarray:
        changed = np.asarray(data).copy()
        changed[split:] += 10_000
        return changed

    assert_prefix_invariant(
        lambda data: pd.Series(np.asarray(data)).rolling(3).mean(),
        values,
        prefix_length=10,
        future_mutator=mutate,
    )
    try:
        assert_prefix_invariant(
            lambda data: pd.Series(np.asarray(data)).shift(-1),
            values,
            prefix_length=10,
            future_mutator=mutate,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("deliberately leaky feature escaped the detector")


def test_synthetic_one_bar_alpha_collapses_with_extra_lag() -> None:
    rng = np.random.default_rng(8)
    signal = rng.choice([-1.0, 1.0], size=2_000)
    returns = np.r_[0.0, signal[:-1] * 0.005] + rng.normal(0.0, 0.01, size=2_000)
    result = shift_and_collapse(signal, returns)
    assert result.collapsed


def test_reversal_replication_requires_cost_damage() -> None:
    result = replicate_short_term_reversal(np.full(200, 0.001), np.full(200, 0.0004))
    assert result.passed


def test_replication_diagnostic_requires_six_numerically_executed_checks() -> None:
    checks = tuple(
        ReplicationCheck(str(index), False, float(index), "frozen", {}, "source")
        for index in range(6)
    )
    assert ReplicationSuiteResult(checks).executed

    invalid = (
        *checks[:-1],
        ReplicationCheck("invalid", False, np.nan, "frozen", {}, "source"),
    )
    assert not ReplicationSuiteResult(invalid).executed
