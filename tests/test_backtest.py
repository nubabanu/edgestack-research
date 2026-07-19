from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from edgestack.backtest.confirm import (
    ConfirmationData,
    ConfirmationEngine,
    ZiplineBackendStatus,
    zipline_backend_status,
)
from edgestack.backtest.costs import CostModel, MarketContext, TradeIntent
from edgestack.backtest.engine import (
    BacktestResult,
    close_derived_execution_lag,
    next_eligible_execution,
    vectorized_backtest,
)
from edgestack.backtest.metrics import performance_metrics
from edgestack.hypotheses.grid import GridConfig, enumerate_hypotheses
from edgestack.models import (
    DecayClass,
    Direction,
    EvidenceBundle,
    ExecutionStatus,
    HypothesisSpec,
    Session,
    Verdict,
    VerdictRecord,
)
from edgestack.pipeline import validation_run
from edgestack.pipeline.research import prepare_research, run_trial
from edgestack.stats.tests import summarize_returns
from edgestack.validation.cpcv import PBOResult


def test_vectorized_engine_never_uses_signal_bar() -> None:
    signal = np.array([1.0, 0.0, -1.0, 0.0])
    returns = np.array([0.50, 0.10, 0.20, -0.10])
    gross, _, positions = vectorized_backtest(signal, returns, cost_model=CostModel())
    np.testing.assert_allclose(positions, [0.0, 1.0, 0.0, -1.0])
    np.testing.assert_allclose(gross, [0.0, 0.10, 0.0, 0.10])


def test_close_derived_conventions_resolve_exact_fill_and_return_timestamps() -> None:
    opens = tuple(datetime(2024, 1, day, 14, 30, tzinfo=UTC) for day in (2, 3, 4))
    closes = tuple(datetime(2024, 1, day, 21, 0, tzinfo=UTC) for day in (2, 3, 4))
    available = datetime(2024, 1, 2, 21, 5, tzinfo=UTC)

    intraday = next_eligible_execution(
        available, opens, closes, session=Session.INTRADAY
    )
    overnight = next_eligible_execution(
        available, opens, closes, session=Session.OVERNIGHT
    )
    close_to_close = next_eligible_execution(
        available, opens, closes, session=Session.CLOSE_TO_CLOSE
    )

    assert intraday.fill_index == intraday.return_index == 1
    assert intraday.fill_time == opens[1]
    assert overnight.fill_index == close_to_close.fill_index == 1
    assert overnight.return_index == close_to_close.return_index == 2
    assert overnight.fill_time == close_to_close.fill_time == closes[1]
    assert all(
        item.fill_time > available for item in (intraday, overnight, close_to_close)
    )


@pytest.mark.parametrize(
    ("session", "expected_lag", "expected_positions", "expected_gross"),
    [
        (Session.INTRADAY, 1, [0.0, 1.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]),
        (Session.OVERNIGHT, 2, [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.2, 0.0]),
        (
            Session.CLOSE_TO_CLOSE,
            2,
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.2, 0.0],
        ),
    ],
)
def test_close_derived_positions_earn_only_the_first_eligible_interval(
    session: Session,
    expected_lag: int,
    expected_positions: list[float],
    expected_gross: list[float],
) -> None:
    signal = np.array([1.0, 0.0, 0.0, 0.0])
    interval_returns = np.array([0.0, 0.5, 0.2, -0.1])
    lag = close_derived_execution_lag(session)

    gross, _, positions = vectorized_backtest(
        signal, interval_returns, execution_lag=lag
    )

    assert lag == expected_lag
    np.testing.assert_allclose(positions, expected_positions)
    np.testing.assert_allclose(gross, expected_gross)


def test_cost_breakdown_matches_frozen_formula() -> None:
    result = CostModel().estimate(
        TradeIntent(order_dollars=10_000.0, fills=2),
        MarketContext(adv_dollars=1_000_000.0, asset_type="etf"),
    )
    # 1 bp round-trip spread + 2 bp/fill slippage + 1 bp turnover penalty.
    np.testing.assert_allclose(result.total_bps, 6.0)


def test_capacity_curve_raises_impact_without_changing_gross_alpha() -> None:
    gross = np.array([0.0, 0.01, 0.01])
    positions = np.array([0.0, 1.0, 0.0])
    curve = CostModel().capacity_curve(
        gross,
        positions,
        capital_multipliers=(1.0, 100.0),
        adv_dollars=1_000_000.0,
    )

    assert np.nanmean(curve[1.0]) > np.nanmean(curve[100.0])


def test_metrics_capture_drawdown_and_benchmark_relative_fields() -> None:
    values = np.array([0.10, -0.20, 0.05, 0.03])
    benchmark = np.array([0.02, -0.01, 0.01, 0.01])
    result = performance_metrics(values, benchmark=benchmark)
    np.testing.assert_allclose(result.max_drawdown, -0.2)
    assert result.beta is not None
    assert result.information_ratio is not None


def test_independent_confirmation_agrees_with_vector_engine() -> None:
    signal = np.array([0.0, 1.0, 1.0, 0.0, -1.0, 0.0])
    returns = np.array([0.0, 0.01, 0.02, -0.01, 0.03, -0.02])
    gross, net, positions = vectorized_backtest(signal, returns)
    spec = enumerate_hypotheses(GridConfig(predicate_levels={"weekday": ("MON",)}))[0]
    vector = BacktestResult(
        spec.hypothesis_id,
        gross,
        net,
        positions,
        summarize_returns(net),
        performance_metrics(net, positions=positions),
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = tuple(start + timedelta(days=value) for value in range(len(signal)))
    result = ConfirmationEngine().confirm(
        spec, ConfirmationData(signal, returns, timestamps), vector_result=vector
    )
    assert result.passed
    assert result.difference_bps_per_trade is not None
    assert result.difference_bps_per_trade < 1e-10


def test_confirmation_reuses_frozen_liquidity_and_asset_type() -> None:
    signal = np.array([0.0, 1.0, 1.0, 0.0, -1.0, 0.0])
    returns = np.array([0.0, 0.01, 0.02, -0.01, 0.03, -0.02])
    adv = np.array([1e6, 2e6, 3e6, 4e6, 5e6, 6e6])
    gross, net, positions = vectorized_backtest(
        signal,
        returns,
        asset_type="etf",
        adv_dollars=adv,
    )
    spec = enumerate_hypotheses(GridConfig(predicate_levels={"weekday": ("MON",)}))[0]
    vector = BacktestResult(
        spec.hypothesis_id,
        gross,
        net,
        positions,
        summarize_returns(net),
        performance_metrics(net, positions=positions),
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = tuple(start + timedelta(days=value) for value in range(len(signal)))

    result = ConfirmationEngine().confirm(
        spec,
        ConfirmationData(
            signal,
            returns,
            timestamps,
            adv_dollars=adv,
            asset_type="etf",
        ),
        vector_result=vector,
    )

    assert result.passed
    assert result.difference_bps_per_trade is not None
    assert result.difference_bps_per_trade < 1e-10


def test_zipline_import_alone_never_passes_finalist_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed package must not masquerade as an executed backend."""

    signal = np.array([0.0, 1.0, 1.0, 0.0, -1.0, 0.0])
    returns = np.array([0.0, 0.01, 0.02, -0.01, 0.03, -0.02])
    gross, net, positions = vectorized_backtest(signal, returns)
    spec = enumerate_hypotheses(GridConfig(predicate_levels={"weekday": ("MON",)}))[0]
    vector = BacktestResult(
        spec.hypothesis_id,
        gross,
        net,
        positions,
        summarize_returns(net),
        performance_metrics(net, positions=positions),
    )
    dates = pd.date_range("2024-01-02", periods=len(signal), freq="B")
    prepared = SimpleNamespace(dates=dates)
    trial = SimpleNamespace(
        signal=signal,
        underlying_returns=returns,
        result=vector,
        spec=spec,
    )
    monkeypatch.setattr(
        validation_run,
        "zipline_backend_status",
        lambda: ZiplineBackendStatus(
            installed=True,
            version="3.1.1",
            executable=False,
            reason="test adapter is deliberately absent",
        ),
    )

    outcome = validation_run._confirm_trial(
        prepared, trial, CostModel()  # type: ignore[arg-type]
    )

    assert not outcome.passed
    assert not outcome.executed
    assert outcome.difference_bps < 1e-10
    assert "zipline_reloaded_3.1.1_not_executed" in outcome.backend
    assert "import_verified" not in outcome.backend


def test_cross_sectional_loop_also_cannot_claim_zipline_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal = np.array([[0.0, 0.0], [0.5, -0.5], [0.5, -0.5]])
    returns = np.array([[0.0, 0.0], [0.01, -0.01], [0.02, -0.02]])
    gross, net, positions = vectorized_backtest(signal, returns)
    spec = enumerate_hypotheses(GridConfig(predicate_levels={"weekday": ("MON",)}))[0]
    vector = BacktestResult(
        spec.hypothesis_id,
        gross,
        net,
        positions,
        summarize_returns(net),
        performance_metrics(net, positions=positions),
    )
    trial: Any = SimpleNamespace(
        signal=signal,
        underlying_returns=returns,
        result=vector,
        spec=spec,
    )
    prepared: Any = SimpleNamespace(
        dates=pd.date_range("2024-01-02", periods=len(signal), freq="B")
    )
    monkeypatch.setattr(
        validation_run,
        "zipline_backend_status",
        lambda: ZiplineBackendStatus(True, "3.1.1", False, "adapter absent"),
    )

    outcome = validation_run._confirm_trial(prepared, trial, CostModel())

    assert not outcome.passed
    assert not outcome.executed
    assert "independent_cross_sectional_loop" in outcome.backend
    assert "not_executed" in outcome.backend


@pytest.mark.skipif(
    not zipline_backend_status().executable,
    reason="zipline-reloaded 3.1.1 confirmation extra is unavailable",
)
def test_actual_zipline_adapter_executes_canonical_bars_and_matches() -> None:
    sessions = pd.bdate_range("2024-01-02", periods=8)
    prices = 100.0 * np.cumprod(np.r_[1.0, np.full(len(sessions) - 1, 0.002)])
    event_time = sessions.tz_localize(UTC) + pd.Timedelta(hours=21)
    bars = pd.DataFrame(
        {
            "symbol": "SPY",
            "session": sessions,
            "event_time": event_time,
            "available_at": event_time + pd.Timedelta(minutes=5),
            "open": prices * 0.999,
            "high": prices * 1.001,
            "low": prices * 0.998,
            "close": prices,
            "adjusted_close": prices,
            "volume": 5_000_000.0,
        }
    )
    prepared = prepare_research(
        bars,
        start=sessions[0],
        end=sessions[-1],
        fomc_dates=pd.DatetimeIndex([]),
        sector_by_symbol={"SPY": "ETF"},
    )
    spec = HypothesisSpec(
        family="calendar",
        description="known-in-advance all-session baseline",
        predicates={},
        direction=Direction.LONG,
        session=Session.CLOSE_TO_CLOSE,
        holding_period=1,
    )
    trial = run_trial(prepared, spec, cost_model=CostModel())

    outcome = validation_run._confirm_trial(prepared, trial, CostModel())

    assert outcome.executed
    assert outcome.passed, outcome.reason
    assert outcome.timestamps_match
    assert outcome.trade_count == outcome.vector_trade_count == 1
    assert outcome.difference_bps < 1e-6
    assert outcome.backend.startswith("zipline-reloaded-3.1.1")


def test_holdout_replay_cannot_resurrect_unexecuted_confirmation() -> None:
    evidence = EvidenceBundle(
        hypothesis_id="edge-1",
        sample_size=500,
        gross_mean=0.002,
        net_mean=0.001,
        hac_t=4.0,
        p_value=0.001,
        sharpe=1.2,
        probabilistic_sharpe=0.99,
        deflated_sharpe_probability=0.99,
        hit_rate=0.55,
        max_drawdown=-0.1,
        turnover=0.5,
        exposure=0.5,
        skew=0.0,
        kurtosis=3.0,
        mean_ci=(0.0005, 0.0015),
        sharpe_ci=(0.5, 1.8),
        oos_t=3.0,
        oos_positive_fraction=0.75,
        stability_score=0.8,
        pbo=0.1,
        confirmation_difference_bps=0.0,
        annotations={
            "confirmation_executed": False,
            "confirmation_pass": False,
        },
    )
    stale_provisional = VerdictRecord(
        hypothesis_id="edge-1",
        execution_status=ExecutionStatus.TESTED,
        verdict=Verdict.WORKS,
        decay=DecayClass.STABLE,
        reasons=("legacy import-only confirmation",),
        evidence=evidence,
        provisional=True,
    )

    (final,) = validation_run.final_records(
        (stale_provisional,),
        {"edge-1": 0.0005},
        evaluated_ids={"edge-1"},
    )

    assert final.verdict is Verdict.WEAK
    assert any("confirmation" in reason for reason in final.reasons)


def test_empty_signal_is_invalid_not_an_empirical_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation_run, "zipline_available", lambda: False)
    metrics = pd.DataFrame(
        [
            {
                "hypothesis_id": "empty-edge",
                "sample_size": 0,
                "empty_signal": True,
                "placebo_kind": None,
                "discovery_survivor": False,
                "gross_mean": 0.0,
                "net_mean": 0.0,
                "hac_t": 0.0,
                "p_value": 1.0,
                "sharpe": 0.0,
                "probabilistic_sharpe": 0.0,
                "deflated_sharpe_probability": 0.0,
                "hit_rate": 0.0,
                "max_drawdown": 0.0,
                "turnover": 0.0,
                "exposure": 0.0,
                "skew": 0.0,
                "kurtosis": 3.0,
                "mean_ci_lower": 0.0,
                "mean_ci_upper": 0.0,
                "sharpe_ci_lower": 0.0,
                "sharpe_ci_upper": 0.0,
                "decay": DecayClass.INSUFFICIENT.value,
            }
        ]
    )
    undefined_pbo = PBOResult(
        None,
        np.array([], dtype=float),
        np.array([], dtype=int),
        np.array([], dtype=float),
        0,
        False,
        "not enough candidates",
    )

    records = validation_run._records(metrics, undefined_pbo)

    assert records[0].execution_status is ExecutionStatus.INVALID
    assert records[0].verdict is None


def test_corwin_schultz_recovers_the_spread_of_a_flat_tape() -> None:
    from edgestack.data.spreads import corwin_schultz_spread

    sessions = pd.bdate_range("2024-01-02", periods=40)
    # Zero-volatility tape: the entire daily range IS the spread.
    high = pd.Series(100.1, index=sessions)
    low = pd.Series(99.9, index=sessions)
    spread = corwin_schultz_spread(high, low)
    expected = float(np.log(100.1 / 99.9))
    assert spread.iloc[1:].notna().all()
    np.testing.assert_allclose(spread.iloc[1:], expected, rtol=1e-4)


def test_monthly_median_spreads_reject_thin_months() -> None:
    from edgestack.data.spreads import monthly_median_spread_bps

    sessions = pd.bdate_range("2024-01-02", periods=30)
    high = pd.DataFrame({"AAA": 100.1}, index=sessions)
    low = pd.DataFrame({"AAA": 99.9}, index=sessions)
    close = pd.DataFrame({"AAA": 100.0}, index=sessions)
    monthly = monthly_median_spread_bps(high, low, close, minimum_observations=12)
    assert monthly.loc["2024-01", "AAA"] == pytest.approx(
        float(np.log(100.1 / 99.9)) * 10_000.0, rel=1e-4
    )
    # February holds too few sessions in this fixture to qualify.
    assert np.isnan(monthly.loc["2024-02", "AAA"])


def test_floored_spread_matrix_lags_a_month_and_never_undercuts_baseline() -> None:
    from edgestack.data.spreads import floored_spread_matrix

    monthly = pd.DataFrame(
        {"AAA": [12.0, 1.0]},
        index=pd.PeriodIndex(["2024-01", "2024-02"], freq="M"),
    )
    sessions = pd.DatetimeIndex(["2024-01-15", "2024-02-15", "2024-03-15"])
    matrix = floored_spread_matrix(monthly, sessions, baseline_bps=3.0)
    # January has no prior-month estimate; February uses January's 12 bps;
    # March's measured 1 bps is floored at the 3 bps baseline.
    assert matrix.loc[sessions[0], "AAA"] == 3.0
    assert matrix.loc[sessions[1], "AAA"] == 12.0
    assert matrix.loc[sessions[2], "AAA"] == 3.0


def test_portfolio_costs_measured_spread_override_is_floored_and_applied() -> None:
    positions = np.array([[0.0, 0.0], [1.0, 1.0]])
    baseline = CostModel().portfolio_costs(
        positions, asset_type=("equity", "equity"), adv_dollars=1_000_000_000.0
    )
    measured = CostModel().portfolio_costs(
        positions,
        asset_type=("equity", "equity"),
        adv_dollars=1_000_000_000.0,
        full_spread_bps=np.full_like(positions, 9.0),
    )
    # 9 bps vs the assumed 3 bps: two names each pay 3 extra half-spread bps.
    assert (measured[1] - baseline[1]) * 10_000 == pytest.approx(6.0)
    with pytest.raises(ValueError, match="floored at the"):
        CostModel().portfolio_costs(
            positions,
            asset_type=("equity", "equity"),
            adv_dollars=1_000_000_000.0,
            full_spread_bps=np.full_like(positions, 1.0),
        )
    with pytest.raises(ValueError, match="align with the positions"):
        CostModel().portfolio_costs(
            positions,
            asset_type=("equity", "equity"),
            adv_dollars=1_000_000_000.0,
            full_spread_bps=np.array([9.0, 9.0]),
        )
