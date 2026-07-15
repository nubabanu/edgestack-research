from __future__ import annotations

from datetime import UTC
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from edgestack.backtest.costs import CostModel
from edgestack.backtest.engine import vectorized_backtest
from edgestack.config import EdgeStackConfig
from edgestack.features.cross_sectional import decile_weights
from edgestack.hypotheses.grid import cross_sectional_hypotheses
from edgestack.models import Direction, HypothesisSpec, RationaleCategory, Session
from edgestack.pipeline import research
from edgestack.pipeline.research import prepare_research, run_trial


def _overnight_alpha_bars(
    sessions: pd.DatetimeIndex,
    feature: pd.DataFrame,
) -> pd.DataFrame:
    """Create returns predictable only from the immediately preceding close."""

    weights = decile_weights(feature).to_numpy(dtype=float)
    count, assets = weights.shape
    open_ = np.full((count, assets), 100.0)
    close = np.full((count, assets), 100.0)
    for row in range(1, count):
        overnight = 0.01 * weights[row - 1]
        open_[row] = close[row - 1] * (1.0 + overnight)
        close[row] = open_[row]
    rows: list[dict[str, object]] = []
    event_times = sessions.tz_localize(UTC) + pd.Timedelta(hours=21)
    for row, session in enumerate(sessions):
        for column, symbol in enumerate(feature.columns):
            high = max(open_[row, column], close[row, column]) * 1.001
            low = min(open_[row, column], close[row, column]) * 0.999
            rows.append(
                {
                    "symbol": str(symbol),
                    "session": session,
                    "event_time": event_times[row],
                    "available_at": event_times[row] + pd.Timedelta(minutes=5),
                    "open": open_[row, column],
                    "high": high,
                    "low": low,
                    "close": close[row, column],
                    "adjusted_close": close[row, column],
                    "volume": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("return_session", "expected"),
    [
        (Session.CLOSE_TO_CLOSE, 0.03),
        (Session.OVERNIGHT, 0.02),
        (Session.INTRADAY, 103.0 / 102.0 - 1.0),
    ],
)
def test_calendar_predicate_earns_the_labeled_target_session(
    return_session: Session, expected: float
) -> None:
    sessions = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"])
    open_ = np.array([100.0, 102.0, 103.0, 103.0])
    close = np.array([100.0, 103.0, 103.0, 103.0])
    bars = pd.DataFrame(
        {
            "symbol": "SPY",
            "session": sessions,
            "event_time": sessions.tz_localize(UTC),
            "available_at": sessions.tz_localize(UTC) + pd.Timedelta(hours=22),
            "open": open_,
            "high": np.maximum(open_, close) + 0.5,
            "low": np.minimum(open_, close) - 0.5,
            "close": close,
            "adjusted_close": close,
            "volume": 1_000_000.0,
        }
    )
    prepared = prepare_research(
        bars,
        start=sessions.min(),
        end=sessions.max(),
        fomc_dates=pd.DatetimeIndex([]),
        sector_by_symbol={"SPY": "ETF"},
    )
    specification = HypothesisSpec(
        family="calendar",
        description=f"Monday {return_session.value}",
        predicates={"weekday": "MON"},
        direction=Direction.LONG,
        session=return_session,
        holding_period=1,
        rationale=RationaleCategory.NONE,
    )

    trial = run_trial(prepared, specification, cost_model=CostModel())

    monday = int(np.flatnonzero(sessions.weekday == 0)[0])
    assert trial.result.positions[monday] == 1.0
    assert trial.result.positions[monday + 1] == 0.0
    assert trial.result.gross_returns[monday] == pytest.approx(expected)
    assert trial.benchmark_returns is not None
    assert trial.result.performance.beta is not None


def test_campaign_preparation_rejects_a_bar_unavailable_at_next_fill() -> None:
    sessions = pd.bdate_range("2024-01-02", periods=3)
    bars = pd.DataFrame(
        {
            "symbol": "SPY",
            "session": sessions,
            "event_time": sessions.tz_localize(UTC) + pd.Timedelta(hours=21),
            "available_at": sessions.tz_localize(UTC) + pd.Timedelta(hours=22),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": 1_000_000.0,
        }
    )
    bars.loc[0, "available_at"] = bars.loc[1, "event_time"]

    with pytest.raises(ValueError, match="next eligible bar"):
        prepare_research(
            bars,
            start=sessions.min(),
            end=sessions.max(),
            fomc_dates=pd.DatetimeIndex([]),
            sector_by_symbol={"SPY": "ETF"},
        )


def test_reversal_positions_average_five_overlapping_cohorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = pd.bdate_range("2024-01-02", periods=8)
    symbols = [f"S{number:02d}" for number in range(10)]
    close = pd.DataFrame(100.0, index=sessions, columns=symbols)
    feature = pd.DataFrame(
        [np.roll(np.arange(10, dtype=float), day) for day in range(len(sessions))],
        index=sessions,
        columns=symbols,
    )
    prepared = SimpleNamespace(close=close, close_returns=close.pct_change())
    monkeypatch.setattr(
        research,
        "canonical_features",
        lambda _: SimpleNamespace(
            momentum=feature,
            reversal=feature,
            low_volatility=feature,
            high_proximity=feature,
        ),
    )
    specification = next(
        item
        for item in cross_sectional_hypotheses()
        if item.family == "reversal_5d" and item.direction is Direction.LONG
    )

    signal, _ = research._cross_sectional_trial_inputs(  # type: ignore[arg-type]
        prepared, specification
    )
    expected = decile_weights(feature).rolling(5, min_periods=1).mean()

    np.testing.assert_allclose(signal, expected.to_numpy(float))


def test_close_derived_rank_cannot_capture_immediate_overnight_only_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = pd.bdate_range("2024-01-02", periods=8)
    symbols = [f"S{number:02d}" for number in range(10)]
    ascending = np.arange(10, dtype=float)
    feature = pd.DataFrame(
        [
            ascending if row % 2 == 0 else ascending[::-1]
            for row in range(len(sessions))
        ],
        index=sessions,
        columns=symbols,
    )
    bars = _overnight_alpha_bars(sessions, feature)
    prepared = prepare_research(
        bars,
        start=sessions.min(),
        end=sessions.max(),
        fomc_dates=pd.DatetimeIndex([]),
        sector_by_symbol={symbol: "Test" for symbol in symbols},
    )
    monkeypatch.setattr(
        research,
        "canonical_features",
        lambda _: SimpleNamespace(
            momentum=feature,
            reversal=feature,
            low_volatility=feature,
            high_proximity=feature,
        ),
    )
    specification = HypothesisSpec(
        family="high_52w_proximity",
        description="close-derived leakage fixture",
        predicates={},
        direction=Direction.LONG,
        session=Session.CLOSE_TO_CLOSE,
        holding_period=1,
    )

    trial = run_trial(prepared, specification, cost_model=CostModel())
    naive_gross, _, _ = vectorized_backtest(
        trial.signal, trial.underlying_returns, execution_lag=1
    )
    active = np.abs(trial.result.positions).sum(axis=1) > 0.0

    assert trial.execution_lag == 2
    assert float(np.nanmean(naive_gross[1:])) > 0.019
    assert float(np.nanmean(trial.result.gross_returns[active])) < -0.019


def test_trailing_adv_is_invariant_to_fill_session_volume_mutation() -> None:
    sessions = pd.bdate_range("2024-01-02", periods=5)
    close = np.linspace(100.0, 104.0, len(sessions))
    bars = pd.DataFrame(
        {
            "symbol": "SPY",
            "session": sessions,
            "event_time": sessions.tz_localize(UTC) + pd.Timedelta(hours=21),
            "available_at": sessions.tz_localize(UTC)
            + pd.Timedelta(hours=21, minutes=5),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "adjusted_close": close,
            "volume": 1_000_000.0,
        }
    )
    mutated = bars.copy()
    mutated.loc[mutated.index[-1], "volume"] = 1_000_000_000_000.0
    prepared = [
        prepare_research(
            frame,
            start=sessions.min(),
            end=sessions.max(),
            fomc_dates=pd.DatetimeIndex([]),
            sector_by_symbol={"SPY": "ETF"},
        )
        for frame in (bars, mutated)
    ]
    specification = HypothesisSpec(
        family="calendar",
        description="known schedule",
        predicates={},
        direction=Direction.LONG,
        session=Session.INTRADAY,
        holding_period=1,
    )

    trials = [
        run_trial(item, specification, cost_model=CostModel()) for item in prepared
    ]

    np.testing.assert_allclose(trials[0].adv_dollars, trials[1].adv_dollars)


def test_portfolio_costs_use_per_instrument_spreads() -> None:
    positions = np.array([[0.0, 0.0], [1.0, 1.0]])
    mixed = CostModel().portfolio_costs(
        positions,
        asset_type=("etf", "equity"),
        adv_dollars=1_000_000_000.0,
    )
    equities = CostModel().portfolio_costs(
        positions,
        asset_type=("equity", "equity"),
        adv_dollars=1_000_000_000.0,
    )

    assert mixed[1] < equities[1]
    assert (equities[1] - mixed[1]) * 10_000 == pytest.approx(1.0)


def test_discovery_is_compact_and_tests_the_entire_real_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.bdate_range("2020-01-02", periods=120)
    first = HypothesisSpec(
        family="calendar",
        description="first real",
        predicates={"weekday": "MON"},
        direction=Direction.LONG,
        session=Session.CLOSE_TO_CLOSE,
        holding_period=1,
    )
    second = HypothesisSpec(
        family="calendar",
        description="second real",
        predicates={"weekday": "TUE"},
        direction=Direction.LONG,
        session=Session.CLOSE_TO_CLOSE,
        holding_period=1,
    )
    prepared = SimpleNamespace(dates=dates)

    def fake_trial(
        _: object,
        spec: HypothesisSpec,
        *,
        cost_model: CostModel,
    ) -> research.TrialRun:
        signal = np.resize(np.array([0.0, 1.0, 0.0, 0.0]), len(dates))
        sign = 1.0 if spec.hypothesis_id == first.hypothesis_id else -1.0
        returns = np.full(len(dates), sign * 0.002)
        benchmark = np.linspace(-0.001, 0.001, len(dates))
        return research._run_explicit(
            spec,
            signal,
            returns,
            cost_model,
            adv_dollars=100_000_000.0,
            asset_type="equity",
            benchmark_returns=benchmark,
        )

    family_shape: list[tuple[int, int]] = []

    def fake_family(matrix: np.memmap, **_: object) -> research._FamilyPValues:
        family_shape.append(matrix.shape)
        return research._FamilyPValues(0.01, 0.01)

    def fake_gauntlet(**kwargs: object) -> SimpleNamespace:
        size = len(np.asarray(kwargs["sample_sizes"]))
        survivors = np.zeros(size, dtype=bool)
        survivors[0] = True
        return SimpleNamespace(
            minimum_sample=np.ones(size, dtype=bool),
            directed_positive=np.ones(size, dtype=bool),
            t_gate=np.ones(size, dtype=bool),
            fdr_gate=np.ones(size, dtype=bool),
            dsr_gate=survivors.copy(),
            survivors=survivors,
            adjusted_p_values=np.full(size, 0.01),
        )

    monkeypatch.setattr(research, "declared_hypotheses", lambda *_: (first, second))
    monkeypatch.setattr(research, "run_trial", fake_trial)
    monkeypatch.setattr(research, "_bounded_family_p_values", fake_family)
    monkeypatch.setattr(research, "discovery_gauntlet", fake_gauntlet)
    monkeypatch.setattr(
        research,
        "_apply_shared_bootstrap_intervals",
        lambda *args, **kwargs: None,
    )
    checkpoints: list[research.DiscoveryProgress] = []

    bundle = research.run_discovery(
        prepared,  # type: ignore[arg-type]
        EdgeStackConfig(),
        batch_size=1,
        checkpoint_callback=checkpoints.append,
    )

    assert family_shape == [(len(dates), 2)]
    assert len(bundle.specs) == 6
    assert sum(spec.placebo_kind is not None for spec in bundle.specs) == 4
    assert bundle.survivor_ids == (first.hypothesis_id,)
    assert list(bundle.net_returns.columns) == [first.hypothesis_id]
    assert list(bundle.gross_returns.columns) == [first.hypothesis_id]
    assert {"bonferroni_pass", "bonferroni_adjusted_p"}.issubset(bundle.metrics.columns)
    assert bundle.metrics["benchmark_available"].all()
    assert checkpoints[-1].phase == "complete"
    assert not any(checkpoint.resumable for checkpoint in checkpoints)


def test_bounded_family_bootstrap_is_deterministic(tmp_path: Path) -> None:
    matrix = np.memmap(
        tmp_path / "family.dat",
        mode="w+",
        dtype=np.float64,
        shape=(101, 3),
        order="F",
    )
    matrix[:] = np.random.default_rng(7).normal(0.0002, 0.01, size=matrix.shape)
    matrix.flush()
    try:
        first = research._bounded_family_p_values(
            matrix,
            n_bootstrap=7_000,
            seed=42,
            workdir=tmp_path,
            strategy_batch=2,
            bootstrap_batch=16,
            callbacks=(),
            completed_trials=9,
        )
        second = research._bounded_family_p_values(
            matrix,
            n_bootstrap=7_000,
            seed=42,
            workdir=tmp_path,
            strategy_batch=3,
            bootstrap_batch=31,
            callbacks=(),
            completed_trials=9,
        )
    finally:
        research._close_memmap(matrix)

    assert first == second
    assert first.method == "BOUNDED_ARCH_EQUIVALENT"
    assert 0.0 <= first.spa <= 1.0
    assert 0.0 <= first.reality_check <= 1.0
