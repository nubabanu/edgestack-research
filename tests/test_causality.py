"""Causality and look-ahead regression tests.

These tests use prefix replay rather than merely inspecting source code.  A
feature is causal only when the value it reports for a decision is unchanged
after observations arriving later are appended or mutated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from edgestack.features.cross_sectional import (
    momentum_12_1,
    proximity_to_high,
    realized_volatility,
    short_term_reversal,
)
from edgestack.features.sessions import decompose_sessions
from edgestack.models import CausalDataView, Feature, ensure_fill_after_signal

FIXTURES = Path(__file__).parent / "fixtures"


def _recorded_bars() -> pd.DataFrame:
    frame = pd.read_csv(
        FIXTURES / "recorded_causal_bars.csv",
        parse_dates=["open_time", "event_time", "available_at"],
    )
    return frame.sort_values("event_time", kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class _TrailingReturnFeature:
    """Adapter exercising a real trailing feature through the public protocol."""

    required_fields = frozenset({"close"})
    lookback_sessions = 2

    def compute(self, view: CausalDataView) -> pd.Series:
        prices = view.frame.set_index("event_time")[["close"]]
        return short_term_reversal(
            prices, lookback=self.lookback_sessions, contrarian=False
        )["close"]


@dataclass(frozen=True)
class _DeliberatelyLeakyFeature:
    """Negative control: next close is impermissibly read into the current row."""

    required_fields = frozenset({"close"})
    lookback_sessions = 0

    def compute(self, view: CausalDataView) -> pd.Series:
        close = view.frame.set_index("event_time")["close"]
        return close.shift(-1) / close - 1.0


def _assert_prefix_invariant(feature: Feature, frame: pd.DataFrame) -> None:
    """Compare retrospective output with output available at every prefix end."""

    final_decision = frame["available_at"].iloc[-1].to_pydatetime()
    retrospective = feature.compute(CausalDataView(frame, final_decision))
    for stop in range(1, len(frame) + 1):
        prefix = frame.iloc[:stop].copy()
        decision = prefix["available_at"].iloc[-1].to_pydatetime()
        real_time = feature.compute(CausalDataView(prefix, decision)).iloc[-1]
        historical = retrospective.iloc[stop - 1]
        if pd.isna(real_time) and pd.isna(historical):
            continue
        if not np.isclose(real_time, historical, equal_nan=True):
            raise AssertionError(
                f"prefix invariance failed at observation {stop - 1}: "
                f"real_time={real_time!r}, retrospective={historical!r}"
            )


def test_causal_view_filters_by_availability_not_event_date() -> None:
    bars = _recorded_bars()
    decision = datetime(2024, 1, 5, 22, tzinfo=UTC)

    view = CausalDataView.as_of(bars, decision)

    # The 5 January bar has occurred, but its deliberately delayed payload has
    # not arrived.  Event-time filtering would incorrectly expose it.
    assert view.frame["session"].tolist() == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]
    assert bool((view.frame["available_at"] <= pd.Timestamp(decision)).all())

    delayed_arrival = datetime(2024, 1, 8, 12, tzinfo=UTC)
    at_arrival = CausalDataView.as_of(bars, delayed_arrival)
    assert at_arrival.frame["session"].iloc[-1] == "2024-01-05"


def test_causal_view_rejects_even_one_future_observation() -> None:
    bars = _recorded_bars()
    decision = bars["available_at"].iloc[-2].to_pydatetime()

    with pytest.raises(ValueError, match="future data present"):
        CausalDataView(bars, decision)


def test_feature_prefix_replay_accepts_trailing_feature_and_catches_leak() -> None:
    bars = (
        _recorded_bars()
        .sort_values("available_at", kind="stable")
        .reset_index(drop=True)
    )
    causal: Feature = _TrailingReturnFeature()
    leaky: Feature = _DeliberatelyLeakyFeature()

    assert isinstance(causal, Feature)
    assert isinstance(leaky, Feature)
    _assert_prefix_invariant(causal, bars)
    with pytest.raises(AssertionError, match="prefix invariance failed"):
        _assert_prefix_invariant(leaky, bars)


def test_cross_sectional_features_are_invariant_to_future_price_mutation() -> None:
    sessions = pd.bdate_range("2023-01-02", periods=48)
    changes = np.array(
        [
            np.sin(np.arange(48) / 4.0) * 0.006 + 0.0005,
            np.cos(np.arange(48) / 5.0) * 0.005 + 0.0002,
            np.sin(np.arange(48) / 7.0 + 1.0) * 0.004 - 0.0001,
        ]
    ).T
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(changes, axis=0)),
        index=sessions,
        columns=["ALPHA", "BETA", "GAMMA"],
    )
    cutoff = 30
    mutated = prices.copy()
    mutated.iloc[cutoff + 1 :] *= np.array([7.0, 0.2, 11.0])

    computations = (
        lambda values: momentum_12_1(values, lookback=10, skip=2),
        lambda values: short_term_reversal(values, lookback=4),
        lambda values: realized_volatility(values, window=8),
        lambda values: proximity_to_high(values, window=8),
    )
    for compute in computations:
        original = compute(prices).iloc[: cutoff + 1]
        changed = compute(mutated).iloc[: cutoff + 1]
        pdt.assert_frame_equal(original, changed, check_exact=True)


def test_extended_features_are_invariant_to_future_mutation() -> None:
    from edgestack.features.cross_sectional import (
        amihud_illiquidity,
        max_lottery,
        overnight_intraday_gap,
    )

    sessions = pd.bdate_range("2023-01-02", periods=60)
    rng = np.random.default_rng(13)
    closes = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (60, 3)), axis=0)),
        index=sessions,
        columns=["ALPHA", "BETA", "GAMMA"],
    )
    opens = closes.shift(1).fillna(closes.iloc[0]) * (
        1.0 + rng.normal(0.0, 0.002, (60, 3))
    )
    volume = pd.DataFrame(
        rng.integers(100_000, 1_000_000, (60, 3)).astype(float),
        index=sessions,
        columns=closes.columns,
    )
    cutoff = 40
    mutated_close = closes.copy()
    mutated_close.iloc[cutoff + 1 :] *= 5.0
    mutated_open = opens.copy()
    mutated_open.iloc[cutoff + 1 :] *= 0.3
    mutated_volume = volume.copy()
    mutated_volume.iloc[cutoff + 1 :] *= 100.0

    computations = (
        lambda c, o, v: amihud_illiquidity(c, v, window=10),
        lambda c, o, v: max_lottery(c, window=10),
        lambda c, o, v: overnight_intraday_gap(o, c, window=10),
    )
    for compute in computations:
        original = compute(closes, opens, volume).iloc[: cutoff + 1]
        changed = compute(mutated_close, mutated_open, mutated_volume).iloc[
            : cutoff + 1
        ]
        pdt.assert_frame_equal(original, changed, check_exact=True)


def test_etf_relative_reversal_scores_only_etf_columns() -> None:
    from edgestack.models import Direction, RationaleCategory, Session
    from edgestack.models import HypothesisSpec as Spec
    from edgestack.pipeline.research import _cross_sectional_feature

    prepared = _prepared_spy_panel()
    spec = Spec(
        family="etf_relative_reversal",
        description="LONG cross-sectional etf_relative_reversal",
        predicates={},
        session=Session.CLOSE_TO_CLOSE,
        holding_period=5,
        direction=Direction.LONG,
        rationale=RationaleCategory.MICROSTRUCTURE,
        parameters={"lookback": 21},
    )
    feature = _cross_sectional_feature(prepared, spec)
    assert list(feature.columns) == list(prepared.close.columns)
    # The only fixture instrument is the SPY ETF, so its column carries the
    # score and nothing else exists to leak in.
    assert feature["SPY"].iloc[30:].notna().all()


def test_calendar_gated_trial_holds_positions_only_in_gate_cohorts() -> None:
    from edgestack.backtest.costs import CostModel
    from edgestack.data.calendars import NYSECalendar
    from edgestack.models import Direction, RationaleCategory, Session
    from edgestack.models import HypothesisSpec as Spec
    from edgestack.pipeline.research import prepare_research, run_trial

    sessions = NYSECalendar().sessions("2020-01-02", "2020-12-31")[:200]
    rng = np.random.default_rng(21)
    frames = []
    sectors = {}
    for symbol in ("SPY", "AAA", "BBB", "CCC", "DDD"):
        close = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, len(sessions)))
        frames.append(_spy_bars(sessions, close).assign(symbol=symbol))
        sectors[symbol] = "ETF" if symbol == "SPY" else "Tech"
    prepared = prepare_research(
        pd.concat(frames, ignore_index=True),
        start=sessions.min(),
        end=sessions.max(),
        fomc_dates=pd.DatetimeIndex([]),
        sector_by_symbol=sectors,
    )
    gated = Spec(
        family="reversal_5d",
        description="LONG cross-sectional reversal_5d gated when weekday=FRI",
        predicates={"weekday": "FRI"},
        session=Session.CLOSE_TO_CLOSE,
        holding_period=1,
        direction=Direction.LONG,
        rationale=RationaleCategory.MICROSTRUCTURE,
        parameters={"lookback": 5, "combination": "CALENDAR_GATED"},
    )
    trial = run_trial(prepared, gated, cost_model=CostModel())
    active = np.abs(trial.result.positions).sum(axis=1) > 0.0
    active_days = set(pd.DatetimeIndex(prepared.dates[active]).dayofweek)
    # A one-session hold entered for Friday-earning cohorts is live only on
    # Fridays; the gate may never leak exposure into other sessions.
    assert active.any()
    assert active_days == {4}

    ungated = Spec(
        family="reversal_5d",
        description="LONG cross-sectional reversal_5d",
        predicates={},
        session=Session.CLOSE_TO_CLOSE,
        holding_period=1,
        direction=Direction.LONG,
        rationale=RationaleCategory.MICROSTRUCTURE,
        parameters={"lookback": 5},
    )
    plain = run_trial(prepared, ungated, cost_model=CostModel())
    plain_active = np.abs(plain.result.positions).sum(axis=1) > 0.0
    assert plain_active.sum() > active.sum()


def test_session_decomposition_has_exact_information_boundaries() -> None:
    index = pd.bdate_range("2024-03-01", periods=12)
    open_price = pd.Series(
        [
            100.0,
            101.0,
            100.5,
            102.0,
            101.5,
            103.0,
            102.0,
            104.0,
            103.5,
            105.0,
            104.5,
            106.0,
        ],
        index=index,
        name="open",
    )
    close = pd.Series(
        [
            100.8,
            100.7,
            101.7,
            101.8,
            102.8,
            102.6,
            103.6,
            103.8,
            104.8,
            104.6,
            105.8,
            105.7,
        ],
        index=index,
        name="close",
    )
    returns = decompose_sessions(open_price, close, log=True)

    assert pd.isna(returns.overnight.iloc[0])
    assert pd.isna(returns.close_to_close.iloc[0])
    pdt.assert_series_equal(
        returns.overnight + returns.intraday,
        returns.close_to_close,
        check_names=False,
        check_exact=False,
        rtol=1e-13,
        atol=1e-13,
    )

    cutoff = 7
    changed_open = open_price.copy()
    changed_close = close.copy()
    changed_open.iloc[cutoff + 1 :] *= 3.0
    changed_close.iloc[cutoff + 1 :] *= 0.25
    changed = decompose_sessions(changed_open, changed_close, log=True)
    for name in ("overnight", "intraday", "close_to_close"):
        pdt.assert_series_equal(
            getattr(returns, name).iloc[: cutoff + 1],
            getattr(changed, name).iloc[: cutoff + 1],
            check_exact=True,
        )


def test_fill_is_the_first_eligible_bar_strictly_after_signal() -> None:
    bars = _recorded_bars()
    signal_row = bars.loc[bars["session"] == "2024-01-03"].iloc[0]
    signal_time = signal_row["event_time"].to_pydatetime()
    eligible = bars.loc[bars["open_time"] > signal_row["event_time"]]
    fill_time = eligible["open_time"].min().to_pydatetime()

    assert fill_time == datetime(2024, 1, 4, 14, 30, tzinfo=UTC)
    ensure_fill_after_signal(signal_time, fill_time)
    with pytest.raises(ValueError, match="strictly later"):
        ensure_fill_after_signal(signal_time, signal_time)
    with pytest.raises(ValueError, match="strictly later"):
        ensure_fill_after_signal(signal_time, signal_row["open_time"].to_pydatetime())


def _spy_bars(sessions: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "symbol": "SPY",
            "session": sessions,
            "event_time": sessions.tz_localize(UTC) + pd.Timedelta(hours=21),
            "available_at": sessions.tz_localize(UTC) + pd.Timedelta(hours=22),
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "adjusted_close": close,
            "volume": 1_000_000.0,
        }
    )


def _prepared_spy_panel(periods: int = 160):  # noqa: ANN202
    from edgestack.data.calendars import NYSECalendar
    from edgestack.pipeline.research import prepare_research

    sessions = NYSECalendar().sessions("2020-01-02", "2020-12-31")[:periods]
    rng = np.random.default_rng(3)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, periods))
    return prepare_research(
        _spy_bars(sessions, close),
        start=sessions.min(),
        end=sessions.max(),
        fomc_dates=pd.DatetimeIndex([]),
        sector_by_symbol={"SPY": "ETF"},
    )


def _monday_spec():  # noqa: ANN202
    from edgestack.models import Direction, HypothesisSpec, RationaleCategory, Session

    return HypothesisSpec(
        family="calendar",
        description="Monday close-to-close",
        predicates={"weekday": "MON"},
        direction=Direction.LONG,
        session=Session.CLOSE_TO_CLOSE,
        holding_period=1,
        rationale=RationaleCategory.NONE,
    )


def test_truncated_prepared_slices_every_date_axis() -> None:
    from edgestack.pipeline.research import _truncated_prepared

    prepared = _prepared_spy_panel()
    truncated = _truncated_prepared(prepared, 100)
    assert len(truncated.dates) == 100
    assert len(truncated.close) == 100
    assert len(truncated.calendar) == 100
    assert len(truncated.market_close) == 100
    assert truncated.sector_by_symbol == prepared.sector_by_symbol


def test_honest_calendar_survivor_passes_pipeline_causality_checks() -> None:
    from edgestack.backtest.costs import CostModel
    from edgestack.pipeline.research import (
        _survivor_causality_evidence,
        run_trial,
    )

    prepared = _prepared_spy_panel()
    trial = run_trial(prepared, _monday_spec(), cost_model=CostModel())
    evidence = _survivor_causality_evidence(prepared, trial, cost_model=CostModel())
    assert evidence["causality_prefix_invariant"]
    assert not evidence["causality_lag_inflation"]
    assert evidence["causality_pass"]
    assert evidence["causality_reason"] == "PASSED"


def test_full_sample_normalized_signal_fails_prefix_invariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from edgestack.backtest.costs import CostModel
    from edgestack.pipeline import research as research_module
    from edgestack.pipeline.research import (
        _survivor_causality_evidence,
        run_trial,
    )

    prepared = _prepared_spy_panel()
    original = research_module._calendar_trial_inputs

    def leaky(prepared_arg, spec):  # noqa: ANN001, ANN202
        signal, returns = original(prepared_arg, spec)
        # Full-sample normalization: past signal values change whenever
        # future sessions are added or removed.
        return signal + float(np.nanmean(returns)), returns

    monkeypatch.setattr(research_module, "_calendar_trial_inputs", leaky)
    trial = run_trial(prepared, _monday_spec(), cost_model=CostModel())
    evidence = _survivor_causality_evidence(prepared, trial, cost_model=CostModel())
    assert not evidence["causality_prefix_invariant"]
    assert not evidence["causality_pass"]
    assert evidence["causality_reason"] == (
        "SIGNAL_CHANGED_WHEN_FUTURE_SESSIONS_REMOVED"
    )


def test_signal_anticipating_the_lagged_fill_is_flagged_as_lag_inflation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from edgestack.backtest.costs import CostModel
    from edgestack.pipeline import research as research_module
    from edgestack.pipeline.research import (
        _survivor_causality_evidence,
        run_trial,
    )

    prepared = _prepared_spy_panel()
    baseline = run_trial(prepared, _monday_spec(), cost_model=CostModel())
    anticipation = baseline.execution_lag + 1
    rng = np.random.default_rng(9)
    returns = rng.normal(0.0, 0.01, len(prepared.dates))

    def leaky(prepared_arg, spec):  # noqa: ANN001, ANN202
        count = len(prepared_arg.dates)
        # The signal "knows" the return earned one session AFTER the mandatory
        # execution lag, so inserting an extra lag makes it strictly better.
        return np.roll(returns[:count], -anticipation), returns[:count]

    monkeypatch.setattr(research_module, "_calendar_trial_inputs", leaky)
    trial = run_trial(prepared, _monday_spec(), cost_model=CostModel())
    evidence = _survivor_causality_evidence(prepared, trial, cost_model=CostModel())
    assert evidence["causality_lag_inflation"]
    assert not evidence["causality_pass"]
    assert evidence["causality_reason"] == "HAC_T_IMPROVED_UNDER_EXTRA_EXECUTION_LAG"


def test_synthetic_one_bar_alpha_collapses_under_timing_shift() -> None:
    fixture = pd.read_csv(
        FIXTURES / "synthetic_one_bar_alpha.csv",
        parse_dates=["session", "signal_available_at"],
    )
    correctly_aligned = fixture["signal"] * fixture["forward_return"]
    one_bar_late = fixture["signal"].shift(1) * fixture["forward_return"]

    assert correctly_aligned.mean() == pytest.approx(0.01)
    assert one_bar_late.dropna().mean() < 0.0
    assert one_bar_late.dropna().mean() < correctly_aligned.mean() * 0.10
