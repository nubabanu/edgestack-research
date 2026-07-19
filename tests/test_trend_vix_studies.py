from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from edgestack.data.calendars import NYSECalendar
from edgestack.edges.trend_study import _monthly_signal, build_streams
from edgestack.edges.vix_study import _expanding_percentile
from edgestack.edges.vix_study import build_streams as vix_build_streams


@pytest.fixture(scope="module")
def synthetic_panel() -> dict[str, pd.DataFrame]:
    sessions = NYSECalendar().sessions("2015-01-02", "2020-12-31")
    rng = np.random.default_rng(11)
    frames: dict[str, pd.DataFrame] = {}
    symbols = ["SPY", "QQQ"]
    closes = {}
    for symbol in symbols:
        drift = 0.0006 if symbol == "SPY" else 0.0004
        closes[symbol] = 100.0 * np.cumprod(
            1.0 + rng.normal(drift, 0.01, len(sessions))
        )
    close = pd.DataFrame(closes, index=sessions)
    frames["adjusted_close"] = close
    frames["close"] = close.copy()
    frames["open"] = close.shift(1).fillna(close.iloc[0])
    frames["volume"] = pd.DataFrame(1e6, index=sessions, columns=symbols)
    frames["asset_types"] = pd.Series({"SPY": "etf", "QQQ": "etf"})
    return frames


def _config(instruments: list[str]) -> dict:
    return {
        "campaign_id": "trend-test",
        "declared_family": {
            "instruments": instruments,
            "real_trial_count": 27,
            "accounting_family_size": 81,
        },
        "holdout": {"start": "2021-01-01"},
    }


def test_monthly_signals_are_causal_and_boolean(synthetic_panel: dict) -> None:
    series = synthetic_panel["adjusted_close"]["SPY"]
    for name in ("tsmom_12_1", "above_sma200", "above_10m_sma"):
        signal = _monthly_signal(series, series, name)
        assert signal.dtype == bool
        # One decision per month, dated at month-end sessions.
        assert signal.index.is_monotonic_increasing
        periods = pd.PeriodIndex(signal.index, freq="M")
        assert periods.is_unique


def test_trend_streams_cost_only_on_flips(synthetic_panel: dict) -> None:
    gross, net, definitions, benchmark = build_streams(
        _config(["SPY", "QQQ"]), synthetic_panel, date(2021, 1, 1)
    )
    assert len(definitions) == 6  # 2 instruments x 3 signals
    for trial_id, definition in definitions.items():
        costed_days = int(
            (gross[trial_id].fillna(0.0) - net[trial_id].fillna(0.0)).gt(1e-12).sum()
        )
        # Cost hits at most once per flip (a flip on an inactive day still
        # charges), never more.
        assert costed_days <= definition["flips"]
        assert definition["flips"] < 40  # monthly cadence keeps flips rare
    assert benchmark.index.equals(gross.index)


def test_trend_positions_start_after_the_decision_close(
    synthetic_panel: dict,
) -> None:
    gross, _, _, _ = build_streams(
        _config(["SPY", "QQQ"]), synthetic_panel, date(2021, 1, 1)
    )
    first_active = gross["trend|SPY|above_10m_sma"].first_valid_index()
    # 10 month-ends of history are needed before the first decision; exposure
    # can only begin strictly after that month-end close.
    assert first_active is not None
    assert first_active > gross.index[200]


def test_expanding_percentile_is_causal() -> None:
    rng = np.random.default_rng(3)
    values = pd.Series(
        np.r_[10.0 + rng.normal(0, 0.5, 300), np.full(50, 30.0)],
        index=pd.RangeIndex(350),
    )
    percentile = _expanding_percentile(values, minimum=252)
    assert percentile.iloc[:252].isna().all()
    # A value far above all prior history ranks at the top regardless of what
    # comes later; an ordinary mid-regime value ranks mid-pack.
    assert percentile.iloc[300] > 0.95
    assert 0.05 < percentile.iloc[260] < 0.95
    # Ties mid-rank: a constant tail cannot all claim the top percentile.
    assert percentile.iloc[340] < percentile.iloc[300]


def test_vix_streams_flip_counts_and_exposure(synthetic_panel: dict) -> None:
    sessions = synthetic_panel["adjusted_close"].index
    rng = np.random.default_rng(5)
    vix = pd.Series(
        18.0 + np.abs(np.cumsum(rng.normal(0, 0.6, len(sessions)))), index=sessions
    )
    config = {
        "campaign_id": "vix-test",
        "declared_family": {
            "thresholds": [0.8],
            "real_trial_count": 6,
            "accounting_family_size": 18,
        },
        "holdout": {"start": "2021-01-01"},
    }
    gross, net, definitions, benchmark = vix_build_streams(
        config, synthetic_panel, vix, date(2021, 1, 1)
    )
    assert set(definitions) == {"vix|0.80|RISK_OFF", "vix|0.80|RISK_SEEK"}
    off = definitions["vix|0.80|RISK_OFF"]
    seek = definitions["vix|0.80|RISK_SEEK"]
    # The two responses exactly partition the sessions with a decided
    # (post-warmup) percentile, lagged one session for exposure.
    decided = _expanding_percentile(vix.reindex(sessions)).notna()
    expected = decided.shift(1, fill_value=False).mean()
    combined = off["active_fraction"] + seek["active_fraction"]
    assert combined == pytest.approx(float(expected), abs=1e-9)
    # Complementary responses share every threshold crossing; only the
    # warm-up entry can differ, by exactly one flip.
    assert abs(off["flips"] - seek["flips"]) <= 1
    assert off["flips"] > 0
