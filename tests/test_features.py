from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.features.calendar_feats import calendar_features, turn_of_month
from edgestack.features.cross_sectional import (
    canonical_features,
    momentum_12_1,
    proximity_to_high,
)
from edgestack.features.sessions import decompose_sessions


def test_session_log_decomposition_is_additive() -> None:
    index = pd.bdate_range("2024-01-02", periods=5)
    open_ = pd.Series([100.0, 102.0, 101.0, 105.0, 107.0], index=index)
    close = pd.Series([101.0, 101.0, 104.0, 106.0, 108.0], index=index)
    result = decompose_sessions(open_, close, log=True)
    np.testing.assert_allclose(
        (result.overnight + result.intraday).iloc[1:],
        result.close_to_close.iloc[1:],
        rtol=0,
        atol=1e-14,
    )
    assert np.isnan(result.overnight.iloc[0])


def test_turn_of_month_uses_exchange_session_positions() -> None:
    sessions = pd.bdate_range("2024-01-02", "2024-02-29")
    flags = turn_of_month(sessions)
    assert flags.loc[
        [
            "2024-01-30",
            "2024-01-31",
            "2024-02-01",
            "2024-02-02",
            "2024-02-05",
            "2024-02-06",
        ]
    ].tolist() == [False, True, True, True, True, False]


def test_calendar_features_marks_event_week_and_opex() -> None:
    sessions = pd.bdate_range("2024-01-08", "2024-01-19")
    features = calendar_features(
        sessions,
        holidays=["2024-01-15"],
        fomc_dates=["2024-01-17"],
    )
    assert bool(features.loc["2024-01-16", "fomc_event_day_before"])
    assert bool(features.loc["2024-01-17", "fomc_event_day_of"])
    assert bool(features.loc["2024-01-19", "opex_week"])
    assert bool(features.loc["2024-01-12", "pre_holiday"])
    assert bool(features.loc["2024-01-16", "post_holiday"])


def test_cross_sectional_features_use_only_trailing_prices() -> None:
    index = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame(
        {"A": np.arange(1.0, 301.0), "B": np.linspace(100.0, 80.0, 300)}, index=index
    )
    momentum = momentum_12_1(prices)
    expected = prices.iloc[278] / prices.iloc[47] - 1.0
    np.testing.assert_allclose(momentum.iloc[299], expected)
    proximity = proximity_to_high(prices)
    assert proximity.iloc[-1, 0] == 1.0
    assert proximity.iloc[-1, 1] < 1.0


def test_all_close_derived_features_are_prefix_invariant() -> None:
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2020-01-02", periods=320)
    innovations = rng.normal(0.0002, 0.01, size=(len(index), 12))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(innovations, axis=0)),
        index=index,
        columns=[f"S{number:02d}" for number in range(12)],
    )
    complete = canonical_features(prices)

    for endpoint in (260, 289, 319):
        prefix = canonical_features(prices.iloc[: endpoint + 1])
        for name in ("momentum", "reversal", "low_volatility", "high_proximity"):
            np.testing.assert_allclose(
                getattr(prefix, name).iloc[-1],
                getattr(complete, name).iloc[endpoint],
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
