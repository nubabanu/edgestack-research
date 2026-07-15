from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from edgestack.data.calendars import (
    NYSECalendar,
    fomc_event_labels,
    monthly_opex_sessions,
    parse_fomc_calendar_html,
    parse_fomc_historical_html,
)
from edgestack.data.quality import (
    audit_instrument,
    causal_outlier_mask,
    causal_winsorize_prices,
    reconcile_action_stratified_returns,
    reconcile_adjusted_series,
)


def test_nyse_early_close_and_good_friday_opex() -> None:
    calendar = NYSECalendar()
    assert calendar.close_time(date(2024, 11, 29)).hour == 18  # UTC
    assert monthly_opex_sessions(date(2019, 4, 1), date(2019, 4, 30))[
        0
    ] == pd.Timestamp("2019-04-18")
    calendar.assert_reference_match(date(2024, 1, 1), date(2024, 12, 31))


def test_fomc_parsers_and_event_week() -> None:
    current_html = """
    <h4><a>2024 FOMC Meetings</a></h4>
    <div class="fomc-meeting__month"><strong>March</strong></div>
    <div class="fomc-meeting__date">19-20*</div>
    """
    historical_html = '<h5 class="panel-heading">January 29-30 Meeting - 2019</h5>'
    current = parse_fomc_calendar_html(current_html)
    historical = parse_fomc_historical_html(historical_html, 2019, "official")
    assert current[0].end == date(2024, 3, 20)
    assert current[0].projections
    assert historical[0].start == date(2019, 1, 29)

    labels = fomc_event_labels(date(2024, 3, 18), date(2024, 3, 22), current)
    row = labels.set_index("session")
    assert bool(row.loc[pd.Timestamp("2024-03-19"), "fomc_day_before"])
    assert bool(row.loc[pd.Timestamp("2024-03-20"), "fomc_day_of"])
    assert row["fomc_event_week"].all()


def test_missing_denominator_and_causal_outlier_prefix_invariance() -> None:
    frame = pd.DataFrame(
        {
            "symbol": "ABC",
            "session": pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-05"]),
            "close": [100.0, 101.0, 102.0],
            "volume": [1000, 1000, 1000],
        }
    )
    result = audit_instrument(frame)
    assert result.expected_sessions == 4
    assert result.missing_sessions == 1
    assert result.missing_fraction == 0.25

    rng = np.random.default_rng(42)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80))))
    original = causal_outlier_mask(prices)
    extended = causal_outlier_mask(
        pd.concat([prices, pd.Series([1e9])], ignore_index=True)
    )
    pd.testing.assert_series_equal(original, extended.iloc[:-1], check_names=False)


def test_reconciliation_uses_rebased_total_return_levels() -> None:
    sessions = pd.date_range("2024-01-02", periods=3, freq="B")
    left = pd.DataFrame({"session": sessions, "close": [100, 101, 102]})
    right = pd.DataFrame({"session": sessions, "close": [50, 50.5, 51]})
    result = reconcile_adjusted_series(
        left,
        right,
        symbol="ABC",
        source_a="one",
        source_b="two",
    )
    assert result.passed
    assert result.agreement_fraction == 1


def test_action_stratified_reconciliation_separates_prices_and_actions() -> None:
    sessions = pd.date_range("2024-01-02", periods=5, freq="B")
    left = pd.DataFrame({"session": sessions, "close": [50.0, 50.5, 50.0, 51.0, 51.5]})
    right = pd.DataFrame(
        {
            "session": sessions,
            "close": [100.0, 101.0, 100.0, 102.0, 103.0],
            "adjusted_close": [100.0, 101.0, 101.0, 103.02, 104.03],
            "dividend": [0.0, 0.0, 1.0, 0.0, 0.0],
            "split_factor": [1.0] * 5,
        }
    )
    result = reconcile_action_stratified_returns(
        left,
        right,
        symbol="ABC",
        source_a="stooq",
        source_b="yfinance",
        comparison_start=date(2024, 1, 2),
    )

    assert result.passed
    assert result.method == "action_stratified_returns"
    assert result.price_observations == 3
    assert result.excluded_action_sessions == 1
    assert result.action_sessions == 1
    assert result.action_agreement_fraction == 1
    assert result.provenance_warning is not None
    assert "SINGLE_SOURCE_ACTIONS" in result.provenance_warning

    conflicting = left.copy()
    conflicting.loc[4, "close"] = 60.0
    failed = reconcile_action_stratified_returns(
        conflicting,
        right,
        symbol="ABC",
        source_a="stooq",
        source_b="yfinance",
        comparison_start=date(2024, 1, 2),
    )
    assert not failed.passed


def test_linear_winsorizer_matches_naive_expanding_reference() -> None:
    rng = np.random.default_rng(77)
    returns = rng.normal(0.0002, 0.012, 400)
    returns[[80, 190, 310]] = [0.60, -0.55, 0.75]
    prices = 100.0 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame(
        {
            "symbol": "ABC",
            "session": pd.bdate_range("2020-01-02", periods=len(prices)),
            "close": prices,
        }
    )

    corrected, records = causal_winsorize_prices(
        frame, symbol="ABC", sigma=4.0, minimum_history=20
    )

    expected = prices.copy()
    history: list[float] = []
    for index in range(1, len(prices)):
        raw = float(np.log(prices[index] / expected[index - 1]))
        accepted = raw
        if len(history) >= 20:
            mean = float(np.mean(history))
            deviation = float(np.std(history, ddof=1))
            if deviation > 0.0:
                accepted = min(
                    max(raw, mean - 4.0 * deviation),
                    mean + 4.0 * deviation,
                )
        expected[index] = expected[index - 1] * np.exp(accepted)
        history.append(accepted)

    np.testing.assert_allclose(corrected["research_close"], expected, rtol=1e-12)
    assert records
