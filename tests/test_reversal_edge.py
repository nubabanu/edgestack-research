from __future__ import annotations

import exchange_calendars
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest

from edgestack.backtest.engine import overlapping_cohort_targets
from edgestack.edges.global_holdout import GlobalHoldoutRecord
from edgestack.edges.reversal_edge import (
    _assert_record_matches_freeze,
    _execute_with_fill_availability,
    _filter_frozen_universe,
    _normalize_known_zipline_calendar_exception,
)


def test_five_name_half_gross_contract_respects_name_cap() -> None:
    entries = np.zeros((7, 8), dtype=float)
    entries[:, :5] = 0.2
    targets = overlapping_cohort_targets(entries, holding_period=5) * 0.5
    assert np.allclose(targets.sum(axis=1), 0.5)
    assert float(targets.max()) <= 0.100000000001


def test_repeated_daily_cohort_contributes_two_percent_per_name() -> None:
    entries = np.zeros((5, 10), dtype=float)
    for row in range(5):
        entries[row, row : row + 5] = 0.2
    targets = overlapping_cohort_targets(entries, holding_period=5) * 0.5
    mature = pd.Series(targets[-1])
    assert mature.iloc[4] == 0.10
    assert mature.iloc[0] == 0.02
    assert mature.sum() == 0.5


def test_zero_volume_closing_fill_carries_prior_position() -> None:
    desired = np.array([[0.0], [0.1], [0.0], [0.0]])
    close = np.full_like(desired, 100.0)
    volume = np.array([[1_000.0], [0.0], [2_000.0], [2_000.0]])
    executed = _execute_with_fill_availability(
        desired, close=close, volume=volume, gross_cap=0.5
    )
    # Entry for row 1 fills on row 0. The intended exit for row 2 cannot fill
    # on zero-volume row 1 and therefore completes one close later.
    np.testing.assert_allclose(executed[:, 0], [0.0, 0.1, 0.1, 0.0])


def test_locked_position_reserves_capacity_before_new_fills() -> None:
    desired = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.2, 0.1],
            [0.0, 0.25, 0.25],
        ]
    )
    close = np.full_like(desired, 100.0)
    volume = np.array(
        [
            [1_000.0, 1_000.0, 1_000.0],
            [0.0, 1_000.0, 1_000.0],
            [1_000.0, 1_000.0, 1_000.0],
        ]
    )
    executed = _execute_with_fill_availability(
        desired, close=close, volume=volume, gross_cap=0.5
    )
    # The first 20% position is locked by zero volume. The two tradable 25%
    # requests share only the remaining 30% capacity.
    np.testing.assert_allclose(executed[2], [0.2, 0.15, 0.15])
    assert np.all(np.abs(executed).sum(axis=1) <= 0.500000000001)


def test_only_exact_preregistered_zipline_calendar_exception_is_normalized() -> None:
    validation = {
        "zipline_tolerance_bps_per_trade": 1.0,
        "zipline_known_calendar_exceptions": [
            {
                "exception_id": "test",
                "session": "2005-06-01",
                "canonical_close_utc": "2005-06-01T19:56:00+00:00",
                "backend_close_utc": "2005-06-01T20:00:00+00:00",
                "expected_event_count": 2,
                "chunk_start": "2004-01-02",
                "chunk_end": "2005-12-30",
                "zipline_backend": "zipline-reloaded-3.1.1-in-memory-adjusted-ohlcv",
                "zipline_version": "3.1.1",
                "canonical_calendar": "pandas-market-calendars==5.4.0",
                "backend_calendar": "exchange-calendars==4.13.2",
                "source": "documented-source",
            }
        ],
    }
    payload = {
        "passed": False,
        "timestamps_match": False,
        "trade_count": 2,
        "vector_trade_count": 2,
        "difference_bps_per_trade": 0.01,
        "reason": "transaction timestamps differ (documented fixture)",
        "backend": "zipline-reloaded-3.1.1-in-memory-adjusted-ohlcv",
        "convention_supported": True,
        "start": "2004-01-02",
        "end": "2005-12-30",
        "missing_fill_events": (
            (3, "2005-06-01T19:56:00+00:00"),
            (7, "2005-06-01T19:56:00+00:00"),
        ),
        "extra_fill_events": (
            (3, "2005-06-01T20:00:00+00:00"),
            (7, "2005-06-01T20:00:00+00:00"),
        ),
    }
    result = _normalize_known_zipline_calendar_exception(payload, validation)
    assert result["raw_passed"] is False
    assert result["normalized_pass"] is True
    assert result["normalized_exception_event_count"] == 2


def test_unknown_zipline_timestamp_mismatch_remains_a_failure() -> None:
    validation = {
        "zipline_tolerance_bps_per_trade": 1.0,
        "zipline_known_calendar_exceptions": [
            {
                "exception_id": "test",
                "session": "2005-06-01",
                "canonical_close_utc": "2005-06-01T19:56:00+00:00",
                "backend_close_utc": "2005-06-01T20:00:00+00:00",
                "expected_event_count": 1,
                "chunk_start": "2004-01-02",
                "chunk_end": "2005-12-30",
                "zipline_backend": "zipline-reloaded-3.1.1-in-memory-adjusted-ohlcv",
                "zipline_version": "3.1.1",
                "canonical_calendar": "pandas-market-calendars==5.4.0",
                "backend_calendar": "exchange-calendars==4.13.2",
            }
        ],
    }
    payload = {
        "passed": False,
        "timestamps_match": False,
        "trade_count": 1,
        "vector_trade_count": 1,
        "difference_bps_per_trade": 0.01,
        "reason": "transaction timestamps differ (unknown fixture)",
        "backend": "zipline-reloaded-3.1.1-in-memory-adjusted-ohlcv",
        "convention_supported": True,
        "start": "2004-01-02",
        "end": "2005-12-30",
        "missing_fill_events": ((3, "2005-06-02T20:00:00+00:00"),),
        "extra_fill_events": ((3, "2005-06-02T20:01:00+00:00"),),
    }
    result = _normalize_known_zipline_calendar_exception(payload, validation)
    assert result["normalized_pass"] is False
    assert result["normalized_exception_event_count"] == 0


def test_pinned_calendars_expose_documented_2005_close_difference() -> None:
    canonical = mcal.get_calendar("NYSE").schedule(
        start_date="2005-06-01", end_date="2005-06-01"
    )
    backend = exchange_calendars.get_calendar(
        "XNYS", start="2005-06-01", end="2005-06-02", side="right"
    ).schedule
    assert canonical.iloc[0]["market_close"].isoformat() == (
        "2005-06-01T19:56:00+00:00"
    )
    assert backend.iloc[0]["close"].isoformat() == "2005-06-01T20:00:00+00:00"


def test_frozen_universe_drops_extra_equities_and_etfs() -> None:
    bars = pd.DataFrame(
        {
            "symbol": ["KEEP", "ROGUE", "SPY"],
            "asset_type": ["equity", "equity", "etf"],
            "close": [10.0, 1.0, 100.0],
        }
    )
    result = _filter_frozen_universe(bars, {"KEEP": "equity", "SPY": "etf"})
    assert result["symbol"].tolist() == ["KEEP", "SPY"]


def test_sealed_record_from_another_freeze_is_rejected() -> None:
    record = GlobalHoldoutRecord(
        scope_id="scope",
        program_id="program",
        market="XNYS",
        promotion_class="FINAL",
        data_snapshot_id="snapshot",
        start="2023-01-01",
        end="2025-12-31",
        state="SEALED",
        freeze_id="other-freeze",
        evaluator_sha256="evaluator",
        consumed_at="now",
        result_sha256="result",
        result_path="result.json",
    )
    with pytest.raises(RuntimeError, match="another freeze"):
        _assert_record_matches_freeze(
            record,
            {
                "holdout_program_id": "program",
                "holdout_market": "XNYS",
                "promotion_class": "FINAL",
                "data_snapshot_id": "snapshot",
                "holdout_start": "2023-01-01",
                "holdout_end": "2025-12-31",
                "freeze_id": "current-freeze",
                "evaluator_sha256": "evaluator",
            },
        )
