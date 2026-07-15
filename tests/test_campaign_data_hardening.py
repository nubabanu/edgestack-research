from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime

import pandas as pd

from edgestack.config import DataConfig, EdgeStackConfig, StatsConfig
from edgestack.data.calendars import NYSECalendar
from edgestack.data.quality import QAReport, ReconciliationResult, audit_survivorship
from edgestack.models import AssetKey, BarRequest, SourceCapabilities
from edgestack.pipeline.campaign_data import (
    IngestedCampaignData,
    _apply_causal_corrections,
    _CommonProviderEvidence,
    _fetch_individually,
    _fixed_symbol_gate_evidence,
    _merge_reference_factors,
    _persist_correction_evidence,
    deterministic_smoke_data,
)


def test_empty_acquisition_evidence_is_total_and_retains_failures() -> None:
    survivorship = audit_survivorship(("SPY",), (), point_in_time=False)
    qa = QAReport(
        datetime(2024, 1, 2, tzinfo=UTC),
        (),
        (),
        survivorship,
        0.001,
    )
    data = IngestedCampaignData(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DatetimeIndex([]),
        (),
        qa,
        "empty",
        {},
        (),
        {"SPY": "NoDataError: provider unavailable"},
        False,
        False,
        False,
        False,
        False,
    )

    evidence = data.evidence()

    assert evidence["symbols"] == 0
    assert evidence["start"] is None
    assert evidence["end"] is None
    assert evidence["failures"]["SPY"].startswith("NoDataError")
    assert evidence["vix"] == {
        "status": "DATA_UNAVAILABLE",
        "series": "VIXCLS",
        "observations": 0,
        "event_time_column": None,
        "available_at_column": None,
        "source_snapshot": None,
        "failure": None,
    }
    assert not data.passed


def test_vix_reference_merge_preserves_causal_availability_and_evidence() -> None:
    french = pd.DataFrame(
        {
            "session": pd.to_datetime(["1989-12-29", "1990-01-02"]),
            "event_time": pd.to_datetime(
                ["1989-12-29T21:00:00Z", "1990-01-02T21:00:00Z"], utc=True
            ),
            "available_at": pd.to_datetime(
                ["1990-01-02T13:00:00Z", "1990-01-03T13:00:00Z"], utc=True
            ),
            "market_return": [0.001, -0.002],
        }
    )
    vix = pd.DataFrame(
        {
            "session": pd.to_datetime(["1990-01-02", "1990-01-03"]),
            "VIXCLS": [17.24, 18.19],
            "VIXCLS__event_time": pd.to_datetime(
                ["1990-01-02T21:00:00Z", "1990-01-03T21:00:00Z"], utc=True
            ),
            "VIXCLS__available_at": pd.to_datetime(
                ["1990-01-02T21:15:00Z", "1990-01-03T21:15:00Z"], utc=True
            ),
        }
    )

    merged = _merge_reference_factors(french, vix)

    assert merged["session"].tolist() == list(
        pd.to_datetime(["1989-12-29", "1990-01-02", "1990-01-03"])
    )
    assert merged.loc[1, "VIXCLS__available_at"] == pd.Timestamp("1990-01-02T21:15:00Z")
    assert pd.isna(merged.loc[0, "VIXCLS"])
    assert pd.isna(merged.loc[2, "market_return"])

    survivorship = audit_survivorship((), (), point_in_time=False)
    qa = QAReport(datetime(2024, 1, 2, tzinfo=UTC), (), (), survivorship, 0.001)
    campaign = IngestedCampaignData(
        pd.DataFrame(),
        merged,
        pd.DatetimeIndex([]),
        (),
        qa,
        "with-vix",
        {"fred_vixcls": "vix-snapshot-id"},
        (),
        {},
        True,
        False,
        False,
        False,
        False,
    )
    assert campaign.evidence()["vix"] == {
        "status": "AVAILABLE",
        "series": "VIXCLS",
        "observations": 2,
        "event_time_column": "VIXCLS__event_time",
        "available_at_column": "VIXCLS__available_at",
        "source_snapshot": "vix-snapshot-id",
        "failure": None,
    }


def test_smoke_honors_frozen_start_and_has_deterministic_content_identity() -> None:
    config = EdgeStackConfig(
        profile="smoke",
        data=DataConfig(start=date(2020, 2, 1)),
        stats=StatsConfig(seed=123, bootstrap_reps=100, finalist_bootstrap_reps=100),
    )
    first = deterministic_smoke_data(config, as_of=date(2021, 2, 1))
    second = deterministic_smoke_data(config, as_of=date(2021, 2, 1))

    assert first.bars["session"].min() == pd.Timestamp("2020-02-03")
    assert first.snapshot_id == second.snapshot_id
    assert first.source_hashes == second.source_hashes
    pd.testing.assert_frame_equal(first.bars, second.bars)
    assert first.qa.created_at == second.qa.created_at
    assert {item.available_at for item in first.memberships} == {
        datetime(2021, 2, 1, 23, 59, 59, tzinfo=UTC)
    }

    changed = deterministic_smoke_data(
        config.model_copy(
            update={"stats": config.stats.model_copy(update={"seed": 124})}
        ),
        as_of=date(2021, 2, 1),
    )
    assert changed.snapshot_id != first.snapshot_id


def test_old_first_observation_does_not_substitute_for_eligible_coverage() -> None:
    as_of = date(2024, 1, 2)
    expected = NYSECalendar().sessions(date(2004, 1, 2), as_of)
    # This would pass the old first-observation cutoff despite a 20-year hole.
    bars = pd.DataFrame(
        {"symbol": "SPY", "session": [expected[0], expected[-1]], "close": [1, 2]}
    )
    config = EdgeStackConfig(
        profile="full",
        data=DataConfig(
            start=date(2000, 1, 1),
            reconciliation_tickers=("SPY",),
        ),
    )
    reconciliation = ReconciliationResult(
        "SPY", "stooq", "yfinance", len(expected), 1.0, 0.005, 0.0, True
    )
    evidence = _fixed_symbol_gate_evidence(
        config,
        as_of=as_of,
        bars=bars,
        reconciliations=(reconciliation,),
        common_evidence=(_CommonProviderEvidence("SPY", tuple(expected)),),
        failures={},
    )[0]

    assert evidence.observed_span_years > 19.5
    assert evidence.observed_coverage_fraction < 0.001
    assert not evidence.history_pass


def test_reconciliation_requires_roughly_twenty_year_common_session_span() -> None:
    as_of = date(2024, 1, 2)
    expected = NYSECalendar().sessions(date(2004, 1, 2), as_of)
    bars = pd.DataFrame(
        {"symbol": "SPY", "session": expected, "close": range(1, len(expected) + 1)}
    )
    only_recent = expected[expected >= pd.Timestamp("2014-01-01")]
    config = EdgeStackConfig(
        profile="full",
        data=DataConfig(
            start=date(2000, 1, 1),
            reconciliation_tickers=("SPY",),
        ),
    )
    result = ReconciliationResult(
        "SPY", "stooq", "yfinance", len(only_recent), 1.0, 0.005, 0.0, True
    )
    evidence = _fixed_symbol_gate_evidence(
        config,
        as_of=as_of,
        bars=bars,
        reconciliations=(result,),
        common_evidence=(_CommonProviderEvidence("SPY", tuple(only_recent)),),
        failures={},
    )[0]

    assert evidence.history_pass
    assert evidence.common_span_years < 11
    assert not evidence.reconciliation_pass


def test_fetch_individually_retains_every_provider_failure() -> None:
    class FailedSource:
        capabilities = SourceCapabilities("failed")

        async def fetch_bars(self, request: BarRequest):  # type: ignore[no-untyped-def]
            raise RuntimeError(f"cannot fetch {request.asset.symbol}")

    requests = tuple(
        BarRequest(AssetKey(symbol), date(2020, 1, 1), date(2021, 1, 1))
        for symbol in ("SPY", "QQQ")
    )
    batches, failures = asyncio.run(
        _fetch_individually(FailedSource(), requests, concurrency=2)
    )

    assert batches == ()
    assert set(failures) == {"SPY", "QQQ"}
    assert all("RuntimeError" in failure for failure in failures.values())


def test_causal_corrections_preserve_source_close_and_persist_immutable_log(
    tmp_path,
) -> None:
    sessions = pd.date_range("2020-01-02", periods=40, freq="B")
    returns = [0.001 if index % 2 else -0.001 for index in range(40)]
    close = pd.Series(returns).add(1).cumprod().mul(100)
    close.iloc[30] *= 5
    bars = pd.DataFrame(
        {
            "symbol": "ABC",
            "session": sessions,
            "close": close,
            "adjusted_close": close,
        }
    )

    corrected, records = _apply_causal_corrections(bars, sigma=3.0)
    assert records
    assert corrected.loc[30, "close"] == bars.loc[30, "close"]
    assert corrected.loc[30, "adjusted_close"] != bars.loc[30, "adjusted_close"]
    first = _persist_correction_evidence(records, tmp_path)
    second = _persist_correction_evidence(records, tmp_path)
    assert first == second
    assert len(list(tmp_path.glob("*.json"))) == 1
