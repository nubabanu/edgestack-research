from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgestack.live.state import SignalDecision, StateStore
from edgestack.models import (
    AssetKey,
    CorporateEvent,
    CorporateEventKind,
    DataTier,
    EstimateVintage,
    IntradayMarketRecord,
    MarketRecordKind,
    MembershipInterval,
    TickerValidityInterval,
)
from edgestack.v2.campaign import create_free_only_diagnostic, freeze_forward_model
from edgestack.v2.gates import CapabilityStatus, evaluate_capabilities
from edgestack.v2.importers import bounded_intraday_storage, import_pit_memberships
from edgestack.v2.metrics import gap_adjusted_stop_return, loss_metrics
from edgestack.v2.provenance import causal_prefix, load_hash_pinned
from edgestack.v2.research import (
    Horizon,
    PortfolioForm,
    SignalFamily,
    TrialSpec,
    declared_trials,
    run_trial,
)
from edgestack.v2.sec_events import events_from_sec_submissions
from edgestack.v2.veto import (
    VetoEvidence,
    VetoKind,
    VetoSpec,
    enabled_plateau,
    event_is_vetoed,
)

HASH = "a" * 64


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    frame.to_csv(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hash_pinned_pit_import_and_changed_hash_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "pit.csv"
    frame = pd.DataFrame(
        {
            "security_id": ["perm-1"],
            "ticker": ["AAA"],
            "exchange": ["US"],
            "start": ["2020-01-01"],
            "end": ["2021-01-01"],
            "event_time": ["2020-01-01T00:00:00Z"],
            "available_at": ["2019-12-20T00:00:00Z"],
            "source": ["licensed-fixture"],
            "revision": ["1"],
            "fetched_at": ["2024-01-01T00:00:00Z"],
            "content_hash": [HASH],
        }
    )
    digest = _write_csv(path, frame)
    memberships = import_pit_memberships(path, digest)
    assert memberships[0].security_id == "perm-1"
    assert memberships[0].data_tier is DataTier.POINT_IN_TIME
    with pytest.raises(ValueError, match="hash mismatch"):
        load_hash_pinned(path, "0" * 64)


def test_malformed_overlapping_pit_intervals_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "overlap.csv"
    frame = pd.DataFrame(
        {
            "security_id": ["perm-1", "perm-1"],
            "ticker": ["AAA", "AAA"],
            "start": ["2020-01-01", "2020-06-01"],
            "end": ["2021-01-01", "2022-01-01"],
            "event_time": ["2020-01-01T00:00:00Z"] * 2,
            "available_at": ["2020-01-02T00:00:00Z"] * 2,
            "source": ["fixture"] * 2,
            "revision": ["1", "2"],
            "fetched_at": ["2024-01-01T00:00:00Z"] * 2,
            "content_hash": [HASH] * 2,
        }
    )
    digest = _write_csv(path, frame)
    with pytest.raises(ValueError, match="overlapping"):
        import_pit_memberships(path, digest)


def test_free_only_gates_are_unavailable_and_approximation_fails() -> None:
    report = evaluate_capabilities()
    assert not report.promotable
    assert report.pit_membership.status is CapabilityStatus.DATA_UNAVAILABLE
    approximate = MembershipInterval(
        AssetKey("AAA"),
        date(2020, 1, 1),
        None,
        available_at=datetime(2020, 1, 2, tzinfo=UTC),
        security_id="perm-1",
        data_tier=DataTier.PIT_APPROXIMATION,
        fetched_at=datetime(2024, 1, 1, tzinfo=UTC),
        content_hash=HASH,
    )
    ticker = TickerValidityInterval(
        "perm-1",
        "AAA",
        "US",
        datetime(2020, 1, 1, tzinfo=UTC),
        None,
        datetime(2020, 1, 2, tzinfo=UTC),
        "fixture",
        datetime(2024, 1, 1, tzinfo=UTC),
        HASH,
    )
    assert (
        evaluate_capabilities((approximate,), (ticker,)).pit_membership.status
        is CapabilityStatus.FAIL
    )


def test_entitled_capability_fixtures_pass_all_three_gates() -> None:
    known = datetime(2024, 1, 2, tzinfo=UTC)
    member = MembershipInterval(
        AssetKey("AAA"),
        date(2020, 1, 1),
        None,
        available_at=known,
        security_id="perm-1",
        data_tier=DataTier.POINT_IN_TIME,
        fetched_at=known,
        content_hash=HASH,
    )
    ticker = TickerValidityInterval(
        "perm-1", "AAA", "US", known, None, known, "licensed", known, HASH
    )
    estimate = EstimateVintage(
        "est-1",
        "perm-1",
        "EPS",
        date(2024, 3, 31),
        1.0,
        known,
        known,
        "v1",
        "licensed",
        known,
        HASH,
    )
    records = tuple(
        IntradayMarketRecord(
            "perm-1", kind, known, known, "licensed", "v1", known, HASH
        )
        for kind in (
            MarketRecordKind.NBBO,
            MarketRecordKind.TRADE,
            MarketRecordKind.IMBALANCE,
            MarketRecordKind.AUCTION_PRINT,
        )
    )
    assert evaluate_capabilities((member,), (ticker,), (estimate,), records).promotable


def test_intraday_storage_keeps_full_minutes_and_finalist_close_ticks_only() -> None:
    def record(
        security_id: str, kind: MarketRecordKind, hour: int, minute: int
    ) -> IntradayMarketRecord:
        event = datetime(2024, 1, 2, hour, minute, tzinfo=UTC)
        return IntradayMarketRecord(
            security_id, kind, event, event, "licensed", "1", event, HASH
        )

    # January ET is UTC-5: 20:30 UTC = 15:30 ET.
    records = (
        record("ordinary", MarketRecordKind.MINUTE_BAR, 15, 0),
        record("ordinary", MarketRecordKind.TRADE, 20, 30),
        record("final", MarketRecordKind.TRADE, 20, 30),
        record("final", MarketRecordKind.TRADE, 19, 0),
    )
    retained = bounded_intraday_storage(
        records, finalist_security_ids=frozenset({"final"})
    )
    assert [(item.security_id, item.kind) for item in retained] == [
        ("ordinary", MarketRecordKind.MINUTE_BAR),
        ("final", MarketRecordKind.TRADE),
    ]


def test_causal_prefix_ignores_future_revision(tmp_path: Path) -> None:
    path = tmp_path / "vintages.csv"
    frame = pd.DataFrame(
        {
            "event_time": ["2024-01-01T00:00:00Z"] * 2,
            "available_at": ["2024-01-02T00:00:00Z", "2024-01-10T00:00:00Z"],
            "source": ["fixture"] * 2,
            "revision": ["1", "2"],
            "fetched_at": ["2024-02-01T00:00:00Z"] * 2,
            "content_hash": [HASH] * 2,
        }
    )
    digest = _write_csv(path, frame)
    loaded = load_hash_pinned(path, digest)
    prefix = causal_prefix(loaded.frame, datetime(2024, 1, 5, tzinfo=UTC))
    assert prefix["revision"].astype(str).tolist() == ["1"]


def test_loss_metrics_and_gap_through_stop_hand_calculations() -> None:
    outcomes = np.array([-0.20, -0.10, 0.10, 0.20])
    paths = np.array(
        [
            [-0.10, -0.1111111111],
            [-0.05, -0.0526315789],
            [0.05, 0.0476190476],
            [0.10, 0.0909090909],
        ]
    )
    metrics = loss_metrics(outcomes, path_returns=paths, bootstrap_draws=500)
    assert metrics.loss_probability == 0.5
    assert metrics.expected_shortfall_95 == pytest.approx(0.20)
    assert metrics.trade_mae == pytest.approx(-0.20)
    assert metrics.maximum_losing_streak == 2
    assert gap_adjusted_stop_return(
        entry_price=100,
        stop_price=95,
        first_tradable_price=90,
        direction="LONG",
        costs_bps=10,
    ) == pytest.approx(-0.101)


def test_declared_trial_count_and_financing_make_leverage_costlier() -> None:
    trials = declared_trials()
    assert len(trials) == 900
    assert len({item.trial_id for item in trials}) == 900
    index = pd.bdate_range("2022-01-03", periods=560)
    trends = np.linspace(-0.0005, 0.001, 10)
    close = pd.DataFrame(
        {
            f"S{column}": 100 * np.cumprod(np.full(len(index), 1 + trend))
            for column, trend in enumerate(trends)
        },
        index=index,
    )
    baseline = TrialSpec(
        Horizon.MONTHLY,
        SignalFamily.MOMENTUM_252_SKIP_21,
        "LONG",
        PortfolioForm.LONG_ONLY,
        VetoSpec(VetoKind.NONE),
        1.0,
    )
    leveraged = TrialSpec(
        Horizon.MONTHLY,
        SignalFamily.MOMENTUM_252_SKIP_21,
        "LONG",
        PortfolioForm.LONG_ONLY,
        VetoSpec(VetoKind.NONE),
        2.0,
    )
    one = run_trial(baseline, close, annual_sofr=0.05)
    two = run_trial(leveraged, close, annual_sofr=0.05)
    assert two.net_mean < 2 * one.net_mean
    assert two.forward_promotion_required


def test_veto_requires_causal_events_and_adjacent_plateau() -> None:
    decision = datetime(2024, 1, 2, tzinfo=UTC)
    event = CorporateEvent(
        "event-1",
        "perm-1",
        CorporateEventKind.EARNINGS,
        decision + timedelta(days=2),
        decision - timedelta(days=1),
        "licensed",
        "1",
        decision,
        HASH,
    )
    assert event_is_vetoed(
        VetoSpec(VetoKind.EARNINGS_WINDOW),
        (event,),
        decision_time=decision,
        hold_end=decision + timedelta(days=5),
    )
    future_known = CorporateEvent(
        "event-2",
        "perm-1",
        CorporateEventKind.EARNINGS,
        decision + timedelta(days=2),
        decision + timedelta(days=1),
        "licensed",
        "1",
        decision + timedelta(days=1),
        HASH,
    )
    assert not event_is_vetoed(
        VetoSpec(VetoKind.EARNINGS_WINDOW),
        (future_known,),
        decision_time=decision,
        hold_end=decision + timedelta(days=5),
    )
    evidence = tuple(
        VetoEvidence(
            VetoSpec(VetoKind.GAP_PERCENT, threshold), 0.001, -0.01, -0.01, 0.01, sharpe
        )
        for threshold, sharpe in ((0.03, 0.9), (0.05, 1.0), (0.08, 0.2))
    )
    assert enabled_plateau(evidence) == ("GAP_PERCENT:0.03", "GAP_PERCENT:0.05")


def test_sec_acceptance_timestamp_is_preserved_and_date_only_is_skipped() -> None:
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["a", "b"],
                "acceptanceDateTime": ["2024-01-02T21:01:02Z", None],
                "form": ["8-K", "10-Q"],
                "items": ["2.02", ""],
                "primaryDocument": ["results.htm", "quarterly.htm"],
            }
        }
    }
    events = events_from_sec_submissions(
        payload, security_id="perm-1", fetched_at=datetime(2024, 1, 3, tzinfo=UTC)
    )
    assert len(events) == 1
    assert events[0].kind is CorporateEventKind.PRELIMINARY_RESULTS
    assert events[0].available_at == datetime(2024, 1, 2, 21, 1, 2, tzinfo=UTC)


def test_forward_ledger_is_atomic_idempotent_and_replay_only(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "v2.sqlite")
    decision_time = datetime(2024, 1, 2, 21, tzinfo=UTC)
    decision = SignalDecision(
        "decision-1",
        "perm-1",
        "AAA",
        "CANDIDATE",
        "LONG",
        None,
        1.0,
        decision_time + timedelta(days=1),
        decision_time + timedelta(days=22),
        1.0,
        21,
        {"rank": 1},
    )
    assert store.create_signal_run(
        run_id="run-1",
        decision_time=decision_time,
        created_at=decision_time,
        causal_data_hash=HASH,
        config_hash=HASH,
        decisions=(decision,),
    )
    assert not store.create_signal_run(
        run_id="run-1",
        decision_time=decision_time,
        created_at=decision_time,
        causal_data_hash=HASH,
        config_hash=HASH,
        decisions=(decision,),
    )
    fill = decision_time + timedelta(days=1)
    store.record_paper_fill("decision-1", fill, 100)
    assert store.record_paper_mark(
        "decision-1",
        mark_at=fill,
        available_at=fill + timedelta(minutes=1),
        price=95,
        causal_data_hash=HASH,
    ) == pytest.approx((-0.05, -0.05))
    with pytest.raises(ValueError, match="retroactive"):
        store.record_paper_mark(
            "decision-1",
            mark_at=fill,
            available_at=fill + timedelta(minutes=1),
            price=96,
            causal_data_hash=HASH,
        )
    exit_at = fill + timedelta(days=21)
    store.record_paper_outcome(
        "decision-1",
        exit_at=exit_at,
        exit_price=105,
        gross_pnl=50,
        costs=2,
        stop_gap=None,
        stop_slippage=None,
        transition_reason="TIME_EXIT",
        recorded_at=exit_at,
    )
    scorecard = store.paper_scorecard()
    assert scorecard["decisions"] == 1
    assert scorecard["total_net_pnl"] == 48
    with pytest.raises(sqlite3.IntegrityError):
        store.record_paper_outcome(
            "decision-1",
            exit_at=exit_at,
            exit_price=106,
            gross_pnl=60,
            costs=2,
            stop_gap=None,
            stop_slippage=None,
            transition_reason="REWRITE",
            recorded_at=exit_at,
        )


def test_v2_campaign_namespace_does_not_create_or_read_v1_holdout(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("campaign_namespace: loss-aware-v2\n", encoding="utf-8")
    output = create_free_only_diagnostic(
        tmp_path / "artifacts", campaign_id="forward-001", config_path=config
    )
    payload = output.read_text(encoding="utf-8")
    assert "FORBIDDEN_NOT_READ" in payload
    assert "DATA_UNAVAILABLE" in payload
    assert not (tmp_path / "artifacts" / "campaigns").exists()
    with pytest.raises(RuntimeError, match="DATA_UNAVAILABLE"):
        freeze_forward_model(
            tmp_path / "artifacts",
            campaign_id="forward-001",
            model={"family": "momentum"},
            config_sha256=HASH,
            data_contract_sha256=HASH,
            capabilities=evaluate_capabilities(),
        )
