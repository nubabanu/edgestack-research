from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from edgestack.live.demo import run
from edgestack.live.monitor import revalidate_pending
from edgestack.live.scanner import ScoredCandidate, scan
from edgestack.live.state import StateStore
from edgestack.models import (
    AssetKey,
    Direction,
    EntryPlan,
    OrderType,
    Quote,
    Recommendation,
    RecommendationState,
    TimingVerdict,
)


def _recommendation(now: datetime) -> Recommendation:
    plan = EntryPlan(
        "immediate_at_close",
        OrderType.MOC,
        Direction.LONG,
        TimingVerdict.WAIT_UNTIL,
        now + timedelta(hours=3),
        "test",
        validity_end=now + timedelta(days=2),
        data_timestamp=now,
    )
    return Recommendation(
        "rec-1",
        AssetKey("AAPL"),
        Direction.LONG,
        75,
        0.01,
        (0, 0.02),
        3,
        plan,
        ("edge",),
        now,
    )


def test_state_transition_and_dedupe(tmp_path) -> None:
    now = datetime(2024, 1, 2, tzinfo=UTC)
    store = StateStore(tmp_path / "state.sqlite")
    recommendation = _recommendation(now)
    assert store.add(recommendation, ["console"])
    assert not store.add(recommendation, ["console"])
    store.transition("rec-1", RecommendationState.WAITING, "wait", "wait", ["console"])
    with pytest.raises(ValueError, match="invalid transition"):
        store.transition("rec-1", RecommendationState.EXITED, "bad", "bad", ["console"])


def test_monitor_revalidates_and_cancels_stale() -> None:
    now = datetime(2024, 1, 2, tzinfo=UTC)
    recommendation = _recommendation(now)
    fresh = Quote(
        AssetKey("AAPL"),
        101,
        now + timedelta(hours=3),
        now + timedelta(hours=3),
        "test",
    )
    result = revalidate_pending(
        recommendation,
        fresh,
        scan_price=100,
        atr_value=2,
        regime_gate_passes=True,
        now=now + timedelta(hours=3, minutes=1),
    )
    assert result.state is RecommendationState.CONFIRMED
    stale = Quote(AssetKey("AAPL"), 101, now, now + timedelta(hours=4), "test")
    result = revalidate_pending(
        recommendation,
        stale,
        scan_price=100,
        atr_value=2,
        regime_gate_passes=True,
        now=now + timedelta(hours=4),
    )
    assert result.state is RecommendationState.CANCELLED


def test_forced_restart_demo_has_no_logical_duplicates(tmp_path) -> None:
    counts = run(tmp_path / "demo.sqlite")
    assert counts["sent"] == counts["receiver_unique"]


def test_expired_lease_is_reclaimed_and_event_id_is_channel_independent(
    tmp_path,
) -> None:
    now = datetime(2024, 1, 2, tzinfo=UTC)
    store = StateStore(tmp_path / "reclaim.sqlite")
    store.add(_recommendation(now), ["one", "two"])
    first = store.lease_outbox(limit=2, lease_seconds=60)
    assert len(first) == 2
    assert first[0].event.event_id == first[1].event.event_id
    assert first[0].idempotency_key != first[1].idempotency_key
    with store.connect() as connection:
        connection.execute(
            "UPDATE outbox SET lease_until=? WHERE status='leased'",
            ((now - timedelta(days=1)).isoformat(),),
        )
    replay = store.lease_outbox(limit=2)
    assert len(replay) == 2
    assert all(record.attempts == 2 for record in replay)


def test_monitor_waits_for_trigger_then_honors_expiry_fallback() -> None:
    now = datetime(2024, 1, 2, tzinfo=UTC)
    plan = EntryPlan(
        "pullback_with_expiry",
        OrderType.LOC,
        Direction.LONG,
        TimingVerdict.WAIT_FOR_TRIGGER,
        now + timedelta(hours=1),
        "wait",
        expiry_at=now + timedelta(hours=2),
        expiry_action="enter_at_next_eligible_close",
        validity_end=now + timedelta(days=1),
        data_timestamp=now,
    )
    recommendation = Recommendation(
        "trigger-1",
        AssetKey("MSFT"),
        Direction.LONG,
        80,
        0.01,
        (0, 0.02),
        3,
        plan,
        ("edge",),
        now,
    )
    before = now + timedelta(minutes=30)
    quote = Quote(AssetKey("MSFT"), 100, before, before, "test")
    waiting = revalidate_pending(
        recommendation,
        quote,
        scan_price=100,
        atr_value=2,
        regime_gate_passes=True,
        trigger_passes=False,
        now=before,
    )
    assert waiting.state is RecommendationState.WAITING
    expiry = now + timedelta(hours=2)
    quote = Quote(AssetKey("MSFT"), 100, expiry, expiry, "test")
    fallback = revalidate_pending(
        recommendation,
        quote,
        scan_price=100,
        atr_value=2,
        regime_gate_passes=True,
        trigger_passes=False,
        now=expiry,
    )
    assert fallback.state is RecommendationState.CONFIRMED


def test_scanner_keeps_failed_model_candidates_in_skip_audit() -> None:
    now = datetime(2024, 1, 2, tzinfo=UTC)

    def candidate(symbol: str, promoted: bool) -> ScoredCandidate:
        plan = EntryPlan(
            "immediate_at_close",
            OrderType.MOC,
            Direction.LONG,
            TimingVerdict.ACT_NOW,
            now + timedelta(hours=7),
            "base",
            data_timestamp=now,
        )
        return ScoredCandidate(
            AssetKey(symbol),
            Direction.LONG,
            80,
            0.01,
            (0, 0.02),
            3,
            plan,
            ("edge",),
            True,
            promoted,
        )

    rankings = scan((candidate("AAPL", True), candidate("MSFT", False)), as_of=now)
    assert [item.asset.symbol for item in rankings.longs] == ["AAPL"]
    assert [item.asset.symbol for item in rankings.skipped] == ["MSFT"]
    assert rankings.skipped[0].entry_plan.verdict is TimingVerdict.SKIP


def test_short_paper_ledger_is_direction_adjusted(tmp_path) -> None:
    now = datetime(2024, 1, 2, tzinfo=UTC)
    plan = EntryPlan(
        "immediate_at_close",
        OrderType.MOC,
        Direction.SHORT,
        TimingVerdict.ACT_NOW,
        now + timedelta(hours=1),
        "short",
        data_timestamp=now,
    )
    recommendation = Recommendation(
        "short-1",
        AssetKey("ABC"),
        Direction.SHORT,
        80,
        0.01,
        (0, 0.02),
        3,
        plan,
        ("edge",),
        now,
    )
    store = StateStore(tmp_path / "short.sqlite")
    store.add(recommendation, ["console"])
    store.transition("short-1", RecommendationState.CONFIRMED, "ok", "ok", ["console"])
    store.transition(
        "short-1", RecommendationState.ENTERED, "fill", "fill", ["console"]
    )
    store.record_entry("short-1", now + timedelta(hours=1), 100, 10, 1)
    store.transition("short-1", RecommendationState.EXITED, "exit", "exit", ["console"])
    assert store.record_exit("short-1", now + timedelta(days=1), 90, 1) == 98
