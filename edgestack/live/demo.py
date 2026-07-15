"""Accelerated deterministic restart demo for the notification lifecycle."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from edgestack.live.notify import InMemoryIdempotentChannel, dispatch_outbox
from edgestack.live.state import StateStore
from edgestack.models import (
    AssetKey,
    Direction,
    EntryPlan,
    OrderType,
    Recommendation,
    RecommendationState,
    TimingVerdict,
)


async def run_live_demo(database: str | Path) -> dict[str, int]:
    """Exercise proposal, restart, update, confirmation, entry, and exit."""

    path = Path(database)
    now = datetime(2024, 1, 2, 13, 30, tzinfo=UTC)
    plan = EntryPlan(
        method="pullback_with_expiry",
        order_type=OrderType.LOC,
        direction=Direction.LONG,
        verdict=TimingVerdict.WAIT_FOR_TRIGGER,
        earliest_execution=now + timedelta(hours=3),
        rationale="recorded demo",
        trigger="RSI(2)<10",
        expiry_at=now + timedelta(days=5),
        expiry_action="enter_at_next_eligible_close",
        data_timestamp=now,
    )
    recommendation = Recommendation(
        recommendation_id="demo-restart-0001",
        asset=AssetKey("MSFT"),
        direction=Direction.LONG,
        confidence=80,
        expected_net_return=0.01,
        expected_return_ci=(0.001, 0.019),
        holding_period=3,
        entry_plan=plan,
        driving_edges=("demo-edge",),
        created_at=now,
    )
    store = StateStore(path)
    store.add(recommendation, ["memory"])
    receiver = InMemoryIdempotentChannel()
    first_lease = store.lease_outbox(limit=1)
    if first_lease:
        await receiver.send(first_lease[0].event, first_lease[0].idempotency_key)
        # Simulated crash before acknowledgement. Expire lease to force replay.
        with store.connect() as connection:
            connection.execute(
                "UPDATE outbox SET status='retry', lease_until=NULL WHERE outbox_id=?",
                (first_lease[0].outbox_id,),
            )
    restarted = StateStore(path)
    await dispatch_outbox(restarted, {"memory": receiver})
    restarted.transition(
        recommendation.recommendation_id,
        RecommendationState.WAITING,
        "timer queued",
        "WAIT ~3 hours then re-check.",
        ["memory"],
        occurred_at=now + timedelta(minutes=1),
    )
    restarted.transition(
        recommendation.recommendation_id,
        RecommendationState.UPDATED,
        "limit adjusted",
        "Plan ADJUSTED after revalidation.",
        ["memory"],
        occurred_at=now + timedelta(hours=3),
    )
    restarted.transition(
        recommendation.recommendation_id,
        RecommendationState.CONFIRMED,
        "conditions hold",
        "Recommendation HOLDS.",
        ["memory"],
        occurred_at=now + timedelta(hours=3, minutes=15),
    )
    restarted.transition(
        recommendation.recommendation_id,
        RecommendationState.ENTERED,
        "paper fill",
        "Paper position ENTERED.",
        ["memory"],
        occurred_at=now + timedelta(hours=7),
    )
    restarted.record_entry(
        recommendation.recommendation_id, now + timedelta(hours=7), 100.0, 10, 1.0
    )
    restarted.transition(
        recommendation.recommendation_id,
        RecommendationState.EXITED,
        "time exit",
        "Paper position EXITED at its time stop.",
        ["memory"],
        occurred_at=now + timedelta(days=3),
    )
    restarted.record_exit(
        recommendation.recommendation_id, now + timedelta(days=3), 102.0, 1.0
    )
    await dispatch_outbox(restarted, {"memory": receiver})
    counts = restarted.logical_event_counts()
    counts["receiver_unique"] = len(receiver.received)
    return counts


def run(database: str | Path) -> dict[str, int]:
    """Synchronous CLI wrapper."""

    return asyncio.run(run_live_demo(database))
