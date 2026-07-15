from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta

from edgestack.disclaimer import DISCLAIMER
from edgestack.models import (
    AssetKey,
    Direction,
    EntryPlan,
    OrderType,
    Recommendation,
    TimingVerdict,
)
from edgestack.report.output import write_rankings
from edgestack.report.ranker import rank_recommendations


def _recommendation(symbol: str, verdict: TimingVerdict) -> Recommendation:
    now = datetime(2024, 1, 2, tzinfo=UTC)
    plan = EntryPlan(
        "immediate_at_close",
        OrderType.MOC,
        Direction.LONG,
        verdict,
        now + timedelta(hours=7),
        "test rationale",
        data_timestamp=now,
    )
    return Recommendation(
        f"rec-{symbol}",
        AssetKey(symbol),
        Direction.LONG,
        80,
        0.01,
        (0, 0.02),
        3,
        plan,
        ("edge",),
        now,
    )


def test_empty_rankings_have_stable_schema_and_explicit_audit_row(tmp_path) -> None:
    rankings = rank_recommendations([])
    csv_path, html_path = write_rankings(rankings, tmp_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["ranking_bucket"] == "EMPTY"
    assert rows[0]["timing_verdict"] == "SKIP"
    assert rows[0]["disclaimer"] == DISCLAIMER
    assert "No candidate passed" in html_path.read_text(encoding="utf-8")


def test_skipped_recommendations_are_separate_audit_rows(tmp_path) -> None:
    rankings = rank_recommendations(
        (
            _recommendation("AAPL", TimingVerdict.ACT_NOW),
            _recommendation("MSFT", TimingVerdict.SKIP),
        )
    )
    csv_path, _ = write_rankings(rankings, tmp_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["ranking_bucket"] for row in rows} == {"TOP_LONG", "SKIPPED"}
