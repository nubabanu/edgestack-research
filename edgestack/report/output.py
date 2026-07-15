"""Console, CSV, and HTML daily ranking output."""

from __future__ import annotations

import csv
import html
from collections.abc import Iterable
from pathlib import Path

from rich.console import Console
from rich.table import Table

from edgestack.disclaimer import DISCLAIMER
from edgestack.models import Recommendation
from edgestack.report.ranker import RankedRecommendations

_HEADERS = (
    "ranking_bucket",
    "recommendation_id",
    "ticker",
    "direction",
    "confidence",
    "holding_period",
    "timing_verdict",
    "entry_method",
    "order_type",
    "limit_price",
    "earliest_execution",
    "trigger",
    "trigger_value",
    "expiry",
    "expiry_action",
    "validity_end",
    "stop",
    "shares",
    "rationale",
    "data_timestamp",
    "expected_net_return",
    "ci_low",
    "ci_high",
    "driving_edges",
    "bias_tier",
    "borrow_verified",
    "disclaimer",
)


def _rows(
    recommendations: Iterable[Recommendation], ranking_bucket: str
) -> list[dict[str, object]]:
    return [
        {
            "ranking_bucket": ranking_bucket,
            "recommendation_id": item.recommendation_id,
            "ticker": item.asset.symbol,
            "direction": item.direction.value,
            "confidence": item.confidence,
            "holding_period": item.holding_period,
            "timing_verdict": item.entry_plan.verdict.value,
            "entry_method": item.entry_plan.method,
            "order_type": item.entry_plan.order_type.value,
            "limit_price": (
                item.entry_plan.limit_price
                if item.entry_plan.limit_price is not None
                else ""
            ),
            "earliest_execution": item.entry_plan.earliest_execution.isoformat(),
            "trigger": item.entry_plan.trigger or "",
            "trigger_value": (
                item.entry_plan.trigger_value
                if item.entry_plan.trigger_value is not None
                else ""
            ),
            "expiry": (
                item.entry_plan.expiry_at.isoformat()
                if item.entry_plan.expiry_at
                else ""
            ),
            "expiry_action": item.entry_plan.expiry_action or "",
            "validity_end": (
                item.entry_plan.validity_end.isoformat()
                if item.entry_plan.validity_end
                else ""
            ),
            "stop": (
                item.entry_plan.stop_price
                if item.entry_plan.stop_price is not None
                else ""
            ),
            "shares": (
                item.entry_plan.suggested_shares
                if item.entry_plan.suggested_shares is not None
                else ""
            ),
            "rationale": item.entry_plan.rationale,
            "data_timestamp": (
                item.entry_plan.data_timestamp.isoformat()
                if item.entry_plan.data_timestamp
                else ""
            ),
            "expected_net_return": item.expected_net_return,
            "ci_low": item.expected_return_ci[0],
            "ci_high": item.expected_return_ci[1],
            "driving_edges": "|".join(item.driving_edges),
            "bias_tier": item.bias_tier,
            "borrow_verified": item.borrow_verified,
            "disclaimer": DISCLAIMER,
        }
        for item in recommendations
    ]


def print_rankings(
    rankings: RankedRecommendations, console: Console | None = None
) -> None:
    """Print compact LONG/SHORT tables and the mandatory disclosure."""

    output = console or Console()
    output.print(f"[bold red]{DISCLAIMER}[/bold red]")
    for title, values in (
        ("TOP LONG", rankings.longs),
        ("TOP SHORT", rankings.shorts),
        ("SKIPPED AUDIT", rankings.skipped),
    ):
        table = Table(title=title)
        for column in ("Ticker", "Confidence", "Timing", "Expected net", "Plan"):
            table.add_column(column)
        for item in values:
            table.add_row(
                item.asset.symbol,
                str(item.confidence),
                item.entry_plan.verdict.value,
                f"{item.expected_net_return:.3%}",
                item.entry_plan.method,
            )
        if not values:
            table.add_row("—", "—", "—", "—", "No candidate passed every gate")
        output.print(table)


def write_rankings(
    rankings: RankedRecommendations, directory: str | Path
) -> tuple[Path, Path]:
    """Write machine-readable CSV and standalone HTML rankings."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        *_rows(rankings.longs, "TOP_LONG"),
        *_rows(rankings.shorts, "TOP_SHORT"),
        *_rows(rankings.skipped, "SKIPPED"),
    ]
    csv_path = root / "daily_rankings.csv"
    headers = list(_HEADERS)
    export_rows = rows or [
        {
            **dict.fromkeys(headers, ""),
            "ranking_bucket": "EMPTY",
            "timing_verdict": "SKIP",
            "rationale": "No candidate passed every frozen gate.",
            "disclaimer": DISCLAIMER,
        }
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(export_rows)
    table = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in headers)
        + "</tr>"
        for row in export_rows
    )
    header = "".join(f"<th>{html.escape(key)}</th>" for key in headers)
    html_path = root / "daily_rankings.html"
    html_path.write_text(
        f"<!doctype html><meta charset='utf-8'><h1>EdgeStack Daily Rankings</h1>"
        f"<p style='border:2px solid #900;padding:1rem'>{html.escape(DISCLAIMER)}</p>"
        f"<table border='1'><thead><tr>{header}</tr></thead><tbody>{table}</tbody></table>",
        encoding="utf-8",
    )
    return csv_path, html_path
