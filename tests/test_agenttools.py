from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import edgestack.agenttools as agenttools
from edgestack.agenttools import app
from edgestack.data.calendars import NYSECalendar

runner = CliRunner()


@pytest.fixture(scope="module")
def planted_bars() -> pd.DataFrame:
    """Nine years of bars with a Friday effect, enough for the advisor."""

    sessions = NYSECalendar().sessions("2015-01-02", "2023-12-29")
    rng = np.random.default_rng(20260719)
    returns = rng.normal(0.0002, 0.006, len(sessions))
    returns[sessions.dayofweek == 4] += 0.004
    prices = 100.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame(
        {
            "symbol": "ACN",
            "session": sessions,
            "open": prices,
            "close": prices,
            "adjusted_close": prices,
        }
    )


@pytest.fixture()
def offline_fetch(monkeypatch: pytest.MonkeyPatch, planted_bars: pd.DataFrame) -> None:
    monkeypatch.setattr(
        agenttools,
        "_fetch_bars",
        lambda symbol, years: (planted_bars, ("SYNTHETIC_FIXTURE",)),
    )


def _invoke(args: list[str]) -> dict:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_describe_lists_every_command() -> None:
    payload = _invoke(["describe"])
    assert set(payload["commands"]) == {
        "describe",
        "overview",
        "advise",
        "calendar",
        "compare",
        "tom",
        "oil",
        "gates",
    }
    assert "honesty_contract" in payload


def test_advise_projection_is_compact_and_keeps_the_stamps(
    offline_fetch: None, tmp_path: Path
) -> None:
    payload = _invoke(["advise", "ACN", "--root", str(tmp_path)])
    assert payload["status"] == "DIAGNOSTIC_NOT_A_VALIDATED_EDGE_NOT_AN_ORDER"
    assert payload["symbol"] == "ACN"
    assert "disclaimer" in payload
    assert "SYNTHETIC_FIXTURE" in payload["provenance_warnings"]
    assert len(payload["best_upcoming_sessions"]) <= 5
    for row in payload["best_upcoming_sessions"]:
        assert set(row) <= set(agenttools._ALIGNMENT_ROW_KEYS)
    for entry in payload["tailwinds"] + payload["headwinds"]:
        assert set(entry) == set(agenttools._CONDITION_BRIEF_KEYS)
    # No full evidence blobs survive the projection.
    assert "best_evidence" not in payload["timing"]["week"]
    assert len(json.dumps(payload)) < 20_000


def test_advise_rates_a_buy_date(offline_fetch: None, tmp_path: Path) -> None:
    payload = _invoke(
        ["advise", "ACN", "--buy-date", "2023-12-15", "--root", str(tmp_path)]
    )
    assessment = payload["buy_time_assessment"]
    assert assessment["overall_rating"] in {
        "POSITIVE",
        "NEGATIVE",
        "NEUTRAL_OR_MIXED",
    }
    assert len(assessment["choice_review"]["better_upcoming_sessions"]) <= 5


def test_calendar_matches_the_nightly_payload_shape_and_publishes(
    offline_fetch: None, tmp_path: Path
) -> None:
    payload = _invoke(
        ["calendar", "ACN", "--publish", "--rows", "7", "--root", str(tmp_path)]
    )
    nightly_keys = {
        "status",
        "symbol",
        "as_of_session",
        "policy",
        "anchors",
        "calendar",
        "validated_edges",
        "provenance_warnings",
        "disclaimer",
    }
    assert nightly_keys <= set(payload)
    assert len(payload["calendar"]) == 7
    published = Path(payload["published_to"])
    assert published.name == "tailwind-calendar-ACN.json"
    on_disk = json.loads(published.read_text(encoding="utf-8"))
    assert set(on_disk) == nightly_keys
    assert len(on_disk["calendar"]) > 7  # publish keeps the full scan


def test_compare_ranks_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch, planted_bars: pd.DataFrame, tmp_path: Path
) -> None:
    def fetch(symbol: str, years: int) -> tuple[pd.DataFrame, tuple[str, ...]]:
        if symbol == "BAD":
            raise ValueError("no bars for BAD")
        return planted_bars.assign(symbol=symbol), ()

    monkeypatch.setattr(agenttools, "_fetch_bars", fetch)
    payload = _invoke(["compare", "ACN,BAD", "--root", str(tmp_path)])
    assert [entry["symbol"] for entry in payload["ranked"]] == ["ACN"]
    assert "no bars for BAD" in payload["failures"]["BAD"]


def test_overview_degrades_gracefully_without_artifacts(tmp_path: Path) -> None:
    payload = _invoke(["overview", "--root", str(tmp_path)])
    assert payload["status"] == "DIAGNOSTIC_SNAPSHOT_FROM_PERSISTED_ARTIFACTS"
    for section in ("gates", "turn_of_month", "forward", "calendars", "oil"):
        assert payload[section]["status"] == "NOT_AVAILABLE"
        assert payload[section]["reason"]
    assert "disclaimer" in payload
    # Degrading must not create artifacts as a side effect.
    assert not (tmp_path / "artifacts").exists()


def test_gates_and_oil_report_missing_stores(tmp_path: Path) -> None:
    gates = _invoke(["gates", "--root", str(tmp_path)])
    assert gates["status"] == "NOT_AVAILABLE"
    oil = _invoke(["oil", "--root", str(tmp_path)])
    assert oil["status"] == "NOT_AVAILABLE"
    assert "oil-decision" in oil["reason"]
