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
        "leverage-check",
        "entry-check",
        "telegram-test",
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


@pytest.fixture(scope="module")
def ohlc_bars() -> pd.DataFrame:
    """Calm drift with one planted -30% intraday crash two years from the end."""

    sessions = NYSECalendar().sessions("2016-01-04", "2023-12-29")
    rng = np.random.default_rng(20260720)
    returns = rng.normal(0.0005, 0.008, len(sessions))
    crash = len(sessions) - 500
    returns[crash] = -0.30
    closes = 100.0 * np.cumprod(1.0 + returns)
    lows = closes * (1.0 - np.abs(rng.normal(0.004, 0.003, len(sessions))))
    lows[crash] = closes[crash] * 0.97
    highs = closes * (1.0 + np.abs(rng.normal(0.004, 0.003, len(sessions))))
    return pd.DataFrame(
        {
            "symbol": "MU",
            "session": sessions,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "adjusted_close": closes,
        }
    )


def test_leverage_check_reports_conditions_and_caveats(
    monkeypatch: pytest.MonkeyPatch, ohlc_bars: pd.DataFrame
) -> None:
    monkeypatch.setattr(
        agenttools, "_fetch_bars", lambda symbol, years: (ohlc_bars, ())
    )
    payload = _invoke(["leverage-check", "MU", "--leverage", "5"])
    assert payload["status"] == "DIAGNOSTIC_NOT_A_VALIDATED_EDGE_NOT_AN_ORDER"
    assert set(payload["conditions"]) == {"any_entry", "calm_regime", "calm_and_dip"}
    any_entry = payload["conditions"]["any_entry"]
    assert any_entry["n"] > 100
    # Positions holding through the planted -30% crash day are liquidated
    # at 5x (threshold 10%), so the rate must be visibly nonzero.
    assert any_entry["liquidated_fraction"] > 0.0
    assert any_entry["max_leverage_95pct_survival"] is not None
    assert "CLEAN_LIQUIDATION_ASSUMED_GAPS_MAKE_TAILS_WORSE" in payload["caveats"]
    assert "disclaimer" in payload


def test_leverage_check_flags_tiny_samples(
    monkeypatch: pytest.MonkeyPatch, ohlc_bars: pd.DataFrame
) -> None:
    short = ohlc_bars.tail(260).reset_index(drop=True)
    monkeypatch.setattr(agenttools, "_fetch_bars", lambda symbol, years: (short, ()))
    payload = _invoke(["leverage-check", "MU", "--horizon", "60"])
    # 260 sessions leave almost no complete windows after the SMA200 warmup.
    assert payload["conditions"]["any_entry"]["status"] == "TOO_FEW_ENTRIES"


def test_entry_state_reports_regime_and_dip(ohlc_bars: pd.DataFrame) -> None:
    state = agenttools.entry_state(ohlc_bars, "MU")
    assert set(state) >= {
        "calm_regime",
        "dip",
        "trend_above_ma200",
        "vol20_annualized",
        "rsi2",
        "ibs",
        "three_down_days",
    }
    # Planted three-down-days tail forces a dip verdict.
    forced = ohlc_bars.copy()
    closes = forced["close"].to_numpy().copy()
    closes[-3:] = closes[-4] * np.array([0.99, 0.98, 0.97])
    forced["close"] = closes
    forced["adjusted_close"] = closes
    state_dip = agenttools.entry_state(forced, "MU")
    assert state_dip["three_down_days"] is True
    assert state_dip["dip"] is True


def test_earnings_estimate_projects_quarterly_cadence(tmp_path: Path) -> None:
    out = tmp_path / "artifacts" / "earnings"
    out.mkdir(parents=True)
    dates = pd.date_range("2024-01-25", periods=8, freq="91D", tz="UTC")
    pd.DataFrame(
        {"symbol": "EPAM", "acceptance": [d.isoformat() for d in dates]}
    ).to_parquet(out / "announcements.parquet", index=False)
    estimate = agenttools._earnings_estimate("EPAM", tmp_path)
    assert estimate["status"] == "EARNINGS_WINDOW_ESTIMATED_NOT_CONFIRMED"
    assert estimate["median_gap_days"] == 91
    assert (
        estimate["window_start"] < estimate["estimated_next"] < estimate["window_end"]
    )
    missing = agenttools._earnings_estimate("ACN", tmp_path)
    assert missing["status"] == "NOT_AVAILABLE"


def test_telegram_test_skips_cleanly_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EDGESTACK_TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("EDGESTACK_TELEGRAM_CHAT", raising=False)
    payload = _invoke(["telegram-test"])
    assert payload["status"] == "SKIPPED_NO_CREDENTIALS"
    assert any("BotFather" in step for step in payload["setup"])


def test_earnings_estimate_includes_live_announcements(tmp_path: Path) -> None:
    out = tmp_path / "artifacts" / "earnings"
    out.mkdir(parents=True)
    # 21:30 UTC = 16:30/17:30 New York — same calendar date after conversion.
    sealed = pd.date_range("2024-01-25 21:30", periods=8, freq="91D", tz="UTC")
    pd.DataFrame(
        {"symbol": "EPAM", "acceptance": [d.isoformat() for d in sealed]}
    ).to_parquet(out / "announcements.parquet", index=False)
    fresh = sealed[-1] + pd.Timedelta(days=91)
    pd.DataFrame({"symbol": "EPAM", "acceptance": [fresh.isoformat()]}).to_parquet(
        out / "live-announcements.parquet", index=False
    )
    estimate = agenttools._earnings_estimate("EPAM", tmp_path)
    # The live print rolls the projection one quarter forward.
    assert estimate["last_announcement"] == str(fresh.date())


def test_gates_and_oil_report_missing_stores(tmp_path: Path) -> None:
    gates = _invoke(["gates", "--root", str(tmp_path)])
    assert gates["status"] == "NOT_AVAILABLE"
    oil = _invoke(["oil", "--root", str(tmp_path)])
    assert oil["status"] == "NOT_AVAILABLE"
    assert "oil-decision" in oil["reason"]
