from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgestack.data.calendars import NYSECalendar
from edgestack.live.daily_job import (
    build_signal_payload,
    update_forward_ledger,
)
from edgestack.live.forward_ledger import ForwardLedger
from edgestack.mobile.service import MobileSnapshotService

AS_OF = date(2026, 7, 16)


def _panel(symbols: dict[str, float]) -> dict[str, pd.DataFrame]:
    """Synthetic 60-session panel; `symbols` maps name -> planted 5d move."""

    sessions = NYSECalendar().sessions("2026-04-20", "2026-07-16")
    panel = {}
    rng = np.random.default_rng(7)
    for symbol, planted in symbols.items():
        drift = rng.normal(0.0, 0.002, len(sessions))
        closes = 100.0 * np.cumprod(1.0 + drift)
        # Impose the planted trailing 5-session move exactly.
        closes[-1] = closes[-6] * (1.0 + planted)
        closes[-5:-1] = np.linspace(closes[-6], closes[-1], 6)[1:-1]
        panel[symbol] = pd.DataFrame(
            {
                "session": sessions,
                "open": closes,
                "high": closes * 1.01,
                "low": closes * 0.99,
                "close": closes,
                "adjusted_close": closes,
            }
        )
    return panel


@pytest.fixture(scope="module")
def payload() -> dict:
    moves = {f"S{index:02d}": 0.01 for index in range(10)}
    moves.update(
        {"AAA": -0.30, "BBB": -0.25, "CCC": -0.20, "DDD": -0.15, "EEE": -0.10,
         "FFF": -0.05}
    )
    return build_signal_payload(
        _panel(moves), tuple(moves), as_of=AS_OF, source_label="fixture"
    )


def test_signal_selects_the_five_most_oversold_names(payload: dict) -> None:
    assert [item["symbol"] for item in payload["candidates"]] == [
        "AAA", "BBB", "CCC", "DDD", "EEE",
    ]
    assert payload["candidates"][0]["trailing_5_session_return"] == pytest.approx(
        -0.30, abs=1e-6
    )
    assert payload["market_as_of"] == "2026-07-16_CLOSE"
    # Entry is the next NYSE session; exit five sessions later.
    assert payload["entry"]["session"] == "2026-07-17"
    assert payload["exit"]["session"] == "2026-07-24"


def test_signal_is_consumable_by_the_mobile_service(
    payload: dict, tmp_path: Path
) -> None:
    campaign = tmp_path / "campaigns" / "sealed-001"
    (campaign / "holdout").mkdir(parents=True)
    (campaign / "live").mkdir()
    (campaign / "holdout" / "result.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "holdout_pass": True,
                "second_evaluation": "FORBIDDEN_REPLAY_ONLY",
                "holdout_start": "2023-01-01",
                "holdout_end": "2026-01-01",
                "observations": 750,
                "expected_sessions": 750,
                "net_mean": 0.001,
                "benchmark_excess_mean": 0.0002,
                "terminal_net_wealth": 1.2,
                "terminal_benchmark_wealth": 1.1,
                "freeze_id": "freeze",
            }
        ),
        encoding="utf-8",
    )
    (campaign / "live" / "2026-07-16-signal.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    snapshot = MobileSnapshotService(tmp_path, campaign_id="sealed-001").load()
    assert snapshot.model_status == "PROMOTED"
    assert [item.symbol for item in snapshot.recommendations] == [
        "AAA", "BBB", "CCC", "DDD", "EEE",
    ]


def test_forward_ledger_is_append_only(tmp_path: Path, payload: dict) -> None:
    ledger = ForwardLedger(tmp_path / "ledger.sqlite")
    assert ledger.record_signal(payload) is True
    assert ledger.record_signal(payload) is False, "signals record exactly once"
    rec = payload["candidates"][0]["recommendation_id"]
    assert ledger.record_event(
        rec, symbol="AAA", event="FILL", session="2026-07-17", price=70.0
    )
    assert ledger.record_event(
        rec, symbol="AAA", event="MARK", session="2026-07-20", price=71.0
    )
    # Retroactive or duplicate marks are rejected.
    assert not ledger.record_event(
        rec, symbol="AAA", event="MARK", session="2026-07-20", price=72.0
    )
    assert not ledger.record_event(
        rec, symbol="AAA", event="MARK", session="2026-07-17", price=69.0
    )
    with pytest.raises(ValueError, match="FILL, MARK, or EXIT"):
        ledger.record_event(
            rec, symbol="AAA", event="EDIT", session="2026-07-21", price=70.0
        )


def test_ledger_flow_fills_marks_and_exits_from_closes(
    tmp_path: Path, payload: dict
) -> None:
    ledger = ForwardLedger(tmp_path / "ledger.sqlite")
    ledger.record_signal(payload)
    sessions = NYSECalendar().sessions("2026-07-16", "2026-07-31")
    frames = {}
    for item in payload["candidates"]:
        closes = np.linspace(100.0, 110.0, len(sessions))
        frames[item["symbol"]] = pd.DataFrame(
            {"session": sessions, "adjusted_close": closes}
        )
    # Day after entry: fill recorded, no exits yet.
    counts = update_forward_ledger(ledger, frames, as_of=date(2026, 7, 20))
    assert counts["fills"] == 5
    assert counts["exits"] == 0
    assert counts["marks"] == 5
    # At the exit session everything completes and the scorecard sees it.
    counts = update_forward_ledger(ledger, frames, as_of=date(2026, 7, 24))
    assert counts["exits"] == 5
    scorecard = ledger.scorecard()
    assert scorecard["completed_positions"] == 5
    assert scorecard["mean_position_return"] > 0
    assert scorecard["win_rate"] == 1.0
