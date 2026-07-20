from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.live.daily_job import entry_signal_line
from edgestack.live.preclose import peers_healing, preclose_lines

_ROW = {"session": "2026-07-29", "weekday": "WED", "expected_daily_bp": 23.8}


def test_alert_fires_above_threshold_with_full_context() -> None:
    line = entry_signal_line("EPAM", _ROW, "BELOW_MA200", True, None)
    assert line is not None
    assert "EPAM" in line and "2026-07-29" in line
    assert "~24bp" in line
    assert "regime=BELOW_MA200" in line and "dip=yes" in line
    assert line.startswith("ENTRY_SIGNAL (diagnostic):")


def test_alert_silent_below_threshold() -> None:
    weak = {**_ROW, "expected_daily_bp": 9.0}
    assert entry_signal_line("EPAM", weak, "ABOVE_MA200", False, None) is None


def test_alert_suppressed_inside_estimated_earnings_window() -> None:
    window = ("2026-07-25", "2026-08-05")
    assert entry_signal_line("CTSH", _ROW, "BELOW_MA200", False, window) is None
    outside = ("2026-08-10", "2026-08-20")
    assert entry_signal_line("CTSH", _ROW, "BELOW_MA200", False, outside) is not None


def test_alert_reports_unknown_dip_state() -> None:
    line = entry_signal_line("SPY", _ROW, "ABOVE_MA200", None, None)
    assert line is not None and "dip=unknown" in line


def test_alert_appends_peers_note() -> None:
    line = entry_signal_line(
        "ACN", _ROW, "BELOW_MA200", False, None, peers_note="[peers healing: 2/4]"
    )
    assert line is not None and line.endswith("[peers healing: 2/4]")


def _peer_frame(daily_return: float, sessions: int = 40) -> pd.DataFrame:
    prices = 100.0 * np.cumprod(np.full(sessions, 1.0 + daily_return))
    return pd.DataFrame({"close": prices, "adjusted_close": prices})


def test_peers_healing_counts_recovering_names_only() -> None:
    frames = {
        "ACN": _peer_frame(0.01),  # rising: above 20d MA, positive 5d
        "CTSH": _peer_frame(-0.01),  # falling: neither
        "EPAM": _peer_frame(0.0, sessions=5),  # too short: excluded from total
    }
    healing, total = peers_healing(frames, ("ACN", "CTSH", "EPAM", "IBM"))
    assert (healing, total) == (1, 2)


def test_preclose_lines_apply_threshold_earnings_and_peers() -> None:
    rows = [
        {"symbol": "EPAM", "session": "2026-07-29", "expected_daily_bp": 23.8},
        {"symbol": "CTSH", "session": "2026-07-29", "expected_daily_bp": 16.0},
        {"symbol": "SPY", "session": "2026-07-29", "expected_daily_bp": 9.0},
    ]
    lines = preclose_lines(
        rows,
        earnings_windows={"CTSH": ("2026-07-22", "2026-08-05")},
        quotes={"EPAM": "$61.20"},
        peers_note="[peers healing: 3/4]",
    )
    assert len(lines) == 1  # CTSH suppressed by earnings, SPY under threshold
    line = lines[0]
    assert "EPAM" in line and "~24bp" in line
    assert "[peers healing: 3/4]" in line
    assert "$61.20" in line and "15-min delayed" in line
    assert "15:45 ET" in line
