from __future__ import annotations

from edgestack.live.daily_job import entry_signal_line

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
