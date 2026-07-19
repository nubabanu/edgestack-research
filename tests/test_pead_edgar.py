from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from edgestack.data import edgar_earnings, intraday_collector
from edgestack.data.calendars import NYSECalendar
from edgestack.edges import pead_study


def test_frame_key_parses_calendar_quarters() -> None:
    assert pead_study._frame_key("CY2020Q1") == 2020 * 4
    assert pead_study._frame_key("CY2019Q4") == 2019 * 4 + 3
    assert pead_study._frame_key("CY2020") is None
    assert pead_study._frame_key("garbage") is None


def _eps_rows(symbol: str, values: dict[str, float]) -> pd.DataFrame:
    rows = []
    for frame, value in values.items():
        year, quarter = int(frame[2:6]), int(frame[-1])
        end = pd.Timestamp(year=year, month=quarter * 3, day=28)
        rows.append(
            {
                "symbol": symbol,
                "frame": frame,
                "end": end,
                "value": value,
                "filed": end + pd.Timedelta(days=40),
                "form": "10-Q",
            }
        )
    return pd.DataFrame(rows)


def test_compute_sue_drops_zero_variance_histories() -> None:
    # Perfectly steady seasonal growth: every trailing difference is equal,
    # the scale is zero, and no SUE may be fabricated from a 0/0.
    values = {}
    for index, (year, quarter) in enumerate(
        [(y, q) for y in (2018, 2019, 2020, 2021) for q in (1, 2, 3, 4)]
    ):
        values[f"CY{year}Q{quarter}"] = 1.0 + 0.1 * index
    sue = pead_study.compute_sue(_eps_rows("AAA", values))
    assert len(sue) == 0


def test_compute_sue_with_noisy_history_produces_finite_scores() -> None:
    rng = np.random.default_rng(7)
    values = {}
    for index, (year, quarter) in enumerate(
        [(y, q) for y in (2017, 2018, 2019, 2020, 2021) for q in (1, 2, 3, 4)]
    ):
        values[f"CY{year}Q{quarter}"] = 1.0 + 0.1 * index + rng.normal(0, 0.05)
    sue = pead_study.compute_sue(_eps_rows("AAA", values))
    assert len(sue) >= 4
    assert np.isfinite(sue["sue"]).all()
    # Each SUE quarter has at least 6 usable trailing seasonal differences.
    assert sue["quarter_end"].min() >= pd.Timestamp("2019-06-01")


def test_attach_availability_prefers_8k_and_falls_back_to_filed() -> None:
    sue = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "quarter_end": pd.Timestamp("2021-03-28"),
                "filed": pd.Timestamp("2021-05-07"),
                "sue": 2.0,
            },
            {
                "symbol": "AAA",
                "quarter_end": pd.Timestamp("2021-06-28"),
                "filed": pd.Timestamp("2021-08-06"),
                "sue": -1.0,
            },
        ]
    )
    announcements = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "acceptance": "2021-04-20T21:05:00.000Z",
                "form": "8-K",
                "items": "2.02,9.01",
            }
        ]
    )
    events = pead_study.attach_availability(sue, announcements)
    first = events.iloc[0]
    assert first["source"] == "EDGAR_8K_202_ACCEPTANCE"
    assert first["available"] == pd.Timestamp("2021-04-20")
    second = events.iloc[1]
    assert second["source"] == "XBRL_FILED_DATE"
    assert second["available"] == pd.Timestamp("2021-08-06")


def test_sue_panel_lands_strictly_after_availability() -> None:
    sessions = NYSECalendar().sessions("2021-04-15", "2021-04-30")
    events = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "quarter_end": pd.Timestamp("2021-03-28"),
                "filed": pd.Timestamp("2021-05-07"),
                "sue": 2.0,
                "available": pd.Timestamp("2021-04-20"),
                "source": "EDGAR_8K_202_ACCEPTANCE",
            }
        ]
    )
    panel = pead_study.build_sue_panel(events, sessions, ["AAA", "BBB"])
    placed = panel["AAA"].dropna()
    assert len(placed) == 1
    assert placed.index[0] > pd.Timestamp("2021-04-20")
    assert panel["BBB"].isna().all()


def test_announcement_parser_filters_on_item_202() -> None:
    payload = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q", "8-K"],
                "items": ["2.02,9.01", "", "5.02"],
                "acceptanceDateTime": [
                    "2024-01-25T21:30:00.000Z",
                    "2024-02-01T12:00:00.000Z",
                    "2024-03-01T12:00:00.000Z",
                ],
                "filingDate": ["2024-01-25", "2024-02-01", "2024-03-01"],
                "accessionNumber": ["a", "b", "c"],
            }
        }
    }
    rows = edgar_earnings._announcements_from_submissions("AAA", payload)
    assert len(rows) == 1
    assert rows[0]["accession"] == "a"


def test_eps_parser_keeps_only_frame_tagged_quarters() -> None:
    payload = {
        "units": {
            "USD/shares": [
                {
                    "frame": "CY2020Q1",
                    "end": "2020-03-31",
                    "val": 1.5,
                    "filed": "2020-05-01",
                    "form": "10-Q",
                    "fp": "Q1",
                },
                {
                    "end": "2020-03-31",
                    "val": 1.5,
                    "filed": "2020-05-01",
                    "form": "10-Q",
                    "fp": "Q1",
                },
                {
                    "frame": "CY2020",
                    "end": "2020-12-31",
                    "val": 6.0,
                    "filed": "2021-02-01",
                    "form": "10-K",
                    "fp": "FY",
                },
            ]
        }
    }
    rows = edgar_earnings._eps_from_concept("AAA", "EarningsPerShareDiluted", payload)
    assert len(rows) == 1
    assert rows[0]["frame"] == "CY2020Q1"


def test_intraday_collector_degrades_without_keys(
    monkeypatch: __import__("pytest").MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ALPACA_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    summary = intraday_collector.capture("2026-07-17", root=tmp_path)
    assert summary["status"] == "DATA_UNAVAILABLE"
    assert "Alpaca" in summary["reason"]
    assert not (tmp_path / "artifacts").exists()
