from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from edgestack.backtest.costs import CostAssumptions, CostModel
from edgestack.data.calendars import NYSECalendar
from edgestack.edges.turn_of_month import build_turn_of_month_episodes


def _synthetic_bars(start: str, end: str, daily_return: float = 0.001) -> pd.DataFrame:
    sessions = NYSECalendar().sessions(start, end)
    adjusted = 100.0 * np.cumprod(np.full(len(sessions), 1.0 + daily_return))
    return pd.DataFrame(
        {
            "session": sessions,
            "close": adjusted,
            "adjusted_close": adjusted,
            "volume": 10_000_000.0,
        }
    )


def test_exact_entry_exposure_and_exit_sessions() -> None:
    bars = _synthetic_bars("2024-01-01", "2024-04-30")
    model = CostModel(
        CostAssumptions(
            portfolio_capital=10_000.0,
            etf_full_spread_bps=0.0,
            base_slippage_bps=0.0,
            impact_coefficient_bps=0.0,
            turnover_penalty_bps=0.0,
        )
    )
    episodes = build_turn_of_month_episodes(bars, cost_model=model)
    january = episodes[0]
    assert january.entry_session == date(2024, 1, 30)
    assert january.first_exposure_session == date(2024, 1, 31)
    assert january.exit_session == date(2024, 2, 5)
    assert january.gross_return == pytest.approx((1.001**4) - 1.0)
    assert january.net_return == pytest.approx(january.gross_return)


def test_incomplete_boundary_month_is_not_mislabeled() -> None:
    bars = _synthetic_bars("2023-05-01", "2023-07-13")
    episodes = build_turn_of_month_episodes(bars)
    assert [episode.first_exposure_session for episode in episodes] == [
        date(2023, 5, 31),
        date(2023, 6, 30),
    ]
    assert all(episode.exit_session <= date(2023, 7, 13) for episode in episodes)


def test_roundtrip_charges_two_fills_and_two_turnover_penalties() -> None:
    bars = _synthetic_bars("2024-01-01", "2024-03-31", daily_return=0.0)
    model = CostModel(
        CostAssumptions(
            portfolio_capital=10_000.0,
            etf_full_spread_bps=1.0,
            base_slippage_bps=1.0,
            impact_coefficient_bps=0.0,
            turnover_penalty_bps=1.0,
        )
    )
    episode = build_turn_of_month_episodes(bars, cost_model=model)[0]
    assert episode.baseline_cost_bps == pytest.approx(5.0)
    assert episode.net_return == pytest.approx((1.0 - 2.5e-4) ** 2 - 1.0)
