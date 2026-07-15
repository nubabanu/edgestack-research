from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from edgestack.entrytiming.indicators import atr, bollinger_pct_b, rsi, sma
from edgestack.entrytiming.interaction_tests import (
    OverlayEvidence,
    interaction_decision,
)
from edgestack.entrytiming.stops import atr_stop, vol_scaled_size
from edgestack.entrytiming.timers import immediate_at_close, pullback_with_expiry
from edgestack.models import Direction, TimingVerdict
from edgestack.scoring.shrinkage import empirical_bayes_shrinkage
from edgestack.scoring.stacking import build_stack, confidence_score


def test_indicators_use_trailing_data() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isnan(sma(values, 3).iloc[1])
    assert sma(values, 3).iloc[-1] == 4.0
    assert 0 <= rsi(values, 2).iloc[-1] <= 100
    assert bollinger_pct_b(values, 3).notna().sum() == 3
    assert atr(values + 1, values - 1, values, 2).iloc[-1] == 2.0


def test_shrinkage_and_equal_cluster_stack() -> None:
    estimates = {"a": 0.02, "b": -0.01, "c": 0.005}
    variances = {key: 0.00001 for key in estimates}
    result = empirical_bayes_shrinkage(estimates, variances)
    for key, raw in estimates.items():
        assert abs(result.shrunk[key]) <= abs(raw)
        assert np.sign(result.shrunk[key]) in {0, np.sign(raw)}
    streams = pd.DataFrame(
        {
            "a": [0.01, 0.02, -0.01] * 10,
            "b": [0.011, 0.019, -0.009] * 10,
            "c": [-0.01, 0.01, 0.0] * 10,
        }
    )
    stack = build_stack(streams, estimates, variances, 0.98)
    assert abs(sum(stack.artifact.weights.values()) - 1.0) < 1e-12
    assert len(stack.returns) == len(streams)
    assert confidence_score(0.95, 0.8) == 76


def test_timers_are_next_bar_and_risk_bounded() -> None:
    now = datetime(2024, 1, 2, tzinfo=UTC)
    plan = immediate_at_close(
        Direction.LONG, now + timedelta(hours=7), "baseline", data_timestamp=now
    )
    assert plan.earliest_execution > now
    pullback = pullback_with_expiry(
        Direction.LONG,
        now + timedelta(days=1),
        now,
        now + timedelta(days=5),
        rsi2_value=5,
        bollinger_value=0.5,
    )
    assert pullback.verdict is TimingVerdict.ACT_NOW
    assert atr_stop(100, 2, Direction.LONG) == 96
    assert vol_scaled_size(0.005, 4, 100_000, 100) == 100


def test_overlay_requires_adjacent_plateau() -> None:
    evidence = [
        OverlayEvidence(
            value,
            0.001,
            sharpe,
            3.5,
            True,
            0.98,
            2.5,
            0.7,
            ((0.5, 0.001), (1.0, 0.001), (2.0, 0.0008), (4.0, 0.0005)),
            True,
            0.1,
            True,
        )
        for value, sharpe in [(5, 1.0), (10, 0.9), (15, 0.1)]
    ]
    decision = interaction_decision(evidence)
    assert decision.enabled
    assert decision.selected_parameter in {5, 10}


def test_overlay_cannot_enable_without_cost_and_pbo_evidence() -> None:
    incomplete = [
        OverlayEvidence(value, 0.001, 1.0, 3.5, True, 0.98, 2.5, 0.7)
        for value in (5.0, 10.0)
    ]

    decision = interaction_decision(incomplete)

    assert not decision.enabled
    assert "gauntlet" in decision.reason


def test_pullback_expiry_cannot_outlive_base_edge() -> None:
    now = datetime(2024, 1, 2, tzinfo=UTC)
    with pytest.raises(ValueError, match="validity window"):
        pullback_with_expiry(
            Direction.LONG,
            now + timedelta(days=1),
            now,
            now + timedelta(days=6),
            rsi2_value=50,
            bollinger_value=0.5,
            validity_end=now + timedelta(days=5),
        )
    with pytest.raises(ValueError, match="ATR"):
        atr_stop(100, 0, Direction.LONG)
