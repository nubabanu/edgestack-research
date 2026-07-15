from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from edgestack.config import EdgeStackConfig, load_config
from edgestack.models import (
    CausalDataView,
    Direction,
    HypothesisSpec,
    RationaleCategory,
    Session,
    ensure_fill_after_signal,
)


def test_hypothesis_id_is_order_independent() -> None:
    left = HypothesisSpec(
        "calendar",
        "Monday in January",
        {"weekday": "MON", "month": "JAN"},
        Direction.LONG,
        Session.CLOSE_TO_CLOSE,
        1,
        RationaleCategory.BEHAVIORAL,
    )
    right = HypothesisSpec(
        "calendar",
        "Monday in January",
        {"month": "JAN", "weekday": "MON"},
        Direction.LONG,
        Session.CLOSE_TO_CLOSE,
        1,
        RationaleCategory.BEHAVIORAL,
    )
    assert left.hypothesis_id == right.hypothesis_id


def test_causal_view_filters_and_rejects_future() -> None:
    now = datetime(2024, 1, 2, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "available_at": [now - timedelta(days=1), now + timedelta(days=1)],
            "value": [1, 2],
        }
    )
    view = CausalDataView.as_of(frame, now)
    assert view.frame["value"].tolist() == [1]
    with pytest.raises(ValueError, match="future data"):
        CausalDataView(frame, now)


def test_fill_must_follow_signal() -> None:
    signal = datetime(2024, 1, 2, tzinfo=UTC)
    ensure_fill_after_signal(signal, signal + timedelta(seconds=1))
    with pytest.raises(ValueError):
        ensure_fill_after_signal(signal, signal)


def test_yaml_config_and_cross_validation() -> None:
    assert load_config("configs/smoke.yaml").profile == "smoke"
    with pytest.raises(ValueError, match="embargo"):
        EdgeStackConfig.model_validate({"validation": {"embargo_sessions": 5}})
