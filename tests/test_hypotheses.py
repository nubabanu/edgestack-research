from __future__ import annotations

import numpy as np

from edgestack.hypotheses.controls import control_specs, matched_random_signal, turnover
from edgestack.hypotheses.grid import GridConfig, enumerate_hypotheses
from edgestack.models import HypothesisSpec


def test_grid_uses_shared_specs_and_stable_unique_ids() -> None:
    config = GridConfig(
        predicate_levels={"weekday": ("MON", "TUE"), "month": ("1", "2")},
        holding_periods=(1, 3),
    )
    first = enumerate_hypotheses(config)
    second = enumerate_hypotheses(config)
    assert first
    assert all(isinstance(item, HypothesisSpec) for item in first)
    assert [item.hypothesis_id for item in first] == [
        item.hypothesis_id for item in second
    ]
    assert len({item.hypothesis_id for item in first}) == len(first)
    assert {item.session.value for item in first} == {
        "close_to_close",
        "overnight",
        "intraday",
    }


def test_controls_are_deterministic_and_identity_distinct() -> None:
    spec = enumerate_hypotheses(GridConfig(predicate_levels={"weekday": ("MON",)}))[0]
    controls = control_specs(spec, campaign_seed=42)
    assert {item.placebo_kind for item in controls} == {
        "SHUFFLED_DATE",
        "MATCHED_RANDOM",
    }
    assert len({spec.hypothesis_id, *(item.hypothesis_id for item in controls)}) == 3
    assert controls == control_specs(spec, campaign_seed=42)


def test_random_control_matches_exposure_and_turnover() -> None:
    signal = np.array([[0.5, -0.5, 0.0], [0.0, 0.5, -0.5], [-0.5, 0.0, 0.5]])
    control = np.asarray(matched_random_signal(signal, seed=7))
    np.testing.assert_allclose(control.sum(axis=1), signal.sum(axis=1))
    np.testing.assert_allclose(np.abs(control).sum(axis=1), np.abs(signal).sum(axis=1))
    assert turnover(control) == turnover(signal)
