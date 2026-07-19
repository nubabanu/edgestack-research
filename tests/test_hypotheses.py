from __future__ import annotations

import numpy as np

from edgestack.hypotheses.controls import control_specs, matched_random_signal, turnover
from edgestack.hypotheses.grid import (
    DEFAULT_PREDICATES,
    EXTENDED_PREDICATES,
    GridConfig,
    conditional_combination_hypotheses,
    cross_sectional_hypotheses,
    enumerate_hypotheses,
    hypothesis_count,
)
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


def test_extended_families_are_optin_and_leave_original_ids_untouched() -> None:
    base = cross_sectional_hypotheses()
    extended = cross_sectional_hypotheses(extended=True)
    assert len(base) == 8
    assert len(extended) == 16
    assert [item.hypothesis_id for item in extended[:8]] == [
        item.hypothesis_id for item in base
    ]
    new_families = {item.family for item in extended[8:]}
    assert new_families == {
        "amihud_illiquidity",
        "max_lottery",
        "overnight_intraday_gap",
        "etf_relative_reversal",
    }


def test_extended_calendar_predicates_expand_the_declared_count() -> None:
    base = GridConfig()
    extended = GridConfig(
        predicate_levels={**DEFAULT_PREDICATES, **EXTENDED_PREDICATES}
    )
    base_count = hypothesis_count(base)
    extended_count = hypothesis_count(extended)
    assert extended_count > base_count
    quarter_specs = [
        item
        for item in enumerate_hypotheses(extended)
        if item.predicates.get("quarter_end") == "WINDOW"
        and len(item.predicates) == 1
    ]
    assert quarter_specs
    assert all(item.rationale.value == "flow-based" for item in quarter_specs)


def test_combination_candidates_are_declared_distinct_trials() -> None:
    combos = conditional_combination_hypotheses()
    # 2 signal families x 4 calendar gates x 2 directions.
    assert len(combos) == 16
    assert len({spec.hypothesis_id for spec in combos}) == 16
    plain = {spec.hypothesis_id for spec in cross_sectional_hypotheses(extended=True)}
    assert plain.isdisjoint({spec.hypothesis_id for spec in combos})
    for spec in combos:
        assert spec.predicates, "a combination must declare its gate"
        assert spec.parameters["combination"] == "CALENDAR_GATED"


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
