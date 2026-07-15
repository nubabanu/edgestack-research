from __future__ import annotations

import pytest

from edgestack.evaluation.report import render_verdict_report
from edgestack.evaluation.verdicts import VerdictInputs, classify_verdict
from edgestack.models import DecayClass, EvidenceBundle, Verdict


def _evidence(**overrides: object) -> EvidenceBundle:
    values: dict[str, object] = {
        "hypothesis_id": "edge-1",
        "sample_size": 200,
        "gross_mean": 0.002,
        "net_mean": 0.001,
        "hac_t": 3.5,
        "p_value": 0.001,
        "sharpe": 1.0,
        "probabilistic_sharpe": 0.99,
        "deflated_sharpe_probability": 0.98,
        "hit_rate": 0.55,
        "max_drawdown": -0.10,
        "turnover": 1.0,
        "exposure": 0.5,
        "skew": 0.0,
        "kurtosis": 3.0,
        "mean_ci": (0.0001, 0.0019),
        "sharpe_ci": (0.2, 1.5),
        "oos_t": 2.5,
        "oos_positive_fraction": 0.7,
        "stability_score": 0.8,
        "pbo": 0.1,
        "holdout_mean": 0.0002,
        "confirmation_difference_bps": 0.5,
    }
    values.update(overrides)
    return EvidenceBundle(**values)  # type: ignore[arg-type]


def test_verdict_precedence_and_complete_report(tmp_path) -> None:
    gates = VerdictInputs(True, True, True, True, True, holdout_opened=True)
    works = classify_verdict("edge-1", _evidence(), gates, decay=DecayClass.STABLE)
    false = classify_verdict(
        "edge-2", _evidence(hac_t=1.0), gates, decay=DecayClass.STABLE
    )
    assert works.verdict is Verdict.WORKS
    assert false.verdict is Verdict.FALSE_POSITIVE
    html_path, csv_path = render_verdict_report(
        [works, false], {"N tested": 2}, tmp_path, final=True
    )
    assert "EdgeStack is for research" in html_path.read_text(encoding="utf-8")
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 3


def test_net_cost_failure_is_weak_not_false_positive() -> None:
    gates = VerdictInputs(True, True, True, False, True, holdout_opened=True)
    record = classify_verdict(
        "edge-1", _evidence(net_mean=-0.0001), gates, decay=DecayClass.STABLE
    )
    assert record.verdict is Verdict.WEAK
    assert any("after costs" in reason for reason in record.reasons)


def test_empty_report_keeps_disclaimer_and_final_rejects_provisional(tmp_path) -> None:
    html_path, csv_path = render_verdict_report(
        [], {"N tested": 0}, tmp_path, final=True
    )
    assert "NO_HYPOTHESES" in csv_path.read_text(encoding="utf-8")
    assert "EdgeStack is for research" in html_path.read_text(encoding="utf-8")

    gates = VerdictInputs(True, True, True, True, True, holdout_opened=False)
    provisional = classify_verdict(
        "edge-1", _evidence(), gates, decay=DecayClass.STABLE
    )
    with pytest.raises(ValueError, match="provisional"):
        render_verdict_report([provisional], {}, tmp_path, final=True)
