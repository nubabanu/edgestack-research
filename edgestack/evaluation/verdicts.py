"""Deterministic WORKS/WEAK/DEAD/FALSE_POSITIVE classification."""

from __future__ import annotations

from dataclasses import dataclass

from edgestack.models import (
    DecayClass,
    EvidenceBundle,
    ExecutionStatus,
    Verdict,
    VerdictRecord,
)


@dataclass(frozen=True, slots=True)
class VerdictInputs:
    """Non-numerical gates supplementing an evidence bundle."""

    bh_pass: bool
    deflated_sharpe_pass: bool
    spa_pass: bool
    net_cost_pass: bool
    event_confirmation_pass: bool
    regime_currently_active: bool = True
    holdout_opened: bool = False
    is_placebo: bool = False


def classify_verdict(
    hypothesis_id: str,
    evidence: EvidenceBundle | None,
    gates: VerdictInputs,
    *,
    execution_status: ExecutionStatus = ExecutionStatus.TESTED,
    decay: DecayClass = DecayClass.INSUFFICIENT,
    bias_tier: str = "SURVIVORSHIP_BIASED",
) -> VerdictRecord:
    """Apply frozen verdict precedence without discretionary overrides."""

    provisional = not gates.holdout_opened
    if execution_status in {ExecutionStatus.INVALID, ExecutionStatus.DATA_UNAVAILABLE}:
        return VerdictRecord(
            hypothesis_id,
            execution_status,
            None,
            decay,
            (execution_status.value.lower().replace("_", " "),),
            evidence,
            provisional,
            bias_tier,
        )
    if evidence is None:
        raise ValueError("evaluated hypotheses require an EvidenceBundle")
    if execution_status is ExecutionStatus.UNDERPOWERED:
        return VerdictRecord(
            hypothesis_id,
            execution_status,
            Verdict.WEAK,
            decay,
            ("fewer than 100 independent date observations",),
            evidence,
            provisional,
            bias_tier,
        )

    false_positive_reasons: list[str] = []
    if evidence.hac_t <= 3.0:
        false_positive_reasons.append("HAC t-stat did not exceed 3")
    if not gates.bh_pass:
        false_positive_reasons.append("failed global BH FDR")
    if not gates.deflated_sharpe_pass or evidence.deflated_sharpe_probability <= 0.95:
        false_positive_reasons.append("insignificant deflated Sharpe")
    if not gates.spa_pass:
        false_positive_reasons.append("failed Hansen SPA")
    if gates.is_placebo:
        false_positive_reasons.append("placebo/control hypothesis")
    if false_positive_reasons:
        return VerdictRecord(
            hypothesis_id,
            execution_status,
            Verdict.FALSE_POSITIVE,
            decay,
            tuple(false_positive_reasons),
            evidence,
            provisional,
            bias_tier,
        )

    if decay is DecayClass.DEAD:
        return VerdictRecord(
            hypothesis_id,
            execution_status,
            Verdict.DEAD,
            decay,
            ("historically significant but failed frozen recent-window decay rule",),
            evidence,
            provisional,
            bias_tier,
        )

    weak_reasons: list[str] = []
    if evidence.net_mean <= 0:
        weak_reasons.append("directed net mean is not positive after costs")
    if not gates.net_cost_pass:
        weak_reasons.append("gross evidence did not survive baseline costs")
    if evidence.oos_t is None or evidence.oos_t <= 2.0:
        weak_reasons.append("stitched OOS HAC t-stat did not exceed 2")
    if evidence.oos_positive_fraction is None or evidence.oos_positive_fraction <= 0.5:
        weak_reasons.append("not positive in a majority of OOS windows")
    if evidence.stability_score is None or evidence.stability_score < 0.75:
        weak_reasons.append("sub-period stability below 75%")
    if evidence.pbo is not None and evidence.pbo >= 0.20:
        weak_reasons.append("PBO is at least 20%")
    if not gates.event_confirmation_pass:
        weak_reasons.append("independent event-driven confirmation disagreed")
    if decay not in {DecayClass.STABLE, DecayClass.REGIME_DEPENDENT}:
        weak_reasons.append("decay classification is not deployable")
    if decay is DecayClass.REGIME_DEPENDENT and not gates.regime_currently_active:
        weak_reasons.append("validated regime is not currently active")
    if gates.holdout_opened and (
        evidence.holdout_mean is None or evidence.holdout_mean <= 0
    ):
        weak_reasons.append("final holdout net mean is not positive")
    if weak_reasons:
        return VerdictRecord(
            hypothesis_id,
            execution_status,
            Verdict.WEAK,
            decay,
            tuple(weak_reasons),
            evidence,
            provisional,
            bias_tier,
        )
    return VerdictRecord(
        hypothesis_id,
        execution_status,
        Verdict.WORKS,
        decay,
        ("passed all frozen research gates",),
        evidence,
        provisional,
        bias_tier,
    )
