"""Single-use final-holdout ceremony and versioned promotion gate."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from edgestack.models import HoldoutFreezeManifest
from edgestack.storage.catalog import Catalog

HOLDOUT_EVALUATOR_VERSIONS = ("SIGN_V1", "CI_V2")


@dataclass(frozen=True, slots=True)
class HoldoutGateDecision:
    """Deterministic verdict of one holdout promotion evaluator."""

    evaluator_version: str
    edge_positive: bool
    composite_positive: bool
    overlays_nonnegative: bool
    edge_ci_lower_positive: bool | None
    composite_ci_lower_positive: bool | None
    research_gate_pass: bool


def evaluate_holdout_gate(
    *,
    evaluator_version: str,
    edge_means: Mapping[str, float],
    edge_cis: Mapping[str, tuple[float, float]],
    has_edges: bool,
    composite_mean: float | None,
    composite_ci: tuple[float, float] | None,
    overlay_increments: Mapping[str, float],
) -> HoldoutGateDecision:
    """Evaluate the frozen research gate under one declared evaluator version.

    ``SIGN_V1`` reproduces the original semantics exactly: every edge mean and
    the composite mean must be strictly positive and overlay increments
    non-negative. ``CI_V2`` additionally requires the stationary-bootstrap CI
    lower bound of every edge and of the composite to be strictly positive; a
    stream too short to produce a CI fails closed.
    """

    if evaluator_version not in HOLDOUT_EVALUATOR_VERSIONS:
        raise ValueError(f"unknown holdout evaluator version: {evaluator_version}")
    edge_positive = all(value > 0.0 for value in edge_means.values())
    composite_positive = not has_edges or (
        composite_mean is not None and composite_mean > 0.0
    )
    overlays_nonnegative = all(value >= 0.0 for value in overlay_increments.values())
    if evaluator_version == "SIGN_V1":
        return HoldoutGateDecision(
            evaluator_version=evaluator_version,
            edge_positive=edge_positive,
            composite_positive=composite_positive,
            overlays_nonnegative=overlays_nonnegative,
            edge_ci_lower_positive=None,
            composite_ci_lower_positive=None,
            research_gate_pass=(
                edge_positive and composite_positive and overlays_nonnegative
            ),
        )
    edge_ci_lower_positive = all(
        edge_id in edge_cis and edge_cis[edge_id][0] > 0.0 for edge_id in edge_means
    )
    composite_ci_lower_positive = not has_edges or (
        composite_ci is not None and composite_ci[0] > 0.0
    )
    return HoldoutGateDecision(
        evaluator_version=evaluator_version,
        edge_positive=edge_positive,
        composite_positive=composite_positive,
        overlays_nonnegative=overlays_nonnegative,
        edge_ci_lower_positive=edge_ci_lower_positive,
        composite_ci_lower_positive=composite_ci_lower_positive,
        research_gate_pass=(
            edge_positive
            and composite_positive
            and overlays_nonnegative
            and edge_ci_lower_positive
            and composite_ci_lower_positive
        ),
    )


def promotion_decision(
    *, research_gate_pass: bool, has_edges: bool, profile: str
) -> bool:
    """Return whether a holdout result may promote the paper assistant.

    Promotion always requires a passed research gate, a non-empty frozen edge
    stack, and the ``full`` empirical profile; a synthetic smoke campaign can
    never promote regardless of its gate outcomes.
    """

    return bool(research_gate_pass and has_edges and profile == "full")


def retro_ci_diagnostic(result: Mapping[str, Any]) -> dict[str, Any]:
    """Ask whether a sealed holdout result would have passed ``CI_V2``.

    Consumes only the persisted result document — never campaign data — so it
    respects the sealed-evidence replay policy. The answer is report-only and
    can never change the sealed verdict.
    """

    composite_mean = result.get("composite_mean")
    composite_ci = result.get("composite_mean_ci")
    decision = evaluate_holdout_gate(
        evaluator_version="CI_V2",
        edge_means={
            str(key): float(value)
            for key, value in dict(result.get("edge_means", {})).items()
        },
        edge_cis={
            str(key): (float(value[0]), float(value[1]))
            for key, value in dict(result.get("edge_mean_cis", {})).items()
        },
        has_edges=bool(result.get("evaluated_edge_ids")),
        composite_mean=(
            float(composite_mean) if composite_mean is not None else None
        ),
        composite_ci=(
            (float(composite_ci[0]), float(composite_ci[1]))
            if composite_ci is not None
            else None
        ),
        overlay_increments={
            str(key): float(value)
            for key, value in dict(
                result.get("overlay_incremental_means", {})
            ).items()
        },
    )
    return {
        "policy": "POST_HOLDOUT_DIAGNOSTIC_REPORT_ONLY",
        "campaign_id": result.get("campaign_id"),
        "source_result_sha256": result.get("result_sha256"),
        "sealed_research_gate_pass": result.get("research_gate_pass"),
        "ci_v2_would_pass": decision.research_gate_pass,
        "edge_ci_lower_positive": decision.edge_ci_lower_positive,
        "composite_ci_lower_positive": decision.composite_ci_lower_positive,
    }


class HoldoutGuard:
    """Ensure exactly one authorized analytical holdout evaluation per campaign."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    @contextmanager
    def authorize(self, freeze: HoldoutFreezeManifest) -> Iterator[None]:
        """Consume authorization before exposing holdout data to a callback."""

        self.catalog.begin_holdout_access(freeze.campaign_id, freeze.freeze_id)
        yield

    def complete(self, campaign_id: str, result_sha256: str) -> None:
        """Seal a completed holdout result."""

        self.catalog.complete_holdout_access(campaign_id, result_sha256)
