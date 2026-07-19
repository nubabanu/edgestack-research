from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import NoReturn

import pandas as pd
import pytest

from edgestack.config import EdgeStackConfig, HoldoutGateConfig, PathsConfig
from edgestack.models import GateStatus, StackArtifact
from edgestack.pipeline.holdout import (
    evaluate_holdout_gate,
    promotion_decision,
    retro_ci_diagnostic,
)
from edgestack.pipeline.runner import CampaignRunner
from edgestack.provenance import canonical_sha256
from edgestack.scoring.stacking import StackResult


def _scored_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[CampaignRunner, dict[str, object]]:
    monkeypatch.setattr(
        "edgestack.pipeline.runner.source_tree_sha256", lambda _root: "source-tree"
    )
    config = EdgeStackConfig(paths=PathsConfig(root=tmp_path))
    runner = CampaignRunner.create(
        config,
        campaign_id="holdout-governance",
        as_of=date(2024, 12, 31),
    )
    runner._write_parquet(
        "canonical_bars",
        "data/bars.parquet",
        pd.DataFrame(
            {
                "symbol": ["SPY"],
                "session": [pd.Timestamp("2024-12-31")],
                "adjusted_close": [100.0],
            }
        ),
    )
    runner._write_parquet(
        "universe_memberships",
        "data/universe.parquet",
        pd.DataFrame({"symbol": ["SPY"], "sector": ["ETF"]}),
    )
    runner._write_json(
        "data_manifest", "data/manifest.json", {"snapshot_id": "snapshot-1"}
    )
    runner._write_json("hypothesis_registry", "discovery/specs.json", [])
    runner._write_json("provisional_records", "validation/records.json", [])
    runner._write_parquet(
        "validation_metrics",
        "validation/metrics.parquet",
        pd.DataFrame({"cost_sensitivity": pd.Series(dtype=str)}),
    )
    runner._write_json(
        "provisional_overlay_evidence",
        "reports/provisional/overlay_evidence.json",
        {"decisions": {}, "evidence": {}, "neighborhoods": {}},
    )
    empty_stack = StackResult(
        StackArtifact("empty", (), {}, {}, {}, 0.0, False),
        pd.Series(dtype=float, name="composite"),
    )
    monkeypatch.setattr(runner, "_build_provisional_stack", lambda: empty_stack)
    for phase in ("data", "replication", "discovery", "validation", "report"):
        runner.gates.record(phase, True, "fixture pass")
    score = runner.score()
    assert score.status is GateStatus.PASS
    freeze = runner._read_json("score/freeze.json")
    assert isinstance(freeze, dict)
    return runner, freeze


def _persist_empty_result(runner: CampaignRunner, freeze: dict[str, object]) -> str:
    result: dict[str, object] = {
        "campaign_id": runner.campaign_id,
        "freeze_id": freeze["freeze_id"],
        "evaluated_at": "2025-01-01T00:00:00+00:00",
        "holdout_start": runner.holdout_start.isoformat(),
        "holdout_end": runner.as_of.isoformat(),
        "edge_means": {},
        "edge_mean_cis": {},
        "evaluated_edge_ids": [],
        "composite_mean": None,
        "composite_mean_ci": None,
        "overlay_incremental_means": {},
        "edge_positive": True,
        "composite_positive": True,
        "overlays_nonnegative": True,
        "research_gate_pass": True,
        "promoted_for_paper_assistant": False,
        "empty_model_successful_outcome": True,
        "non_promotable_smoke": True,
        "disclaimer": runner._read_json("manifest.json")["disclaimer"],
    }
    result_sha = canonical_sha256(result)
    result["result_sha256"] = result_sha
    runner._write_json("holdout_result", "holdout/result.json", result)
    return result_sha


def test_sign_v1_gate_reproduces_original_sign_semantics() -> None:
    decision = evaluate_holdout_gate(
        evaluator_version="SIGN_V1",
        edge_means={"edge": 0.0001},
        edge_cis={"edge": (-0.01, 0.01)},
        has_edges=True,
        composite_mean=0.0001,
        composite_ci=(-0.01, 0.01),
        overlay_increments={"overlay": 0.0},
    )
    assert decision.research_gate_pass
    assert decision.edge_ci_lower_positive is None
    assert decision.composite_ci_lower_positive is None
    negative = evaluate_holdout_gate(
        evaluator_version="SIGN_V1",
        edge_means={"edge": -0.0001},
        edge_cis={},
        has_edges=True,
        composite_mean=0.5,
        composite_ci=None,
        overlay_increments={},
    )
    assert not negative.research_gate_pass


def test_ci_v2_gate_rejects_positive_mean_with_nonpositive_ci_lower() -> None:
    decision = evaluate_holdout_gate(
        evaluator_version="CI_V2",
        edge_means={"edge": 0.0001},
        edge_cis={"edge": (-0.01, 0.02)},
        has_edges=True,
        composite_mean=0.0001,
        composite_ci=(0.00001, 0.02),
        overlay_increments={},
    )
    assert decision.edge_positive
    assert decision.edge_ci_lower_positive is False
    assert not decision.research_gate_pass


def test_ci_v2_gate_fails_closed_when_an_edge_ci_is_missing() -> None:
    decision = evaluate_holdout_gate(
        evaluator_version="CI_V2",
        edge_means={"edge": 0.01},
        edge_cis={},
        has_edges=True,
        composite_mean=0.01,
        composite_ci=(0.001, 0.02),
        overlay_increments={},
    )
    assert decision.edge_ci_lower_positive is False
    assert not decision.research_gate_pass


def test_ci_v2_gate_passes_when_every_ci_lower_bound_is_positive() -> None:
    decision = evaluate_holdout_gate(
        evaluator_version="CI_V2",
        edge_means={"edge": 0.01},
        edge_cis={"edge": (0.001, 0.02)},
        has_edges=True,
        composite_mean=0.01,
        composite_ci=(0.001, 0.02),
        overlay_increments={"overlay": 0.0},
    )
    assert decision.research_gate_pass


def test_unknown_evaluator_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown holdout evaluator"):
        evaluate_holdout_gate(
            evaluator_version="SIGN_V3",
            edge_means={},
            edge_cis={},
            has_edges=False,
            composite_mean=None,
            composite_ci=None,
            overlay_increments={},
        )


def test_smoke_profile_can_never_promote() -> None:
    assert not promotion_decision(
        research_gate_pass=True, has_edges=True, profile="smoke"
    )
    assert promotion_decision(research_gate_pass=True, has_edges=True, profile="full")
    assert not promotion_decision(
        research_gate_pass=True, has_edges=False, profile="full"
    )


def test_retro_diagnostic_reads_only_the_sealed_document() -> None:
    report = retro_ci_diagnostic(
        {
            "campaign_id": "c1",
            "result_sha256": "sha",
            "research_gate_pass": True,
            "edge_means": {"edge": 0.0001},
            "edge_mean_cis": {"edge": [-0.01, 0.02]},
            "evaluated_edge_ids": ["edge"],
            "composite_mean": 0.0001,
            "composite_mean_ci": [-0.01, 0.02],
            "overlay_incremental_means": {},
        }
    )
    assert report["policy"] == "POST_HOLDOUT_DIAGNOSTIC_REPORT_ONLY"
    assert report["sealed_research_gate_pass"] is True
    assert report["ci_v2_would_pass"] is False


def test_finalize_refuses_an_evaluator_version_not_bound_at_freeze_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _ = _scored_runner(tmp_path, monkeypatch)
    runner.config = runner.config.model_copy(
        update={"holdout_gate": HoldoutGateConfig(evaluator_version="CI_V2")}
    )
    with pytest.raises(RuntimeError, match="evaluator version differs"):
        runner.finalize_holdout()
    assert runner.catalog.holdout_access(runner.campaign_id) is None


def test_ci_v2_freeze_binds_the_evaluator_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "edgestack.pipeline.runner.source_tree_sha256", lambda _root: "source-tree"
    )
    config = EdgeStackConfig(
        paths=PathsConfig(root=tmp_path),
        holdout_gate=HoldoutGateConfig(evaluator_version="CI_V2"),
    )
    runner = CampaignRunner.create(
        config, campaign_id="holdout-governance-v2", as_of=date(2024, 12, 31)
    )
    runner._write_json(
        "data_manifest", "data/manifest.json", {"snapshot_id": "snapshot-1"}
    )
    runner._write_json("hypothesis_registry", "discovery/specs.json", [])
    runner._write_parquet(
        "canonical_bars",
        "data/bars.parquet",
        pd.DataFrame(
            {
                "symbol": ["SPY"],
                "session": [pd.Timestamp("2024-12-31")],
                "adjusted_close": [100.0],
            }
        ),
    )
    runner._write_parquet(
        "universe_memberships",
        "data/universe.parquet",
        pd.DataFrame({"symbol": ["SPY"], "sector": ["ETF"]}),
    )
    runner._write_json(
        "provisional_overlay_evidence",
        "reports/provisional/overlay_evidence.json",
        {"decisions": {}, "evidence": {}, "neighborhoods": {}},
    )
    empty_stack = StackResult(
        StackArtifact("empty", (), {}, {}, {}, 0.0, False),
        pd.Series(dtype=float, name="composite"),
    )
    monkeypatch.setattr(runner, "_build_provisional_stack", lambda: empty_stack)
    for phase in ("data", "replication", "discovery", "validation", "report"):
        runner.gates.record(phase, True, "fixture pass")
    assert runner.score().status is GateStatus.PASS
    freeze = runner._read_json("score/freeze.json")
    assert isinstance(freeze, dict)
    assert freeze["holdout_evaluator_version"] == "CI_V2"


def test_frozen_artifact_tampering_is_rejected_before_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _ = _scored_runner(tmp_path, monkeypatch)
    (runner.campaign_root / "discovery/specs.json").write_text("[ ]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="frozen specs hash mismatch"):
        runner.finalize_holdout()

    assert runner.catalog.holdout_access(runner.campaign_id) is None


@pytest.mark.parametrize("already_sealed", [False, True])
def test_persisted_holdout_result_replays_without_reopening_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    already_sealed: bool,
) -> None:
    runner, freeze = _scored_runner(tmp_path, monkeypatch)
    freeze_id = str(freeze["freeze_id"])
    runner.catalog.begin_holdout_access(runner.campaign_id, freeze_id)
    result_sha = _persist_empty_result(runner, freeze)
    if already_sealed:
        runner.catalog.complete_holdout_access(runner.campaign_id, result_sha)

    def forbidden_data_access(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("replay reopened holdout inputs")

    monkeypatch.setattr(runner, "_load_data", forbidden_data_access)
    gate = runner.finalize_holdout()

    assert gate.status is GateStatus.PASS
    access = runner.catalog.holdout_access(runner.campaign_id)
    assert access is not None
    assert access.result_sha256 == result_sha
    assert (runner.campaign_root / "reports/final/edge_verdict_final.html").is_file()
    with pytest.raises(RuntimeError, match="already been evaluated"):
        runner.finalize_holdout()


def test_replay_rejects_frozen_model_tampering_without_reopening_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, freeze = _scored_runner(tmp_path, monkeypatch)
    runner.catalog.begin_holdout_access(runner.campaign_id, str(freeze["freeze_id"]))
    _persist_empty_result(runner, freeze)
    (runner.campaign_root / "score/stack.json").write_text("{}", encoding="utf-8")

    def forbidden_data_access(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("replay reopened holdout inputs")

    monkeypatch.setattr(runner, "_load_data", forbidden_data_access)
    with pytest.raises(RuntimeError, match="frozen stack hash mismatch"):
        runner.finalize_holdout()
