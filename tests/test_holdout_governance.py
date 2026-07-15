from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import NoReturn

import pandas as pd
import pytest

from edgestack.config import EdgeStackConfig, PathsConfig
from edgestack.models import GateStatus, StackArtifact
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
