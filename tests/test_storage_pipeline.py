from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from edgestack.models import GateResult, GateStatus, HoldoutFreezeManifest
from edgestack.pipeline.holdout import HoldoutGuard
from edgestack.provenance import source_tree_sha256
from edgestack.storage.artifacts import ArtifactStore
from edgestack.storage.catalog import Catalog


def test_content_addressed_raw_is_idempotent(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    first_hash, first_path = store.put_raw(b"same", ".csv")
    second_hash, second_path = store.put_raw(b"same", ".csv")
    assert first_hash == second_hash
    assert first_path == second_path
    assert first_path.read_bytes() == b"same"


def test_named_artifacts_are_idempotent_but_never_overwritten(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    path = store.write_json("evidence/result.json", {"value": 1})
    assert store.write_json("evidence/result.json", {"value": 1}) == path
    with pytest.raises(RuntimeError, match="immutable artifact differs"):
        store.write_json("evidence/result.json", {"value": 2})

    frame = pd.DataFrame({"value": [1.0, 2.0]})
    parquet = store.write_parquet("evidence/result.parquet", frame)
    assert store.write_parquet("evidence/result.parquet", frame) == parquet
    with pytest.raises(RuntimeError, match="immutable artifact differs"):
        store.write_parquet("evidence/result.parquet", pd.DataFrame({"value": [3.0]}))


def test_source_tree_hash_includes_package_data_code_only(tmp_path) -> None:
    package_data = tmp_path / "edgestack" / "data"
    package_data.mkdir(parents=True)
    provider = package_data / "provider.py"
    provider.write_text("VERSION = 1\n", encoding="utf-8")
    runtime_data = tmp_path / "data"
    runtime_data.mkdir()
    cache = runtime_data / "bars.parquet"
    cache.write_bytes(b"first")

    baseline = source_tree_sha256(tmp_path)
    cache.write_bytes(b"second")
    assert source_tree_sha256(tmp_path) == baseline

    provider.write_text("VERSION = 2\n", encoding="utf-8")
    assert source_tree_sha256(tmp_path) != baseline


def test_gate_prerequisites_and_single_holdout_access(tmp_path) -> None:
    catalog = Catalog(tmp_path / "catalog.sqlite")
    catalog.create_campaign("c1", {"id": "c1"})
    catalog.record_gate(
        GateResult("c1", "data", GateStatus.PASS, datetime.now(UTC), "ok", {})
    )
    catalog.require_passed("c1", ["data"])
    with pytest.raises(RuntimeError, match="prerequisites"):
        catalog.require_passed("c1", ["replication"])
    freeze = HoldoutFreezeManifest(
        campaign_id="c1",
        freeze_id="f1",
        frozen_at=datetime.now(UTC),
        edge_ids=(),
        specs_sha256="specs",
        stack_sha256="stack",
        overlay_sha256="overlays",
        cost_sha256="costs",
        config_sha256="config",
        bars_sha256="bars",
        universe_sha256="universe",
        data_manifest_sha256="data-manifest",
        source_tree_sha256="source",
        lock_sha256="lock",
        model_mapping_sha256="mapping",
        data_snapshot_id="data",
    )
    guard = HoldoutGuard(catalog)
    with guard.authorize(freeze):
        pass
    guard.complete("c1", "result")
    with pytest.raises(RuntimeError, match="already"), guard.authorize(freeze):
        pass


def test_smoke_override_gates_lists_only_mechanically_passed_phases(tmp_path) -> None:
    catalog = Catalog(tmp_path / "catalog.sqlite")
    catalog.create_campaign("c1", {"id": "c1"})
    now = datetime.now(UTC)
    catalog.record_gate(
        GateResult(
            "c1",
            "discovery",
            GateStatus.PASS,
            now,
            "smoke override",
            {"smoke_mechanical_override": True},
        )
    )
    catalog.record_gate(
        GateResult(
            "c1",
            "validation",
            GateStatus.PASS,
            now,
            "empirical pass",
            {"smoke_mechanical_override": False},
        )
    )
    catalog.record_gate(
        GateResult("c1", "data", GateStatus.PASS, now, "no marker", {})
    )
    overridden = catalog.smoke_override_gates("c1")
    assert [gate.phase for gate in overridden] == ["discovery"]
    assert [gate.phase for gate in catalog.smoke_override_gates()] == ["discovery"]


def test_holdout_completion_is_idempotent_for_same_result_only(tmp_path) -> None:
    catalog = Catalog(tmp_path / "catalog.sqlite")
    catalog.create_campaign("c1", {"id": "c1"})
    catalog.begin_holdout_access("c1", "freeze-1")
    assert catalog.holdout_access("c1") is not None
    catalog.complete_holdout_access("c1", "result-1")
    catalog.complete_holdout_access("c1", "result-1")
    access = catalog.holdout_access("c1")
    assert access is not None
    assert access.result_sha256 == "result-1"
    with pytest.raises(RuntimeError, match="already sealed"):
        catalog.complete_holdout_access("c1", "result-2")
