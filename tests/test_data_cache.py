from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

import edgestack.data.cache as cache_module
from edgestack.data.cache import ContentAddressedRawStore, DataCache
from edgestack.data.sources import RawPayload
from edgestack.models import AssetKey, Bar, BarRequest, SourceBatch


def test_cache_keeps_raw_and_adjusted_partitions_immutable(tmp_path) -> None:
    cache = DataCache(
        raw_root=tmp_path / "raw",
        canonical_root=tmp_path / "canonical",
        catalog_path=tmp_path / "catalog.sqlite",
    )
    fetched = datetime(2024, 1, 3, tzinfo=UTC)
    payload = RawPayload("test", AssetKey("ABC"), fetched, "text/csv", b"raw bytes")
    digest = cache.raw.store(payload)
    event = datetime(2024, 1, 2, 21, tzinfo=UTC)
    request = BarRequest(AssetKey("ABC"), date(2024, 1, 1), date(2024, 1, 3))
    batch = SourceBatch(
        "test",
        request,
        (
            Bar(
                request.asset,
                event,
                event + timedelta(minutes=15),
                100,
                102,
                99,
                100,
                1000,
                adjusted_close=50,
                dividend=1,
                split_factor=2,
                source="test",
            ),
        ),
        fetched,
        digest,
    )

    snapshot = cache.store_batch(batch)
    assert cache.store_batch(batch) == snapshot
    raw = cache.read_frame(snapshot.snapshot_id, representation="raw")
    adjusted = cache.read_frame(snapshot.snapshot_id, representation="adjusted")
    actions = cache.read_frame(snapshot.snapshot_id, representation="actions")

    assert raw.loc[0, "close"] == 100
    assert adjusted.loc[0, "close"] == 50
    assert adjusted.loc[0, "volume"] == 2000
    assert set(actions["action"]) == {"dividend", "split"}
    assert cache.load_batch(snapshot.snapshot_id).bars[0].adjusted_close == 50


def test_cache_requires_raw_payload_in_same_store(tmp_path) -> None:
    cache = DataCache(
        raw_root=tmp_path / "raw",
        canonical_root=tmp_path / "canonical",
        catalog_path=tmp_path / "catalog.sqlite",
    )
    event = datetime(2024, 1, 2, 21, tzinfo=UTC)
    request = BarRequest(AssetKey("ABC"), date(2024, 1, 1), date(2024, 1, 3))
    batch = SourceBatch(
        "test",
        request,
        (Bar(request.asset, event, event + timedelta(minutes=1), 1, 1, 1, 1, 1),),
        datetime(2024, 1, 3, tzinfo=UTC),
        "0" * 64,
    )
    with pytest.raises(FileNotFoundError):
        cache.store_batch(batch)


def test_atomic_directory_install_retries_transient_windows_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / ".snapshot.tmp"
    target = tmp_path / "snapshot"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    real_replace = cache_module.os.replace
    calls = 0

    def flaky_replace(left: Path, right: Path) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("transient scanner lock")
        real_replace(left, right)

    monkeypatch.setattr(cache_module.os, "replace", flaky_replace)
    monkeypatch.setattr(cache_module.time, "sleep", lambda _: None)

    cache_module._install_directory(source, target)

    assert calls == 3
    assert target.is_dir()
    assert not source.exists()


def test_identical_raw_bytes_keep_every_fetch_observation(tmp_path: Path) -> None:
    store = ContentAddressedRawStore(tmp_path / "raw")
    first = RawPayload(
        "provider-a",
        AssetKey("AAA"),
        datetime(2024, 1, 3, tzinfo=UTC),
        "text/plain",
        b"same provider response",
    )
    second = RawPayload(
        "provider-a",
        AssetKey("BBB"),
        datetime(2024, 1, 4, tzinfo=UTC),
        "text/plain",
        b"same provider response",
    )

    assert store.store(first) == store.store(second)

    observations = store.metadata_records(first.sha256)
    assert len(observations) == 2
    assert {item["asset"]["symbol"] for item in observations} == {"AAA", "BBB"}
    assert store.metadata(first.sha256)["asset"]["symbol"] == "BBB"
