"""Immutable raw-response and partitioned canonical market-data cache.

Raw HTTP bodies are content-addressed by SHA-256 and never overwritten.  A daily
batch is materialized into separate raw and adjusted Parquet representations, then
registered in a SQLite WAL catalog only after the complete snapshot directory has
been atomically installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from edgestack.data.sources import RawPayload, SourceBatch, bars_to_frame
from edgestack.models import AssetKey, Bar, BarRequest

Representation = Literal["raw", "adjusted", "actions"]
_SCHEMA_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class CachedFile:
    """Identity and shape of one immutable Parquet partition."""

    representation: Representation
    relative_path: str
    sha256: str
    rows: int


@dataclass(frozen=True, slots=True)
class CachedSnapshot:
    """Catalog record for one normalized, single-instrument source batch."""

    snapshot_id: str
    asset: AssetKey
    source: str
    start: date
    end: date
    fetched_at: datetime
    raw_sha256: str
    canonical_sha256: str
    warnings: tuple[str, ...]
    files: tuple[CachedFile, ...]


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, body: bytes) -> None:
    """Install ``body`` once; verify, rather than overwrite, an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise RuntimeError(f"immutable file differs at {path}")
        return
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, path)
        except OSError:
            if not path.exists() or path.read_bytes() != body:
                raise
    finally:
        temporary.unlink(missing_ok=True)


def _install_directory(source: Path, target: Path, *, attempts: int = 8) -> None:
    """Atomically rename a snapshot despite transient Windows file locks.

    Antivirus and indexing processes can briefly hold newly written Parquet files
    on Windows. Retrying the same same-volume rename preserves atomicity; this
    intentionally never falls back to copying a partially visible directory.
    """

    if attempts < 1:
        raise ValueError("attempts must be positive")
    if source.parent != target.parent:
        raise ValueError("atomic snapshot installation requires a shared parent")
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except OSError:
            if target.is_dir():
                source_manifest = source / "manifest.json"
                target_manifest = target / "manifest.json"
                if (
                    source_manifest.is_file()
                    and target_manifest.is_file()
                    and source_manifest.read_bytes() == target_manifest.read_bytes()
                ):
                    return
                raise RuntimeError(
                    f"immutable snapshot target differs at {target}"
                ) from None
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (2**attempt))


class ContentAddressedRawStore:
    """Filesystem content store implementing ``RawPayloadSink``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def body_path(self, digest: str) -> Path:
        """Return the deterministic path for a hexadecimal SHA-256 digest."""

        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("digest must be a lowercase SHA-256 hexadecimal string")
        return self.root / digest[:2] / digest[2:4] / f"{digest}.bin"

    def store(self, payload: RawPayload) -> str:
        """Persist exact bytes and every immutable sanitized fetch observation."""

        digest = payload.sha256
        body_path = self.body_path(digest)
        _atomic_bytes(body_path, payload.body)
        metadata = {
            "schema_version": _SCHEMA_VERSION,
            "sha256": digest,
            "source": payload.source,
            "asset": asdict(payload.asset) if payload.asset is not None else None,
            "fetched_at": payload.fetched_at.astimezone(UTC).isoformat(),
            "media_type": payload.media_type,
            "request_url": payload.request_url,
            "status_code": payload.status_code,
            "response_headers": dict(sorted(payload.response_headers.items())),
        }
        metadata_body = (_canonical_json(metadata) + "\n").encode()
        metadata_digest = hashlib.sha256(metadata_body).hexdigest()
        # Identical bytes can legitimately be observed for different requests or
        # at different fetch times. Key metadata by its own identity rather than
        # forcing unrelated observations into one mutable sidecar.
        metadata_path = body_path.with_name(
            f"{digest}.observation-{metadata_digest}.json"
        )
        _atomic_bytes(metadata_path, metadata_body)
        return digest

    def contains(self, digest: str) -> bool:
        """Whether exact bytes for ``digest`` are present and uncorrupted."""

        path = self.body_path(digest)
        return path.is_file() and _sha256_file(path) == digest

    def read(self, digest: str) -> bytes:
        """Read and verify one immutable response body."""

        path = self.body_path(digest)
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != digest:
            raise RuntimeError(f"raw payload hash mismatch: {digest}")
        return body

    def metadata(self, digest: str) -> Mapping[str, Any]:
        """Return the latest sanitized fetch observation for ``digest``."""

        records = self.metadata_records(digest)
        if not records:
            raise FileNotFoundError(f"raw metadata is missing: {digest}")
        return max(records, key=lambda item: str(item.get("fetched_at", "")))

    def metadata_records(self, digest: str) -> tuple[Mapping[str, Any], ...]:
        """Return every immutable fetch observation attached to exact bytes."""

        body_path = self.body_path(digest)
        paths = sorted(body_path.parent.glob(f"{digest}.observation-*.json"))
        legacy = body_path.with_suffix(".json")
        if legacy.is_file():
            paths.insert(0, legacy)
        records: list[Mapping[str, Any]] = []
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError(f"raw metadata is not an object: {path}")
            records.append(cast(dict[str, Any], payload))
        return tuple(records)


class DataCache:
    """SQLite-indexed immutable raw/canonical daily-bar cache."""

    def __init__(
        self,
        *,
        raw_root: str | Path,
        canonical_root: str | Path,
        catalog_path: str | Path,
    ) -> None:
        self.raw = ContentAddressedRawStore(raw_root)
        self.canonical_root = Path(canonical_root).resolve()
        self.canonical_root.mkdir(parents=True, exist_ok=True)
        self.catalog_path = Path(catalog_path).resolve()
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.catalog_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS data_snapshot (
                    snapshot_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    requested_adjusted INTEGER NOT NULL,
                    fetched_at TEXT NOT NULL,
                    raw_sha256 TEXT NOT NULL,
                    canonical_sha256 TEXT NOT NULL,
                    warnings_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS data_snapshot_file (
                    snapshot_id TEXT NOT NULL REFERENCES data_snapshot(snapshot_id),
                    representation TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    PRIMARY KEY (snapshot_id, relative_path)
                );
                CREATE INDEX IF NOT EXISTS idx_data_snapshot_lookup
                    ON data_snapshot(symbol, start_date, end_date, fetched_at);
                """
            )
            connection.commit()

    def store_payload(self, payload: RawPayload) -> str:
        """Store a provider payload for use as an adapter ``raw_sink`` callback."""

        return self.raw.store(payload)

    def store_batch(self, batch: SourceBatch) -> CachedSnapshot:
        """Atomically materialize raw, adjusted, and action Parquet partitions."""

        if not batch.bars:
            raise ValueError("cannot cache an empty SourceBatch")
        if any(bar.asset != batch.request.asset for bar in batch.bars):
            raise ValueError("SourceBatch contains bars for another asset")
        if not self.raw.contains(batch.raw_sha256):
            raise FileNotFoundError(
                "raw response is not in the configured content store; construct the "
                "adapter with raw_sink=cache.raw or import the payload before caching"
            )
        snapshot_id = self.snapshot_id(batch)
        existing = self.get_snapshot(snapshot_id)
        if existing is not None:
            return existing
        final_directory = self.canonical_root / snapshot_id
        if final_directory.exists():
            recovered = self._recover_unregistered(final_directory, snapshot_id)
            if recovered is not None:
                return recovered
            raise RuntimeError(
                f"unrecognized immutable snapshot directory: {final_directory}"
            )

        base = bars_to_frame(batch)
        raw_frame, adjusted_frame, actions_frame = _representations(base)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=self.canonical_root)
        ).resolve()
        if self.canonical_root not in temporary.parents:
            raise RuntimeError("temporary snapshot escaped canonical root")
        try:
            files: list[CachedFile] = []
            for representation, frame in (
                ("raw", raw_frame),
                ("adjusted", adjusted_frame),
                ("actions", actions_frame),
            ):
                files.extend(
                    self._write_partitioned(
                        temporary, representation, frame  # type: ignore[arg-type]
                    )
                )
            canonical_sha = _files_digest(files)
            manifest = _snapshot_payload(batch, snapshot_id, canonical_sha, files)
            _atomic_bytes(
                temporary / "manifest.json",
                (_canonical_json(manifest) + "\n").encode(),
            )
            _install_directory(temporary, final_directory)
            self._register(manifest)
        finally:
            if temporary.exists():
                if self.canonical_root not in temporary.parents:
                    raise RuntimeError("refusing to clean path outside canonical root")
                shutil.rmtree(temporary)
        stored = self.get_snapshot(snapshot_id)
        if stored is None:
            raise RuntimeError("snapshot registration failed")
        return stored

    def snapshot_id(self, batch: SourceBatch) -> str:
        """Derive a stable ID from request, provider, raw bytes, and schema version."""

        identity = {
            "schema_version": _SCHEMA_VERSION,
            "source": batch.source,
            "asset": asdict(batch.request.asset),
            "start": batch.request.start.isoformat(),
            "end": batch.request.end.isoformat(),
            "adjusted": batch.request.adjusted,
            "raw_sha256": batch.raw_sha256,
        }
        return hashlib.sha256(_canonical_json(identity).encode()).hexdigest()

    def _write_partitioned(
        self,
        root: Path,
        representation: Representation,
        frame: pd.DataFrame,
    ) -> list[CachedFile]:
        output: list[CachedFile] = []
        representation_root = root / f"representation={representation}"
        if frame.empty:
            representation_root.mkdir(parents=True, exist_ok=True)
            return output
        dated = frame.copy()
        dated["year"] = pd.to_datetime(dated["session"]).dt.year
        for year, partition in dated.groupby("year", sort=True):
            year_value = int(str(year))
            partition_path = (
                representation_root / f"year={year_value}" / "part-00000.parquet"
            )
            partition_path.parent.mkdir(parents=True, exist_ok=False)
            table = pa.Table.from_pandas(
                partition.drop(columns="year"), preserve_index=False
            )
            pq.write_table(table, partition_path, compression="zstd", version="2.6")
            relative = partition_path.relative_to(root).as_posix()
            output.append(
                CachedFile(
                    representation,
                    relative,
                    _sha256_file(partition_path),
                    len(partition),
                )
            )
        return output

    def _register(self, manifest: Mapping[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR IGNORE INTO data_snapshot (
                    snapshot_id, schema_version, symbol, exchange, asset_type,
                    source, start_date, end_date, requested_adjusted, fetched_at,
                    raw_sha256, canonical_sha256, warnings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest["snapshot_id"],
                    manifest["schema_version"],
                    manifest["asset"]["symbol"],
                    manifest["asset"]["exchange"],
                    manifest["asset"]["asset_type"],
                    manifest["source"],
                    manifest["start"],
                    manifest["end"],
                    int(manifest["requested_adjusted"]),
                    manifest["fetched_at"],
                    manifest["raw_sha256"],
                    manifest["canonical_sha256"],
                    _canonical_json({"warnings": manifest["warnings"]}),
                    manifest["created_at"],
                ),
            )
            for file_record in manifest["files"]:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO data_snapshot_file
                    (snapshot_id, representation, relative_path, sha256, row_count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        manifest["snapshot_id"],
                        file_record["representation"],
                        file_record["relative_path"],
                        file_record["sha256"],
                        file_record["rows"],
                    ),
                )
            connection.commit()

    def _recover_unregistered(
        self, directory: Path, expected_id: str
    ) -> CachedSnapshot | None:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("snapshot_id") != expected_id:
            return None
        files = tuple(CachedFile(**record) for record in manifest.get("files", []))
        for record in files:
            path = directory / record.relative_path
            if not path.is_file() or _sha256_file(path) != record.sha256:
                raise RuntimeError(f"snapshot recovery hash mismatch: {path}")
        if _files_digest(files) != manifest.get("canonical_sha256"):
            raise RuntimeError("snapshot recovery manifest digest mismatch")
        self._register(manifest)
        return self.get_snapshot(expected_id)

    def get_snapshot(self, snapshot_id: str) -> CachedSnapshot | None:
        """Look up one snapshot without opening its Parquet data."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM data_snapshot WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
            if row is None:
                return None
            file_rows = connection.execute(
                """
                SELECT representation, relative_path, sha256, row_count
                FROM data_snapshot_file WHERE snapshot_id=? ORDER BY relative_path
                """,
                (snapshot_id,),
            ).fetchall()
        warnings = tuple(json.loads(row["warnings_json"])["warnings"])
        files = tuple(
            CachedFile(
                item["representation"],
                item["relative_path"],
                item["sha256"],
                item["row_count"],
            )
            for item in file_rows
        )
        return CachedSnapshot(
            row["snapshot_id"],
            AssetKey(row["symbol"], row["exchange"], row["asset_type"]),
            row["source"],
            date.fromisoformat(row["start_date"]),
            date.fromisoformat(row["end_date"]),
            datetime.fromisoformat(row["fetched_at"]),
            row["raw_sha256"],
            row["canonical_sha256"],
            warnings,
            files,
        )

    def latest_snapshot(
        self, asset: AssetKey, *, source: str | None = None
    ) -> CachedSnapshot | None:
        """Return the newest fetched snapshot for an asset and optional provider."""

        query = (
            "SELECT snapshot_id FROM data_snapshot "
            "WHERE symbol=? AND exchange=? AND asset_type=?"
        )
        parameters: list[Any] = [asset.symbol, asset.exchange, asset.asset_type]
        if source is not None:
            query += " AND source=?"
            parameters.append(source)
        query += " ORDER BY fetched_at DESC, snapshot_id DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        return self.get_snapshot(row["snapshot_id"]) if row is not None else None

    def read_frame(
        self,
        snapshot_id: str,
        *,
        representation: Representation = "adjusted",
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read one verified representation from immutable Parquet partitions."""

        snapshot = self.get_snapshot(snapshot_id)
        if snapshot is None:
            raise KeyError(f"unknown snapshot: {snapshot_id}")
        records = [
            record
            for record in snapshot.files
            if record.representation == representation
        ]
        if not records:
            return pd.DataFrame(columns=list(columns or ()))
        frames: list[pd.DataFrame] = []
        for record in records:
            path = self.canonical_root / snapshot_id / record.relative_path
            if _sha256_file(path) != record.sha256:
                raise RuntimeError(f"canonical partition hash mismatch: {path}")
            frames.append(
                pd.read_parquet(path, columns=list(columns) if columns else None)
            )
        result = pd.concat(frames, ignore_index=True)
        sort_columns = [column for column in ("symbol", "session") if column in result]
        return (
            result.sort_values(sort_columns, kind="stable").reset_index(drop=True)
            if sort_columns
            else result
        )

    def load_batch(self, snapshot_id: str) -> SourceBatch:
        """Reconstruct the shared immutable ``SourceBatch`` from cached Parquet."""

        snapshot = self.get_snapshot(snapshot_id)
        if snapshot is None:
            raise KeyError(f"unknown snapshot: {snapshot_id}")
        raw = self.read_frame(snapshot_id, representation="raw")
        adjusted = self.read_frame(snapshot_id, representation="adjusted")
        by_session: dict[object, float] = (
            {
                key: float(value)
                for key, value in adjusted.set_index("session")["close"].items()
            }
            if not adjusted.empty
            else {}
        )
        records = cast(list[dict[str, Any]], raw.to_dict("records"))
        bars = tuple(
            Bar(
                asset=snapshot.asset,
                event_time=pd.Timestamp(row["event_time"]).to_pydatetime(),
                available_at=pd.Timestamp(row["available_at"]).to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                adjusted_close=float(by_session[row["session"]]),
                dividend=float(row["dividend"]),
                split_factor=float(row["split_factor"]),
                source=str(row["source"]),
            )
            for row in records
        )
        request = BarRequest(snapshot.asset, snapshot.start, snapshot.end, True)
        return SourceBatch(
            snapshot.source,
            request,
            bars,
            snapshot.fetched_at,
            snapshot.raw_sha256,
            snapshot.warnings,
        )


def _representations(
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = [
        "symbol",
        "exchange",
        "asset_type",
        "session",
        "event_time",
        "available_at",
        "dividend",
        "split_factor",
        "source",
    ]
    raw = base.loc[:, [*common, "open", "high", "low", "close", "volume"]].copy()
    factor = base["adjusted_close"].div(base["close"]).where(base["close"] > 0, 1.0)
    adjusted = base.loc[:, common].copy()
    for column in ("open", "high", "low", "close"):
        adjusted[column] = base[column] * factor
    adjusted["volume"] = base["volume"].div(factor.where(factor > 0, 1.0))
    ordered = [*common, "open", "high", "low", "close", "volume"]
    raw = raw.loc[:, ordered]
    adjusted = adjusted.loc[:, ordered]
    dividends = base.loc[base["dividend"] != 0, common].copy()
    dividends["action"] = "dividend"
    dividends["value"] = base.loc[base["dividend"] != 0, "dividend"].to_numpy()
    splits = base.loc[base["split_factor"] != 1, common].copy()
    splits["action"] = "split"
    splits["value"] = base.loc[base["split_factor"] != 1, "split_factor"].to_numpy()
    actions = pd.concat([dividends, splits], ignore_index=True)
    return raw, adjusted, actions


def _files_digest(files: Sequence[CachedFile]) -> str:
    identity = [
        {
            "representation": record.representation,
            "relative_path": record.relative_path,
            "sha256": record.sha256,
            "rows": record.rows,
        }
        for record in sorted(files, key=lambda item: item.relative_path)
    ]
    return hashlib.sha256(_canonical_json({"files": identity}).encode()).hexdigest()


def _snapshot_payload(
    batch: SourceBatch,
    snapshot_id: str,
    canonical_sha: str,
    files: Sequence[CachedFile],
) -> Mapping[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "asset": asdict(batch.request.asset),
        "source": batch.source,
        "start": batch.request.start.isoformat(),
        "end": batch.request.end.isoformat(),
        "requested_adjusted": batch.request.adjusted,
        "fetched_at": batch.fetched_at.astimezone(UTC).isoformat(),
        "raw_sha256": batch.raw_sha256,
        "canonical_sha256": canonical_sha,
        "warnings": list(batch.warnings),
        "created_at": datetime.now(UTC).isoformat(),
        "files": [asdict(record) for record in files],
    }


__all__ = [
    "CachedFile",
    "CachedSnapshot",
    "ContentAddressedRawStore",
    "DataCache",
    "Representation",
]
