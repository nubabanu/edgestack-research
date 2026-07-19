"""Hash-pinned import and causal provenance validation for V2 data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd

_PROVENANCE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "event_time",
        "available_at",
        "source",
        "revision",
        "fetched_at",
        "content_hash",
    }
)


@dataclass(frozen=True, slots=True)
class PinnedDataset:
    """Validated local dataset plus immutable file identity."""

    path: Path
    sha256: str
    frame: pd.DataFrame


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hash_pinned(path: str | Path, expected_sha256: str) -> PinnedDataset:
    """Load CSV/Parquet only when its declared content hash matches exactly."""

    source = Path(path)
    actual = sha256_file(source)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"dataset hash mismatch: expected {expected_sha256}, got {actual}"
        )
    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    else:
        raise ValueError("licensed imports must be CSV or Parquet")
    validate_provenance(frame)
    return PinnedDataset(source.resolve(), actual, frame)


def validate_provenance(frame: pd.DataFrame) -> None:
    """Require complete, causal, timezone-aware provenance on every record."""

    missing = _PROVENANCE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing provenance columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("dataset is empty")
    if bool(frame[list(_PROVENANCE_COLUMNS)].isna().any().any()):
        raise ValueError("provenance fields cannot be null")
    pd.to_datetime(frame["event_time"], utc=True, errors="raise")
    available = pd.to_datetime(frame["available_at"], utc=True, errors="raise")
    fetched = pd.to_datetime(frame["fetched_at"], utc=True, errors="raise")
    if bool((fetched < available).any()):
        raise ValueError("fetched_at cannot precede available_at")
    hashes = frame["content_hash"].astype(str)
    if bool((~hashes.str.fullmatch(r"[0-9a-fA-F]{64}")).any()):
        raise ValueError("content_hash must be a SHA-256 hex digest")


def causal_prefix(frame: pd.DataFrame, decision_time: datetime) -> pd.DataFrame:
    """Return only vintages available by the decision cutoff."""

    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    validate_provenance(frame)
    cutoff = pd.Timestamp(decision_time.astimezone(UTC))
    available = pd.to_datetime(frame["available_at"], utc=True)
    return frame.loc[available <= cutoff].copy()
