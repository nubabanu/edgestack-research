"""Atomic, content-addressed artifact persistence."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

from edgestack.provenance import sha256_bytes


class ArtifactStore:
    """Persist immutable raw payloads and derived campaign artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_raw(self, payload: bytes, suffix: str = ".bin") -> tuple[str, Path]:
        """Store bytes by hash without overwriting existing content."""

        digest = sha256_bytes(payload)
        path = self.root / "raw" / digest[:2] / f"{digest}{suffix}"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_bytes(path, payload)
        return digest, path

    def write_json(self, relative: str | Path, value: Any) -> Path:
        """Atomically write deterministic JSON."""

        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, sort_keys=True, indent=2, default=str).encode()
        self._atomic_bytes(path, payload)
        return path

    def write_text(self, relative: str | Path, value: str) -> Path:
        """Atomically install one immutable UTF-8 text artifact."""

        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_bytes(path, value.encode("utf-8"))
        return path

    def write_parquet(self, relative: str | Path, frame: pd.DataFrame) -> Path:
        """Atomically write a compressed Parquet artifact."""

        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".parquet", delete=False
        ) as tmp:
            temp = Path(tmp.name)
        try:
            frame.to_parquet(temp, index=False, compression="zstd")
            ArtifactStore._install_once(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return path

    @staticmethod
    def _atomic_bytes(path: Path, payload: bytes) -> None:
        if path.exists():
            if path.read_bytes() != payload:
                raise RuntimeError(f"immutable artifact differs at {path}")
            return
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp = Path(tmp.name)
        try:
            ArtifactStore._install_once(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def _install_once(source: Path, target: Path, *, attempts: int = 8) -> None:
        """Atomically install an artifact without ever overwriting an identity."""

        for attempt in range(attempts):
            if target.exists():
                if not ArtifactStore._same_file(source, target):
                    raise RuntimeError(f"immutable artifact differs at {target}")
                return
            try:
                # A same-directory hard link publishes a fully flushed inode and
                # fails atomically if another process already installed the name.
                os.link(source, target)
                return
            except OSError:
                if target.exists():
                    continue
                if attempt + 1 == attempts:
                    raise
                time.sleep(0.05 * (2**attempt))

    @staticmethod
    def _same_file(left: Path, right: Path) -> bool:
        """Compare large artifacts without loading either one into memory."""

        if left.stat().st_size != right.stat().st_size:
            return False
        with left.open("rb") as left_stream, right.open("rb") as right_stream:
            while True:
                left_chunk = left_stream.read(1024 * 1024)
                right_chunk = right_stream.read(1024 * 1024)
                if left_chunk != right_chunk:
                    return False
                if not left_chunk:
                    return True
