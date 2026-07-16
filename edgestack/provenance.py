"""Hashing and runtime-provenance helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 hex digest of bytes."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(payload.encode("utf-8"))


def source_tree_sha256(root: str | Path) -> str:
    """Hash tracked source content, excluding data/build artifacts."""

    base = Path(root)
    digest = hashlib.sha256()
    excluded_roots = {
        ".git",
        ".venv",
        "artifacts",
        "build",
        "data",
        "dist",
    }
    excluded_parts = {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    }
    try:
        listed = subprocess.run(
            [
                "git",
                "-C",
                str(base),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        candidates = [
            base / item.decode("utf-8", errors="surrogateescape")
            for item in listed.split(b"\0")
            if item
        ]
    except (subprocess.SubprocessError, OSError):
        candidates = [item for item in base.rglob("*") if item.is_file()]
    for path in sorted(candidates):
        relative = path.relative_to(base)
        if relative.parts[0] in excluded_roots or excluded_parts.intersection(
            relative.parts
        ):
            continue
        digest.update(relative.as_posix().encode())
        digest.update(path.read_bytes() if path.is_file() else b"<MISSING>")
    return digest.hexdigest()


def runtime_manifest() -> Mapping[str, Any]:
    """Capture interpreter, platform, and installed-package versions."""

    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.splitlines()
    except (subprocess.SubprocessError, OSError):
        freeze = []
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": sorted(freeze),
    }
