"""Expiring manual context for eToro costs and geopolitical risk."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from edgestack.oil.models import OilContext


class OilContextStore:
    """Small local state file; expired context is never silently reused."""

    def __init__(self, path: str | Path = "artifacts/oil/context.json") -> None:
        self.path = Path(path).resolve()

    def write(self, context: OilContext) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = context.model_dump_json(indent=2).encode("utf-8") + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.path)
        finally:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
        return self.path

    def read(self, *, at: datetime | None = None) -> OilContext | None:
        if not self.path.is_file():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        context = OilContext.model_validate(payload)
        moment = at or datetime.now(UTC)
        if moment.tzinfo is None:
            raise ValueError("oil context read time must be timezone-aware")
        return context if context.recorded_at <= moment < context.expires_at else None


__all__ = ["OilContextStore"]
