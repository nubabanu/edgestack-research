"""Single-use final-holdout ceremony."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from edgestack.models import HoldoutFreezeManifest
from edgestack.storage.catalog import Catalog


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
