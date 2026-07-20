"""Sealed international country-ETF panel for the 2026-07-19c study round.

One-time seal: fetch the declared US-listed country ETFs (plus SPY as the
benchmark) through the free Stooq→Yahoo chain and freeze them to
``artifacts/intl/bars.parquet`` with a SHA-256 manifest. Studies load ONLY
the sealed file, so every evaluation is reproducible against frozen bytes;
re-sealing overwrites nothing unless ``--force`` is passed.

The instruments are US-listed wrappers: returns embed currency moves and
the wrapper's own NYSE liquidity. That is stamped, not hidden.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from edgestack.provenance import sha256_file

INSTRUMENTS = (
    "EWG",
    "EWU",
    "EWQ",
    "EWP",
    "EWI",
    "EWL",
    "EWD",
    "EWN",
    "FEZ",
    "EWY",
    "EWJ",
    "EWA",
)
BENCHMARK = "SPY"
WATERMARKS = (
    "US_LISTED_WRAPPERS_EMBED_FX_AND_NYSE_LIQUIDITY",
    "SURVIVORSHIP_NOT_APPLICABLE_FIXED_ETF_LIST",
)


async def _fetch_all(symbols: Sequence[str]) -> pd.DataFrame:
    from edgestack.data.sources import (
        FallbackDailyBarSource,
        StooqDailyBarSource,
        YahooDailyBarSource,
        bars_to_frame,
    )
    from edgestack.models import AssetKey, BarRequest

    chain = FallbackDailyBarSource((StooqDailyBarSource(), YahooDailyBarSource()))

    async def one(symbol: str) -> pd.DataFrame:
        batch = await chain.fetch_bars(
            BarRequest(
                AssetKey(symbol, asset_type="etf"),
                date.today() - timedelta(days=365 * 31),
                date.today(),
                adjusted=True,
            )
        )
        return bars_to_frame(batch)

    frames = await asyncio.gather(*[one(symbol) for symbol in symbols])
    return pd.concat(frames, ignore_index=True)


def seal(*, root: str | Path = ".", force: bool = False) -> dict[str, Any]:
    """Fetch and freeze the panel; refuses to overwrite unless forced."""

    out_dir = Path(root).resolve() / "artifacts" / "intl"
    bars_path = out_dir / "bars.parquet"
    if bars_path.is_file() and not force:
        return {"status": "ALREADY_SEALED", "path": str(bars_path)}
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = (*INSTRUMENTS, BENCHMARK)
    frame = asyncio.run(_fetch_all(symbols))
    frame.to_parquet(bars_path, index=False)
    manifest = {
        "sealed_at": datetime.now(UTC).isoformat(),
        "instruments": list(INSTRUMENTS),
        "benchmark": BENCHMARK,
        "rows": len(frame),
        "first_session": str(frame["session"].min().date()),
        "last_session": str(frame["session"].max().date()),
        "bars_sha256": sha256_file(bars_path),
        "watermarks": list(WATERMARKS),
        "source": "Stooq first, Yahoo whole-series fallback (free chain)",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"status": "SEALED", **manifest}


def load_panel(base: str | Path = ".") -> dict[str, Any]:
    """Load the sealed panel in the standard study-panel dict shape."""

    out_dir = Path(base).resolve() / "artifacts" / "intl"
    bars = pd.read_parquet(
        out_dir / "bars.parquet",
        columns=["symbol", "session", "open", "close", "adjusted_close", "volume"],
    )
    bars["session"] = pd.to_datetime(bars["session"])
    frames: dict[str, Any] = {
        field: bars.pivot_table(
            index="session", columns="symbol", values=field, aggfunc="first"
        ).sort_index()
        for field in ("open", "close", "adjusted_close", "volume")
    }
    frames["asset_types"] = pd.Series(dict.fromkeys((*INSTRUMENTS, BENCHMARK), "etf"))
    return frames


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            seal(root=arguments.root, force=arguments.force),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
