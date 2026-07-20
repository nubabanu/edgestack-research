"""Free point-in-time earnings data from SEC EDGAR.

Two facts per company, both free, keyless, and officially timestamped:

* announcement moments — Form 8-K filings whose item list contains 2.02
  (Results of Operations), with EDGAR acceptance datetimes to the second;
* reported quarterly EPS — XBRL companyconcept series (diluted, basic
  fallback), using frame-tagged duration entries.

Exact response bytes are persisted through the content-addressed raw store
before anything is parsed; processed tables are written as parquet plus a
manifest with coverage statistics and watermarks. The declared approximation
is stamped, never hidden: press-release EPS is assumed equal to the later
XBRL figure (PRESS_RELEASE_EQUALS_XBRL), and XBRL coverage effectively
starts in 2009, so studies built on this feed inherit both stamps.

SEC fair-use: identified User-Agent and a global request-rate ceiling.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from aiolimiter import AsyncLimiter

from edgestack.data.cache import ContentAddressedRawStore
from edgestack.data.sources import RawPayload
from edgestack.models import AssetKey

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_CONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"
)
_EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")
_REQUESTS_PER_SECOND = 8
# SEC fair-use policy requires an identifying User-Agent with a contact
# address; the repo owner's public git-author email is the default contact.
_USER_AGENT = os.environ.get(
    "EDGESTACK_EDGAR_USER_AGENT",
    "EdgeStack research (yucelnumankaradavut@gmail.com)",
)

WATERMARKS = (
    "PRESS_RELEASE_EQUALS_XBRL_APPROXIMATION",
    "XBRL_COVERAGE_EFFECTIVELY_2009_PLUS",
    "ANNOUNCEMENT_TIMESTAMP_IS_EDGAR_ACCEPTANCE",
)


def _store(
    sink: ContentAddressedRawStore, url: str, symbol: str | None, body: bytes
) -> str:
    return sink.store(
        RawPayload(
            source="sec_edgar",
            asset=AssetKey(symbol) if symbol else None,
            fetched_at=datetime.now(UTC),
            media_type="application/json",
            body=body,
            request_url=url,
        )
    )


async def _get_json(
    client: httpx.AsyncClient,
    limiter: AsyncLimiter,
    sink: ContentAddressedRawStore,
    url: str,
    symbol: str | None,
) -> Any:
    async with limiter:
        response = await client.get(url)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    _store(sink, url, symbol, response.content)
    return response.json()


async def fetch_cik_map(
    client: httpx.AsyncClient, limiter: AsyncLimiter, sink: ContentAddressedRawStore
) -> dict[str, int]:
    """Symbol (upper, dots normalized to dashes) -> CIK for all EDGAR filers."""

    payload = await _get_json(client, limiter, sink, _TICKERS_URL, None)
    mapping: dict[str, int] = {}
    for row in payload.values():
        mapping[str(row["ticker"]).upper().replace(".", "-")] = int(row["cik_str"])
    return mapping


def _announcements_from_submissions(symbol: str, payload: Any) -> list[dict[str, Any]]:
    recent = payload.get("filings", {}).get("recent", {})
    rows: list[dict[str, Any]] = []
    forms = recent.get("form", [])
    for index, form in enumerate(forms):
        if form not in {"8-K", "8-K/A"}:
            continue
        items = str(recent.get("items", [""] * len(forms))[index])
        if "2.02" not in items:
            continue
        rows.append(
            {
                "symbol": symbol,
                "form": form,
                "items": items,
                "acceptance": recent["acceptanceDateTime"][index],
                "filing_date": recent["filingDate"][index],
                "accession": recent["accessionNumber"][index],
            }
        )
    return rows


def _eps_from_concept(symbol: str, tag: str, payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    units = payload.get("units", {}) if payload else {}
    for entry in units.get("USD/shares", []):
        frame = entry.get("frame")
        # Frame-tagged duration entries are EDGAR's deduplicated canonical
        # quarterly series (CYyyyyQq); everything else is a repeat filing.
        if not frame or "Q" not in frame:
            continue
        rows.append(
            {
                "symbol": symbol,
                "tag": tag,
                "frame": frame,
                "end": entry.get("end"),
                "value": float(entry["val"]),
                "filed": entry.get("filed"),
                "form": entry.get("form"),
                "fiscal_period": entry.get("fp"),
            }
        )
    return rows


async def _crawl_symbol(
    client: httpx.AsyncClient,
    limiter: AsyncLimiter,
    sink: ContentAddressedRawStore,
    symbol: str,
    cik: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    try:
        submissions = await _get_json(
            client, limiter, sink, _SUBMISSIONS_URL.format(cik=cik), symbol
        )
        announcements = (
            _announcements_from_submissions(symbol, submissions) if submissions else []
        )
        eps: list[dict[str, Any]] = []
        for tag in _EPS_TAGS:
            payload = await _get_json(
                client,
                limiter,
                sink,
                _CONCEPT_URL.format(cik=cik, tag=tag),
                symbol,
            )
            eps = _eps_from_concept(symbol, tag, payload)
            if eps:
                break
        return announcements, eps, None
    except Exception as error:  # per-name failure must not kill the crawl
        return [], [], f"{type(error).__name__}: {error}"


async def crawl(symbols: Sequence[str], *, root: str | Path = ".") -> dict[str, Any]:
    """Crawl EDGAR for every symbol; persist raw bytes, tables, manifest."""

    base = Path(root).resolve()
    out_dir = base / "artifacts" / "earnings"
    out_dir.mkdir(parents=True, exist_ok=True)
    sink = ContentAddressedRawStore(out_dir / "raw")
    limiter = AsyncLimiter(_REQUESTS_PER_SECOND, 1)
    headers = {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        cik_map = await fetch_cik_map(client, limiter, sink)
        wanted = [s.upper().replace(".", "-") for s in symbols]
        missing = sorted(s for s in wanted if s not in cik_map)
        tasks = [
            _crawl_symbol(client, limiter, sink, symbol, cik_map[symbol])
            for symbol in wanted
            if symbol in cik_map
        ]
        results = await asyncio.gather(*tasks)
    announcements = [row for result in results for row in result[0]]
    eps = [row for result in results for row in result[1]]
    failures = {
        symbol: error
        for symbol, (_, _, error) in zip(
            [s for s in wanted if s in cik_map], results, strict=True
        )
        if error
    }
    announcements_frame = pd.DataFrame(announcements)
    eps_frame = pd.DataFrame(eps)
    announcements_frame.to_parquet(out_dir / "announcements.parquet", index=False)
    eps_frame.to_parquet(out_dir / "eps.parquet", index=False)
    manifest = {
        "source": "SEC EDGAR (free, keyless)",
        "fetched_at": datetime.now(UTC).isoformat(),
        "symbols_requested": len(wanted),
        "symbols_without_cik": missing,
        "symbols_failed": failures,
        "announcement_rows": len(announcements_frame),
        "eps_rows": len(eps_frame),
        "announcement_symbols": (
            int(announcements_frame["symbol"].nunique())
            if len(announcements_frame)
            else 0
        ),
        "eps_symbols": int(eps_frame["symbol"].nunique()) if len(eps_frame) else 0,
        "watermarks": list(WATERMARKS),
        "user_agent": _USER_AGENT,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def fetch_latest_announcement(symbol: str) -> dict[str, Any] | None:
    """Newest 8-K Item 2.02 for one symbol, live from EDGAR (two requests).

    Used by the nightly window-open check; returns None when the symbol has
    no CIK or no qualifying filings. Same row schema as the sealed crawl.
    """

    import httpx

    headers = {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    ticker = symbol.upper().replace(".", "-")
    with httpx.Client(headers=headers, timeout=30.0) as client:
        tickers = client.get(_TICKERS_URL)
        tickers.raise_for_status()
        cik = next(
            (
                int(row["cik_str"])
                for row in tickers.json().values()
                if str(row["ticker"]).upper().replace(".", "-") == ticker
            ),
            None,
        )
        if cik is None:
            return None
        response = client.get(_SUBMISSIONS_URL.format(cik=cik))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        rows = _announcements_from_submissions(ticker, response.json())
    if not rows:
        return None
    return max(rows, key=lambda row: str(row["acceptance"]))


def load_events(root: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the crawled announcement and EPS tables (raises if absent)."""

    out_dir = Path(root).resolve() / "artifacts" / "earnings"
    announcements = pd.read_parquet(out_dir / "announcements.parquet")
    eps = pd.read_parquet(out_dir / "eps.parquet")
    return announcements, eps


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated override; default is the sealed campaign equities.",
    )
    arguments = parser.parse_args(argv)
    if arguments.symbols:
        symbols = [s.strip() for s in arguments.symbols.split(",") if s.strip()]
    else:
        universe = pd.read_parquet(
            Path(arguments.root)
            / "artifacts/campaigns/full-stooq-literature-v2-20260715-001"
            / "data/universe.parquet"
        )
        symbols = sorted(
            universe.loc[universe["asset_type"] == "equity", "symbol"].astype(str)
        )
    manifest = asyncio.run(crawl(symbols, root=arguments.root))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
