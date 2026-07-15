"""Current S&P 500 plus nine liquid ETFs and a Wikipedia history hook."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from typing import Final

import httpx

from edgestack.data.sources import RawPayload, RawPayloadSink
from edgestack.models import AssetKey, MembershipInterval

WIKIPEDIA_SP500_URL: Final = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
LIQUID_ETFS: Final[tuple[str, ...]] = (
    "SPY",
    "QQQ",
    "IWM",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLY",
    "XLI",
)


@dataclass(frozen=True, slots=True)
class SP500Constituent:
    """One row of Wikipedia's current-constituents table."""

    symbol: str
    security: str
    sector: str
    sub_industry: str
    date_added: date | None
    cik: str | None


@dataclass(frozen=True, slots=True)
class MembershipChange:
    """One effective-date addition/removal from Wikipedia's change log."""

    effective_date: date
    added_symbol: str | None
    added_security: str | None
    removed_symbol: str | None
    removed_security: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    """Membership result plus immutable source identity and bias warnings."""

    as_of: datetime
    source_sha256: str
    memberships: tuple[MembershipInterval, ...]
    changes: tuple[MembershipChange, ...]
    warnings: tuple[str, ...]
    current_constituent_count: int
    etfs: tuple[str, ...] = LIQUID_ETFS


class _WikipediaTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.table: str | None = None
        self.depth = 0
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.rows: dict[str, list[list[str]]] = {"constituents": [], "changes": []}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") in self.rows and self.table is None:
            self.table = attributes["id"]
            self.depth = 1
            return
        if self.table is None:
            return
        if tag == "table":
            self.depth += 1
        elif tag == "tr" and self.depth == 1:
            self.row = []
        elif tag in {"th", "td"} and self.row is not None and self.depth == 1:
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.table is None:
            return
        if tag in {"th", "td"} and self.cell is not None and self.row is not None:
            value = " ".join("".join(self.cell).replace("\xa0", " ").split())
            self.row.append(value)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if any(self.row):
                self.rows[self.table].append(self.row)
            self.row = None
        elif tag == "table":
            self.depth -= 1
            if self.depth == 0:
                self.table = None


def parse_wikipedia_sp500_html(
    html: str,
) -> tuple[tuple[SP500Constituent, ...], tuple[MembershipChange, ...]]:
    """Parse current constituents and membership changes without optional parsers."""

    parser = _WikipediaTableParser()
    parser.feed(html)
    constituents: list[SP500Constituent] = []
    for row in parser.rows["constituents"]:
        if not row or row[0].strip().lower() == "symbol" or len(row) < 4:
            continue
        symbol = _canonical_symbol(row[0])
        if not symbol:
            continue
        date_added = _parse_date(row[5]) if len(row) > 5 else None
        constituents.append(
            SP500Constituent(
                symbol,
                row[1] if len(row) > 1 else symbol,
                row[2] if len(row) > 2 else "Unknown",
                row[3] if len(row) > 3 else "Unknown",
                date_added,
                _digits_or_none(row[6]) if len(row) > 6 else None,
            )
        )
    changes: list[MembershipChange] = []
    for row in parser.rows["changes"]:
        if not row or not _parse_date(row[0]):
            continue
        # Data rows have date, added ticker/security, removed ticker/security, reason.
        # Rowspans occasionally cause an omitted cell; right-pad deterministically.
        values = row + [""] * (6 - len(row))
        effective = _parse_date(values[0])
        if effective is None:
            continue
        added = _canonical_symbol(values[1]) or None
        removed = _canonical_symbol(values[3]) or None
        if added is None and removed is None:
            continue
        changes.append(
            MembershipChange(
                effective,
                added,
                values[2] or None,
                removed,
                values[4] or None,
                values[5],
            )
        )
    if not constituents:
        raise ValueError("Wikipedia HTML did not contain a usable constituents table")
    return (
        tuple(sorted(constituents, key=lambda item: item.symbol)),
        tuple(sorted(changes, key=lambda item: item.effective_date, reverse=True)),
    )


def reconstruct_membership_intervals(
    current: Sequence[SP500Constituent],
    changes: Sequence[MembershipChange],
    *,
    start: date,
    end: date,
) -> tuple[MembershipInterval, ...]:
    """Reverse Wikipedia changes to form best-effort PIT membership intervals.

    Wikipedia's log is a useful hook, not a licensed constituent database.  For
    tickers active before the earliest logged event, the requested ``start`` is
    used as an inferred left boundary.
    """

    if end < start:
        raise ValueError("end must be on or after start")
    current_by_symbol = {item.symbol: item for item in current}
    active: dict[str, date | None] = {symbol: None for symbol in current_by_symbol}
    intervals: list[MembershipInterval] = []
    relevant = [change for change in changes if change.effective_date <= end]
    for change in sorted(relevant, key=lambda item: item.effective_date, reverse=True):
        added = change.added_symbol
        if added and added in active:
            interval_end = active.pop(added)
            constituent = current_by_symbol.get(added)
            interval_start = change.effective_date
            if interval_end is None or interval_end > start:
                intervals.append(
                    MembershipInterval(
                        AssetKey(added),
                        max(start, interval_start),
                        interval_end,
                        constituent.sector if constituent else None,
                        None,
                    )
                )
        removed = change.removed_symbol
        if removed:
            active[removed] = change.effective_date
    for symbol, interval_end in active.items():
        constituent = current_by_symbol.get(symbol)
        known_added = constituent.date_added if constituent else None
        interval_start = max(start, known_added) if known_added else start
        if interval_end is None or interval_end > interval_start:
            intervals.append(
                MembershipInterval(
                    AssetKey(symbol),
                    interval_start,
                    interval_end,
                    constituent.sector if constituent else None,
                    None,
                )
            )
    return tuple(
        sorted(
            (
                interval
                for interval in intervals
                if interval.start <= end
                and (interval.end is None or interval.end > start)
            ),
            key=lambda item: (item.asset.symbol, item.start),
        )
    )


class WikipediaSP500UniverseSource:
    """Current S&P 500 snapshot with optional best-effort change-log reconstruction."""

    def __init__(
        self,
        *,
        include_etfs: bool = True,
        reconstruct_history: bool = False,
        client: httpx.AsyncClient | None = None,
        raw_sink: RawPayloadSink | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.include_etfs = include_etfs
        self.reconstruct_history = reconstruct_history
        self._client = client
        self._raw_sink = raw_sink
        self._timeout = timeout
        self.last_snapshot: UniverseSnapshot | None = None

    async def memberships(
        self, start: date, end: date
    ) -> tuple[MembershipInterval, ...]:
        """Return current-snapshot or reconstructed memberships plus nine ETFs."""

        if end < start:
            raise ValueError("end must be on or after start")
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "EdgeStack/0.1 (research client; contact: "
                    "edgestack-research@example.invalid)"
                )
            },
        )
        try:
            response = await client.get(WIKIPEDIA_SP500_URL)
            response.raise_for_status()
        finally:
            if owns_client:
                await client.aclose()
        fetched_at = datetime.now(UTC)
        digest = hashlib.sha256(response.content).hexdigest()
        if self._raw_sink is not None:
            stored = self._raw_sink.store(
                RawPayload(
                    "wikipedia",
                    None,
                    fetched_at,
                    response.headers.get("content-type", "text/html").split(";")[0],
                    response.content,
                    WIKIPEDIA_SP500_URL,
                    response.status_code,
                    {
                        key.lower(): value
                        for key, value in response.headers.items()
                        if key.lower()
                        in {"content-type", "etag", "last-modified", "date"}
                    },
                )
            )
            if stored != digest:
                raise RuntimeError("raw universe sink returned the wrong content hash")
        constituents, changes = parse_wikipedia_sp500_html(response.text)
        if self.reconstruct_history:
            memberships = list(
                reconstruct_membership_intervals(
                    constituents, changes, start=start, end=end
                )
            )
            warnings = (
                "PIT_APPROXIMATION: Wikipedia change-log reconstruction is incomplete "
                "and is not a licensed point-in-time constituent database.",
            )
        else:
            memberships = [
                MembershipInterval(
                    AssetKey(item.symbol),
                    start,
                    None,
                    item.sector,
                    fetched_at,
                )
                for item in constituents
            ]
            warnings = (
                "SURVIVORSHIP_BIASED: current S&P 500 constituents are applied to "
                "historical returns; every downstream result must retain this watermark.",
            )
        if self.include_etfs:
            memberships.extend(
                MembershipInterval(
                    AssetKey(symbol, asset_type="etf"),
                    start,
                    None,
                    "ETF",
                    fetched_at,
                )
                for symbol in LIQUID_ETFS
            )
        result = tuple(
            sorted(
                memberships,
                key=lambda item: (item.asset.asset_type, item.asset.symbol, item.start),
            )
        )
        self.last_snapshot = UniverseSnapshot(
            fetched_at,
            digest,
            result,
            changes,
            warnings,
            len(constituents),
        )
        return result

    async def membership_changes(
        self, start: date, end: date
    ) -> tuple[MembershipChange, ...]:
        """Fetch and filter the Wikipedia change log as an explicit PIT hook."""

        prior = self.reconstruct_history
        self.reconstruct_history = True
        try:
            await self.memberships(start, end)
        finally:
            self.reconstruct_history = prior
        assert self.last_snapshot is not None
        return tuple(
            change
            for change in self.last_snapshot.changes
            if start <= change.effective_date <= end
        )


def _canonical_symbol(value: str) -> str:
    # Footnote artifacts are removed while preserving canonical S&P dot symbols.
    return re.sub(r"\[[^]]+\]", "", value).strip().upper().replace("\N{EN DASH}", "-")


def _parse_date(value: str) -> date | None:
    cleaned = re.sub(r"\[[^]]+\]", "", value).strip()
    if not cleaned:
        return None
    for pattern in ("%B %d, %Y", "%Y-%m-%d", "%b %d, %Y"):
        try:
            return datetime.strptime(cleaned, pattern).date()
        except ValueError:
            continue
    return None


def _digits_or_none(value: str) -> str | None:
    digits = "".join(character for character in value if character.isdigit())
    return digits or None


__all__ = [
    "LIQUID_ETFS",
    "WIKIPEDIA_SP500_URL",
    "MembershipChange",
    "SP500Constituent",
    "UniverseSnapshot",
    "WikipediaSP500UniverseSource",
    "parse_wikipedia_sp500_html",
    "reconstruct_membership_intervals",
]
