"""SEC filing-time diagnostics for event-risk candidates."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, cast

import httpx

from edgestack.models import CorporateEvent, CorporateEventKind


def events_from_sec_submissions(
    payload: dict[str, Any],
    *,
    security_id: str,
    fetched_at: datetime,
) -> tuple[CorporateEvent, ...]:
    """Normalize exact SEC acceptance times; date-only rows are fail-closed."""

    if fetched_at.tzinfo is None:
        raise ValueError("fetched_at must be timezone-aware")
    recent = cast(
        dict[str, list[Any]],
        cast(dict[str, Any], payload.get("filings", {})).get("recent", {}),
    )
    accessions = recent.get("accessionNumber", [])
    raw_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    events: list[CorporateEvent] = []
    for index, accession in enumerate(accessions):
        accepted = _at(recent, "acceptanceDateTime", index)
        if not accepted:
            continue
        event_time = _parse_sec_time(str(accepted))
        form = str(_at(recent, "form", index) or "")
        items = str(_at(recent, "items", index) or "")
        primary = str(_at(recent, "primaryDocument", index) or "")
        kind = _classify(form, items, primary)
        if kind is None:
            continue
        identity = f"sec:{accession}:{kind.value}"
        events.append(
            CorporateEvent(
                event_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
                security_id=security_id,
                kind=kind,
                event_time=event_time,
                available_at=event_time,
                source="SEC_EDGAR_SUBMISSIONS",
                revision=str(accession),
                fetched_at=fetched_at.astimezone(UTC),
                content_hash=raw_hash,
                metadata={
                    "form": form,
                    "items": items,
                    "primary_document": primary,
                    "diagnostic_candidate": True,
                },
            )
        )
    return tuple(events)


class SecFilingEventSource:
    """Free SEC filing diagnostic; it does not satisfy estimate-vintage gates."""

    def __init__(self, *, user_agent: str, cik_by_security: dict[str, str]) -> None:
        if "@" not in user_agent:
            raise ValueError("SEC user agent must include a contact email")
        self.user_agent = user_agent
        self.cik_by_security = dict(cik_by_security)

    async def fetch_events(
        self,
        security_ids: list[str] | tuple[str, ...],
        start: datetime,
        end: datetime,
    ) -> tuple[CorporateEvent, ...]:
        """Fetch recent submissions and retain only requested acceptance times."""

        if start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError("valid timezone-aware bounds are required")
        fetched_at = datetime.now(UTC)
        output: list[CorporateEvent] = []
        async with httpx.AsyncClient(
            headers={"User-Agent": self.user_agent}, timeout=30, follow_redirects=True
        ) as client:
            for security_id in security_ids:
                cik = self.cik_by_security.get(security_id)
                if cik is None:
                    continue
                response = await client.get(
                    f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
                )
                response.raise_for_status()
                rows = events_from_sec_submissions(
                    response.json(), security_id=security_id, fetched_at=fetched_at
                )
                output.extend(
                    row
                    for row in rows
                    if start <= row.event_time.astimezone(start.tzinfo) <= end
                )
        return tuple(output)


def _classify(
    form: str, items: str, primary_document: str
) -> CorporateEventKind | None:
    normalized_items = {item.strip() for item in items.split(",")}
    document = primary_document.lower()
    if form == "8-K" and "2.02" in normalized_items:
        return CorporateEventKind.PRELIMINARY_RESULTS
    if form == "8-K" and normalized_items.intersection({"7.01", "8.01"}):
        return CorporateEventKind.GUIDANCE
    if form in {"10-Q", "10-K", "6-K", "20-F"} or "earnings" in document:
        return CorporateEventKind.EARNINGS
    return None


def _parse_sec_time(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _at(values: dict[str, list[Any]], key: str, index: int) -> Any | None:
    items = values.get(key, [])
    return items[index] if index < len(items) else None
