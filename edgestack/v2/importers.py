"""Strict licensed-data importers and free diagnostic normalizers."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd

from edgestack.models import (
    AssetKey,
    Bar,
    CorporateEvent,
    CorporateEventKind,
    DataTier,
    EstimateVintage,
    IntradayMarketRecord,
    MarketRecordKind,
    MembershipInterval,
    TickerValidityInterval,
)
from edgestack.v2.provenance import PinnedDataset, load_hash_pinned


def import_pit_memberships(
    path: str | Path, expected_sha256: str
) -> tuple[MembershipInterval, ...]:
    """Import entitled membership intervals; malformed/overlapping data fail."""

    dataset = load_hash_pinned(path, expected_sha256)
    required = {"security_id", "ticker", "start", "end", "available_at"}
    _require(dataset, required)
    frame = dataset.frame.copy()
    frame["start"] = pd.to_datetime(frame["start"], errors="raise").dt.date
    frame["end"] = pd.to_datetime(frame["end"], errors="coerce").dt.date
    _validate_intervals(frame, "security_id", "start", "end")
    return tuple(
        MembershipInterval(
            asset=AssetKey(str(row.ticker), str(getattr(row, "exchange", "US"))),
            start=cast(date, row.start),
            end=None if pd.isna(row.end) else cast(date, row.end),
            sector=None if pd.isna(getattr(row, "sector", None)) else str(row.sector),
            available_at=_utc(row.available_at),
            security_id=str(row.security_id),
            source=str(row.source),
            data_tier=DataTier.POINT_IN_TIME,
            fetched_at=_utc(row.fetched_at),
            content_hash=str(row.content_hash),
        )
        for row in frame.itertuples(index=False)
    )


def import_ticker_history(
    path: str | Path, expected_sha256: str
) -> tuple[TickerValidityInterval, ...]:
    """Import permanent-ID to ticker validity intervals."""

    dataset = load_hash_pinned(path, expected_sha256)
    _require(dataset, {"security_id", "ticker", "exchange", "valid_from", "valid_to"})
    frame = dataset.frame.copy()
    frame["valid_from"] = pd.to_datetime(frame["valid_from"], utc=True)
    frame["valid_to"] = pd.to_datetime(frame["valid_to"], utc=True, errors="coerce")
    _validate_intervals(frame, "security_id", "valid_from", "valid_to")
    return tuple(
        TickerValidityInterval(
            str(row.security_id),
            str(row.ticker),
            str(row.exchange),
            _utc(row.valid_from),
            None if pd.isna(row.valid_to) else _utc(row.valid_to),
            _utc(row.available_at),
            str(row.source),
            _utc(row.fetched_at),
            str(row.content_hash),
        )
        for row in frame.itertuples(index=False)
    )


def import_estimate_vintages(
    path: str | Path, expected_sha256: str
) -> tuple[EstimateVintage, ...]:
    """Import historical estimates without collapsing later revisions."""

    dataset = load_hash_pinned(path, expected_sha256)
    _require(dataset, {"estimate_id", "security_id", "metric", "period_end", "value"})
    rows = dataset.frame.sort_values(["estimate_id", "available_at", "revision"])
    if bool(rows.duplicated(["estimate_id", "revision"]).any()):
        raise ValueError("duplicate estimate revision")
    return tuple(
        EstimateVintage(
            str(row.estimate_id),
            str(row.security_id),
            str(row.metric),
            date.fromisoformat(str(row.period_end)[:10]),
            float(cast(Any, row.value)),
            _utc(row.event_time),
            _utc(row.available_at),
            str(row.revision),
            str(row.source),
            _utc(row.fetched_at),
            str(row.content_hash),
        )
        for row in rows.itertuples(index=False)
    )


def import_intraday(
    path: str | Path, expected_sha256: str
) -> tuple[IntradayMarketRecord, ...]:
    """Import normalized minute/NBBO/trade/imbalance/auction records."""

    dataset = load_hash_pinned(path, expected_sha256)
    _require(dataset, {"security_id", "kind"})
    records: list[IntradayMarketRecord] = []
    for row in dataset.frame.itertuples(index=False):
        kind = MarketRecordKind(str(row.kind))
        records.append(
            IntradayMarketRecord(
                security_id=str(row.security_id),
                kind=kind,
                event_time=_utc(row.event_time),
                available_at=_utc(row.available_at),
                source=str(row.source),
                revision=str(row.revision),
                fetched_at=_utc(row.fetched_at),
                content_hash=str(row.content_hash),
                price=_optional_float(getattr(row, "price", None)),
                size=_optional_float(getattr(row, "size", None)),
                bid=_optional_float(getattr(row, "bid", None)),
                ask=_optional_float(getattr(row, "ask", None)),
            )
        )
    return tuple(records)


def bounded_intraday_storage(
    records: tuple[IntradayMarketRecord, ...],
    *,
    finalist_security_ids: frozenset[str],
) -> tuple[IntradayMarketRecord, ...]:
    """Keep full-session minute bars and finalist ticks only from 15:15-16:05 ET."""

    eastern = ZoneInfo("America/New_York")
    retained: list[IntradayMarketRecord] = []
    for record in records:
        local_time = record.event_time.astimezone(eastern).time()
        if record.kind is MarketRecordKind.MINUTE_BAR:
            if (
                datetime.strptime("09:30", "%H:%M").time()
                <= local_time
                <= datetime.strptime("16:00", "%H:%M").time()
            ):
                retained.append(record)
            continue
        if record.security_id not in finalist_security_ids:
            continue
        if (
            datetime.strptime("15:15", "%H:%M").time()
            <= local_time
            <= datetime.strptime("16:05", "%H:%M").time()
        ):
            retained.append(record)
    return tuple(retained)


def corporate_actions_as_events(
    bars: tuple[Bar, ...], *, fetched_at: datetime
) -> tuple[CorporateEvent, ...]:
    """Normalize existing dividends/splits into the shared event schema."""

    events: list[CorporateEvent] = []
    for bar in bars:
        for kind, value in (
            (CorporateEventKind.DIVIDEND, bar.dividend),
            (
                CorporateEventKind.SPLIT,
                bar.split_factor if bar.split_factor != 1 else 0,
            ),
        ):
            if value == 0:
                continue
            identity = (
                f"{bar.asset.symbol}:{kind.value}:{bar.event_time.isoformat()}:{value}"
            )
            digest = hashlib.sha256(identity.encode()).hexdigest()
            events.append(
                CorporateEvent(
                    event_id=digest[:24],
                    security_id=bar.asset.symbol,
                    kind=kind,
                    event_time=bar.event_time,
                    available_at=bar.available_at,
                    source=bar.source,
                    revision="original",
                    fetched_at=fetched_at.astimezone(UTC),
                    content_hash=digest,
                    metadata={"value": float(value)},
                )
            )
    return tuple(events)


def _require(dataset: PinnedDataset, columns: set[str]) -> None:
    missing = columns.difference(dataset.frame.columns)
    if missing:
        raise ValueError(f"missing dataset columns: {sorted(missing)}")


def _validate_intervals(frame: pd.DataFrame, key: str, start: str, end: str) -> None:
    for identity, group in frame.sort_values([key, start]).groupby(key):
        previous_end: object | None = None
        seen_open = False
        for row in group.itertuples(index=False):
            current_start = getattr(row, start)
            current_end = getattr(row, end)
            if not pd.isna(current_end) and current_end <= current_start:
                raise ValueError(f"invalid interval for {identity}")
            if seen_open or (previous_end is not None and current_start < previous_end):
                raise ValueError(f"overlapping intervals for {identity}")
            seen_open = bool(pd.isna(current_end))
            previous_end = None if seen_open else current_end


def _utc(value: object) -> datetime:
    stamp = pd.Timestamp(cast(Any, value))
    if stamp.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return stamp.tz_convert("UTC").to_pydatetime()


def _optional_float(value: object) -> float | None:
    return (
        None
        if value is None or bool(pd.isna(cast(Any, value)))
        else float(cast(Any, value))
    )
