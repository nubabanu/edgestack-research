"""Free, provenance-retaining oil references with causal publication times."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from io import BytesIO, StringIO
from typing import Any, Final
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from edgestack.data.factors import FredSeriesSpec, ReferenceBatch
from edgestack.data.sources import RawPayload, RawPayloadSink

NEW_YORK: Final = ZoneInfo("America/New_York")
EIA_WPSR_BASE: Final = "https://ir.eia.gov/wpsr"
CFTC_COMBINED_API: Final = (
    "https://publicreporting.cftc.gov/resource/kh3c-gbw2.json"
)
WTI_CFTC_CODE: Final = "067651"

# Keyed by the nominal Wednesday. Values are the actual holiday release.
EIA_2026_RELEASE_OVERRIDES: Final[Mapping[date, datetime]] = {
    date(2026, 1, 21): datetime(2026, 1, 22, 12, 0, tzinfo=NEW_YORK),
    date(2026, 2, 18): datetime(2026, 2, 19, 12, 0, tzinfo=NEW_YORK),
    date(2026, 5, 27): datetime(2026, 5, 28, 12, 0, tzinfo=NEW_YORK),
    date(2026, 9, 9): datetime(2026, 9, 10, 12, 0, tzinfo=NEW_YORK),
    date(2026, 10, 14): datetime(2026, 10, 15, 12, 0, tzinfo=NEW_YORK),
    date(2026, 11, 11): datetime(2026, 11, 12, 12, 0, tzinfo=NEW_YORK),
}


@dataclass(frozen=True, slots=True)
class OilReferenceBatch:
    """Normalized free reference data backed by exact response bytes."""

    kind: str
    frame: pd.DataFrame = field(repr=False)
    fetched_at: datetime
    raw_sha256: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fetched_at.tzinfo is None:
            raise ValueError("oil reference fetched_at must be timezone-aware")
        object.__setattr__(self, "frame", self.frame.copy(deep=True))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class SignedPriceObservation:
    """Commodity reference value that may legitimately be zero or negative."""

    series_id: str
    session: date
    event_time: datetime
    available_at: datetime
    value: float
    source: str
    raw_sha256: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("signed oil price must be finite")
        if self.event_time.tzinfo is None or self.available_at.tzinfo is None:
            raise ValueError("signed oil timestamps must be timezone-aware")
        if self.available_at <= self.event_time:
            raise ValueError("signed oil price availability must follow event time")


OIL_FRED_SPECS: Final = (
    # EIA spot observations are often published in a later weekly batch. Seven
    # days is deliberately conservative when exact vintage snapshots are absent.
    FredSeriesSpec("DCOILWTICO", availability_lag=timedelta(days=7), revised=True),
    FredSeriesSpec("DCOILBRENTEU", availability_lag=timedelta(days=7), revised=True),
    FredSeriesSpec("OVXCLS", availability_lag=timedelta(days=1), revised=False),
    FredSeriesSpec("DTWEXBGS", availability_lag=timedelta(days=1), revised=True),
)
EIA_HISTORY_SERIES: Final[Mapping[str, str]] = {
    "WCESTUS1": "Weekly U.S. commercial crude stocks excluding SPR",
    "W_EPC0_SAX_YCUOK_MBBL": "Weekly Cushing commercial crude stocks",
}


def eia_release_at(nominal_wednesday: date) -> datetime:
    """Return the configured actual WPSR release timestamp."""

    if nominal_wednesday.weekday() != 2:
        raise ValueError("nominal EIA release must be a Wednesday")
    override = EIA_2026_RELEASE_OVERRIDES.get(nominal_wednesday)
    if override is not None:
        return override
    return datetime.combine(nominal_wednesday, time(10, 30), tzinfo=NEW_YORK)


def latest_eia_release_at(moment: datetime) -> datetime:
    """Latest WPSR publication known at a timezone-aware decision moment."""

    if moment.tzinfo is None:
        raise ValueError("decision moment must be timezone-aware")
    local = moment.astimezone(NEW_YORK)
    candidate = local.date() - timedelta(days=(local.weekday() - 2) % 7)
    release = eia_release_at(candidate)
    if release > local:
        candidate -= timedelta(days=7)
        release = eia_release_at(candidate)
    return release.astimezone(UTC)


def cftc_release_at(
    report_date: date,
    *,
    overrides: Mapping[date, datetime] | None = None,
) -> datetime:
    """Causal Friday 15:30 ET availability for a Tuesday COT observation."""

    if overrides and report_date in overrides:
        value = overrides[report_date]
        if value.tzinfo is None:
            raise ValueError("CFTC release override must be timezone-aware")
        return value.astimezone(UTC)
    days = (4 - report_date.weekday()) % 7
    if days == 0:
        days = 7
    release_day = report_date + timedelta(days=days)
    return datetime.combine(release_day, time(15, 30), tzinfo=NEW_YORK).astimezone(
        UTC
    )


def parse_eia_wpsr_csv(
    body: bytes,
    *,
    table_id: str,
    published_at: datetime,
) -> pd.DataFrame:
    """Parse one released WPSR table without inferring pre-release knowledge."""

    if published_at.tzinfo is None:
        raise ValueError("EIA publication time must be timezone-aware")
    try:
        decoded = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        decoded = body.decode("cp1252")

    def slug(value: object) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
        return normalized.strip("_") or "unnamed"

    raw_rows = [
        [cell.strip().strip("\x1a") for cell in row]
        for row in csv.reader(StringIO(decoded))
        if row and any(cell.strip().strip("\x1a") for cell in row)
    ]
    if not raw_rows:
        raise ValueError(f"EIA {table_id} CSV is empty")
    normalized: list[dict[str, Any]] = []
    header: list[str] | None = None
    section = -1
    for raw_row in raw_rows:
        begins_section = raw_row[0].strip().upper() == "STUB_1"
        if header is None or begins_section:
            section += 1
            seen: dict[str, int] = {}
            header = []
            for value in raw_row:
                base = slug(value)
                occurrence = seen.get(base, 0) + 1
                seen[base] = occurrence
                header.append(base if occurrence == 1 else f"{base}_{occurrence}")
            continue
        padded = raw_row + [""] * max(0, len(header) - len(raw_row))
        record: dict[str, Any] = {
            column: value for column, value in zip(header, padded, strict=False)
        }
        record["section"] = section
        normalized.append(record)
    if not normalized:
        raise ValueError(f"EIA {table_id} CSV contains no data rows")
    table = pd.DataFrame(normalized)
    for column in table.columns:
        if table[column].dtype == object:
            cleaned = table[column].astype(str).str.replace(",", "", regex=False)
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if int(numeric.notna().sum()) >= max(1, len(table) // 2):
                table[column] = numeric
    table.insert(0, "table_id", table_id)
    table["available_at"] = pd.Timestamp(published_at).tz_convert(UTC)
    return table


def parse_eia_history_xls(
    body: bytes,
    *,
    series_id: str,
) -> pd.DataFrame:
    """Parse an official no-key EIA weekly history workbook causally."""

    if series_id not in EIA_HISTORY_SERIES:
        raise ValueError("undeclared EIA historical series")
    raw = pd.read_excel(BytesIO(body), sheet_name="Data 1", header=None, engine="xlrd")
    if len(raw) < 4 or str(raw.iloc[1, 1]).strip() != series_id:
        raise ValueError(f"EIA history workbook identity mismatch for {series_id}")
    values = raw.iloc[3:, :2].copy()
    values.columns = ["session", series_id]
    values["session"] = pd.to_datetime(values["session"], errors="coerce").dt.normalize()
    values[series_id] = pd.to_numeric(values[series_id], errors="coerce")
    values = values.dropna().reset_index(drop=True)
    if values.empty:
        raise ValueError(f"EIA history workbook has no {series_id} observations")
    event_times: list[datetime] = []
    availability: list[datetime] = []
    for session in values["session"]:
        report_date = pd.Timestamp(session).date()
        event_times.append(
            datetime.combine(report_date, time(16), tzinfo=NEW_YORK).astimezone(UTC)
        )
        nominal_wednesday = report_date + timedelta(
            days=(2 - report_date.weekday()) % 7
        )
        if nominal_wednesday <= report_date:
            nominal_wednesday += timedelta(days=7)
        availability.append(eia_release_at(nominal_wednesday).astimezone(UTC))
    values[f"{series_id}__event_time"] = event_times
    values[f"{series_id}__available_at"] = availability
    return values


def parse_cftc_cot_json(
    body: bytes,
    *,
    release_overrides: Mapping[date, datetime] | None = None,
) -> pd.DataFrame:
    """Normalize WTI managed-money positioning from CFTC's public API."""

    payload = json.loads(body)
    if not isinstance(payload, list) or not payload:
        raise ValueError("CFTC response contains no records")
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        code = str(item.get("cftc_contract_market_code", "")).strip()
        market = str(item.get("market_and_exchange_names", ""))
        if code and code != WTI_CFTC_CODE:
            continue
        if "CRUDE OIL" not in market.upper() and not code:
            continue
        raw_date = item.get("report_date_as_yyyy_mm_dd") or item.get("report_date")
        if raw_date is None:
            continue
        report = pd.Timestamp(str(raw_date)).date()

        def number(record: Mapping[str, Any], name: str) -> float:
            try:
                value = float(str(record.get(name, "")).replace(",", ""))
            except (TypeError, ValueError):
                return math.nan
            return value if math.isfinite(value) else math.nan

        long = number(item, "m_money_positions_long_all")
        short = number(item, "m_money_positions_short_all")
        open_interest = number(item, "open_interest_all")
        rows.append(
            {
                "report_date": pd.Timestamp(report),
                "event_time": datetime.combine(
                    report, time(15, 0), tzinfo=NEW_YORK
                ).astimezone(UTC),
                "available_at": cftc_release_at(
                    report, overrides=release_overrides
                ),
                "market": market,
                "contract_code": code or WTI_CFTC_CODE,
                "managed_money_long": long,
                "managed_money_short": short,
                "managed_money_net": long - short,
                "managed_money_net_fraction_oi": (
                    (long - short) / open_interest
                    if math.isfinite(open_interest) and open_interest > 0
                    else math.nan
                ),
                "open_interest": open_interest,
            }
        )
    if not rows:
        raise ValueError("CFTC response has no WTI positioning records")
    result = pd.DataFrame(rows).sort_values("report_date", kind="stable")
    if result["report_date"].duplicated().any():
        # The filtered view can contain exchange variants; retain the largest
        # open-interest WTI record per report date deterministically.
        result = (
            result.sort_values(
                ["report_date", "open_interest"], ascending=[True, False]
            )
            .drop_duplicates("report_date", keep="first")
            .sort_values("report_date", kind="stable")
        )
    return result.reset_index(drop=True)


def signed_price_observations(
    batch: ReferenceBatch,
    *,
    series_ids: Sequence[str] = ("DCOILWTICO", "DCOILBRENTEU"),
) -> tuple[SignedPriceObservation, ...]:
    """Convert FRED spot values without applying the equity-price invariant."""

    output: list[SignedPriceObservation] = []
    for series_id in series_ids:
        if series_id not in batch.frame:
            continue
        digest = batch.raw_sha256[
            list(batch.metadata.get("series", series_ids)).index(series_id)
        ]
        rows = batch.frame.loc[batch.frame[series_id].notna()]
        for _, row in rows.iterrows():
            value = float(row[series_id])
            output.append(
                SignedPriceObservation(
                    series_id=series_id,
                    session=pd.Timestamp(row["session"]).date(),
                    event_time=pd.Timestamp(
                        row[f"{series_id}__event_time"]
                    ).to_pydatetime(),
                    available_at=pd.Timestamp(
                        row[f"{series_id}__available_at"]
                    ).to_pydatetime(),
                    value=value,
                    source="fred/eia",
                    raw_sha256=digest,
                )
            )
    return tuple(output)


class _OilHttpSource:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        raw_sink: RawPayloadSink | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.client = client
        self.raw_sink = raw_sink
        self.timeout = timeout

    async def get(self, url: str, *, params: Mapping[str, str] | None = None) -> tuple[bytes, datetime, str]:
        owns = self.client is None
        client = self.client or httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": "EdgeStack/0.1 paper oil research"},
        )
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
        finally:
            if owns:
                await client.aclose()
        fetched_at = datetime.now(UTC)
        digest = hashlib.sha256(response.content).hexdigest()
        if self.raw_sink is not None:
            stored = self.raw_sink.store(
                RawPayload(
                    source="oil_reference",
                    asset=None,
                    fetched_at=fetched_at,
                    media_type=response.headers.get(
                        "content-type", "application/octet-stream"
                    ).split(";")[0],
                    body=response.content,
                    request_url=str(response.request.url),
                    status_code=response.status_code,
                )
            )
            if stored != digest:
                raise RuntimeError("oil raw-payload sink returned the wrong hash")
        return response.content, fetched_at, digest


class EiaWpsrSource:
    """Current official WPSR CSV tables with caller-bound publication time."""

    def __init__(self, **kwargs: Any) -> None:
        self._http = _OilHttpSource(**kwargs)

    async def fetch_table(
        self, table_id: str, *, published_at: datetime
    ) -> OilReferenceBatch:
        normalized = table_id.strip().lower()
        if not re.fullmatch(r"table(?:1|4|9|11)", normalized):
            raise ValueError("oil WPSR table must be one of table1/table4/table9/table11")
        url = f"{EIA_WPSR_BASE}/{normalized}.csv"
        body, fetched_at, digest = await self._http.get(url)
        frame = parse_eia_wpsr_csv(
            body, table_id=normalized, published_at=published_at
        )
        return OilReferenceBatch(
            "eia_wpsr",
            frame,
            fetched_at,
            (digest,),
            metadata={
                "table_id": normalized,
                "url": url,
                "published_at": published_at.astimezone(UTC).isoformat(),
            },
        )


class EiaHistorySource:
    """Official no-key EIA XLS histories for preregistered stock features."""

    def __init__(self, **kwargs: Any) -> None:
        self._http = _OilHttpSource(**kwargs)

    async def fetch_series(
        self,
        series_ids: Sequence[str] = tuple(EIA_HISTORY_SERIES),
    ) -> OilReferenceBatch:
        frames: list[pd.DataFrame] = []
        hashes: list[str] = []
        fetched_times: list[datetime] = []
        for series_id in series_ids:
            if series_id not in EIA_HISTORY_SERIES:
                raise ValueError(f"undeclared EIA history series {series_id}")
            url = f"https://www.eia.gov/dnav/pet/hist_xls/{series_id}w.xls"
            body, fetched_at, digest = await self._http.get(url)
            frames.append(parse_eia_history_xls(body, series_id=series_id))
            hashes.append(digest)
            fetched_times.append(fetched_at)
        combined = frames[0]
        for frame in frames[1:]:
            combined = combined.merge(
                frame, on="session", how="outer", validate="one_to_one"
            )
        return OilReferenceBatch(
            "eia_weekly_history",
            combined.sort_values("session", kind="stable").reset_index(drop=True),
            max(fetched_times),
            tuple(hashes),
            warnings=(
                "EIA weekly histories are current revised workbooks; availability "
                "uses the declared WPSR release calendar.",
            ),
            metadata={
                "series": list(series_ids),
                "descriptions": {
                    key: EIA_HISTORY_SERIES[key] for key in series_ids
                },
            },
        )
class CftcCotSource:
    """No-key CFTC public-reporting adapter for WTI positioning."""

    def __init__(self, **kwargs: Any) -> None:
        self._http = _OilHttpSource(**kwargs)

    async def fetch_wti(
        self,
        *,
        start: date | None = None,
        end: date | None = None,
        release_overrides: Mapping[date, datetime] | None = None,
    ) -> OilReferenceBatch:
        clauses = [f"cftc_contract_market_code='{WTI_CFTC_CODE}'"]
        if start is not None:
            clauses.append(
                f"report_date_as_yyyy_mm_dd >= '{start.isoformat()}T00:00:00.000'"
            )
        if end is not None:
            clauses.append(
                f"report_date_as_yyyy_mm_dd <= '{end.isoformat()}T23:59:59.999'"
            )
        params = {
            "$where": " AND ".join(clauses),
            "$order": "report_date_as_yyyy_mm_dd ASC",
            "$limit": "5000",
        }
        body, fetched_at, digest = await self._http.get(
            CFTC_COMBINED_API, params=params
        )
        frame = parse_cftc_cot_json(
            body, release_overrides=release_overrides
        )
        return OilReferenceBatch(
            "cftc_wti_disaggregated_combined",
            frame,
            fetched_at,
            (digest,),
            metadata={"url": CFTC_COMBINED_API, "contract_code": WTI_CFTC_CODE},
        )


__all__ = [
    "CFTC_COMBINED_API",
    "EIA_2026_RELEASE_OVERRIDES",
    "EIA_HISTORY_SERIES",
    "OIL_FRED_SPECS",
    "WTI_CFTC_CODE",
    "CftcCotSource",
    "EiaHistorySource",
    "EiaWpsrSource",
    "OilReferenceBatch",
    "SignedPriceObservation",
    "cftc_release_at",
    "eia_release_at",
    "latest_eia_release_at",
    "parse_cftc_cot_json",
    "parse_eia_history_xls",
    "parse_eia_wpsr_csv",
    "signed_price_observations",
]
