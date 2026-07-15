"""No-key Ken French factors and public FRED CSV macro observations.

Both adapters retain exact response bytes through ``RawPayloadSink`` and return
availability-aware frames.  The public FRED graph endpoint exposes the latest
vintage, not an ALFRED point-in-time vintage; revised macro series are therefore
explicitly warned and must not silently become historical trading features.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import tempfile
import time as time_module
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from edgestack.data.sources import RawPayload, RawPayloadSink

NEW_YORK: Final = ZoneInfo("America/New_York")
KEN_FRENCH_DAILY_URL: Final = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
FRED_CSV_URL: Final = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_REFERENCE_SCHEMA_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class ReferenceBatch:
    """Normalized reference dataset backed by exact raw response hashes."""

    kind: str
    frame: pd.DataFrame = field(repr=False)
    fetched_at: datetime
    raw_sha256: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fetched_at.tzinfo is None:
            raise ValueError("ReferenceBatch.fetched_at must be timezone-aware")
        object.__setattr__(self, "frame", self.frame.copy(deep=True))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class FredSeriesSpec:
    """FRED series and conservative historical availability policy."""

    series_id: str
    availability_lag: timedelta = timedelta(days=1)
    revised: bool = True

    def __post_init__(self) -> None:
        normalized = self.series_id.strip().upper()
        if not normalized:
            raise ValueError("FRED series_id cannot be empty")
        if self.availability_lag < timedelta(0):
            raise ValueError("availability_lag cannot be negative")
        object.__setattr__(self, "series_id", normalized)


VIXCLS: Final = FredSeriesSpec(
    "VIXCLS", availability_lag=timedelta(minutes=15), revised=False
)
DGS10: Final = FredSeriesSpec(
    "DGS10", availability_lag=timedelta(days=1), revised=False
)
DGS2: Final = FredSeriesSpec("DGS2", availability_lag=timedelta(days=1), revised=False)
FEDFUNDS: Final = FredSeriesSpec(
    "FEDFUNDS", availability_lag=timedelta(days=35), revised=True
)


class _ReferenceHTTP:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None,
        raw_sink: RawPayloadSink | None,
        timeout: float,
        max_attempts: int,
    ) -> None:
        self.client = client
        self.raw_sink = raw_sink
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)

    async def get(
        self, url: str, *, params: Mapping[str, str] | None = None
    ) -> tuple[httpx.Response, datetime, str]:
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "EdgeStack/0.1 research and educational data client"
            },
        )
        error: Exception | None = None
        try:
            for attempt in range(self.max_attempts):
                try:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    break
                except (httpx.HTTPError, httpx.TimeoutException) as exc:
                    error = exc
                    if attempt + 1 == self.max_attempts:
                        raise
                    await asyncio.sleep(min(8.0, 0.5 * 2**attempt))
            else:  # pragma: no cover - loop always returns or raises
                raise RuntimeError("unreachable HTTP retry state") from error
        finally:
            if owns_client:
                await client.aclose()
        fetched_at = datetime.now(UTC)
        digest = hashlib.sha256(response.content).hexdigest()
        if self.raw_sink is not None:
            public_url = f"{url}?{urlencode(params)}" if params else url
            stored = self.raw_sink.store(
                RawPayload(
                    source="reference",
                    asset=None,
                    fetched_at=fetched_at,
                    media_type=response.headers.get(
                        "content-type", "application/octet-stream"
                    ).split(";")[0],
                    body=response.content,
                    request_url=public_url,
                    status_code=response.status_code,
                    response_headers={
                        key.lower(): value
                        for key, value in response.headers.items()
                        if key.lower()
                        in {"content-type", "etag", "last-modified", "date"}
                    },
                )
            )
            if stored != digest:
                raise RuntimeError("raw reference sink returned the wrong hash")
        return response, fetched_at, digest


class KenFrenchDailyFactorsSource:
    """Daily Fama/French three factors and value-weighted market return."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        raw_sink: RawPayloadSink | None = None,
        timeout: float = 30.0,
        max_attempts: int = 4,
    ) -> None:
        self._http = _ReferenceHTTP(
            client=client,
            raw_sink=raw_sink,
            timeout=timeout,
            max_attempts=max_attempts,
        )

    async def fetch_factors(self, start: date, end: date) -> ReferenceBatch:
        """Download and parse factors for the inclusive date interval.

        French-library values are percentages and are converted to decimal returns.
        ``market_return`` equals ``mkt_rf + rf``; using ``mkt_rf`` alone for a
        turn-of-month total-market replication would incorrectly omit cash return.
        """

        if end < start:
            raise ValueError("end must be on or after start")
        response, fetched_at, digest = await self._http.get(KEN_FRENCH_DAILY_URL)
        frame = parse_ken_french_daily_zip(response.content)
        frame = frame.loc[
            frame["session"].between(pd.Timestamp(start), pd.Timestamp(end))
        ].reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                "Ken French response has no observations in requested interval"
            )
        return ReferenceBatch(
            "ken_french_daily_factors",
            frame,
            fetched_at,
            (digest,),
            (
                "Ken French files are current research-library revisions, not "
                "historical publication vintages; use for replication benchmarks only.",
            ),
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "url": KEN_FRENCH_DAILY_URL,
            },
        )

    async def fetch(self, start: date, end: date) -> ReferenceBatch:
        """Alias for :meth:`fetch_factors` used by generic reference pipelines."""

        return await self.fetch_factors(start, end)


class FREDCSVSource:
    """No-key latest-vintage series from FRED's public graph CSV endpoint."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        raw_sink: RawPayloadSink | None = None,
        timeout: float = 30.0,
        max_attempts: int = 4,
    ) -> None:
        self._http = _ReferenceHTTP(
            client=client,
            raw_sink=raw_sink,
            timeout=timeout,
            max_attempts=max_attempts,
        )

    async def fetch_series(
        self,
        specs: Sequence[FredSeriesSpec | str],
        start: date,
        end: date,
    ) -> ReferenceBatch:
        """Fetch and outer-join one or more public FRED CSV series."""

        if end < start:
            raise ValueError("end must be on or after start")
        normalized = tuple(
            spec if isinstance(spec, FredSeriesSpec) else FredSeriesSpec(spec)
            for spec in specs
        )
        if not normalized:
            raise ValueError("at least one FRED series is required")
        frames: list[pd.DataFrame] = []
        hashes: list[str] = []
        fetched_times: list[datetime] = []
        for spec in normalized:
            response, fetched_at, digest = await self._http.get(
                FRED_CSV_URL,
                params={
                    "id": spec.series_id,
                    "cosd": start.isoformat(),
                    "coed": end.isoformat(),
                },
            )
            frame = parse_fred_csv(
                response.text,
                spec,
                start=start,
                end=end,
            )
            frames.append(frame)
            hashes.append(digest)
            fetched_times.append(fetched_at)
        combined = frames[0]
        for frame in frames[1:]:
            combined = combined.merge(
                frame, on="session", how="outer", validate="one_to_one"
            )
        combined = combined.sort_values("session", kind="stable").reset_index(drop=True)
        warnings = []
        if any(spec.revised for spec in normalized):
            warnings.append(
                "LATEST_VINTAGE_NOT_POINT_IN_TIME: one or more FRED series may be "
                "revised; configure ALFRED vintages before using them as historical signals."
            )
        return ReferenceBatch(
            "fred_csv",
            combined,
            max(fetched_times),
            tuple(hashes),
            tuple(warnings),
            {
                "series": [spec.series_id for spec in normalized],
                "start": start.isoformat(),
                "end": end.isoformat(),
                "url": FRED_CSV_URL,
            },
        )

    async def fetch(
        self,
        series_ids: Sequence[str],
        start: date,
        end: date,
    ) -> ReferenceBatch:
        """Generic helper using conservative one-day/latest-vintage policies."""

        return await self.fetch_series(series_ids, start, end)


class ReferenceDataCache:
    """Content-addressed normalized Parquet cache for reference datasets."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def store(self, batch: ReferenceBatch) -> str:
        """Write normalized Parquet/manifest once and return the snapshot ID."""

        identity = {
            "schema_version": _REFERENCE_SCHEMA_VERSION,
            "kind": batch.kind,
            "raw_sha256": list(batch.raw_sha256),
            "metadata": batch.metadata,
        }
        snapshot_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        directory = self.root / batch.kind / snapshot_id
        parquet_path = directory / "data.parquet"
        manifest_path = directory / "manifest.json"
        if directory.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("snapshot_id") != snapshot_id:
                raise RuntimeError("reference cache identity mismatch")
            if _sha256_file(parquet_path) != manifest["parquet_sha256"]:
                raise RuntimeError("reference cache Parquet hash mismatch")
            return snapshot_id
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=directory.parent)
        )
        try:
            temporary_parquet = temporary / "data.parquet"
            batch.frame.to_parquet(temporary_parquet, index=False, compression="zstd")
            manifest = {
                **identity,
                "snapshot_id": snapshot_id,
                "fetched_at": batch.fetched_at.astimezone(UTC).isoformat(),
                "warnings": list(batch.warnings),
                "rows": len(batch.frame),
                "parquet_sha256": _sha256_file(temporary_parquet),
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            _install_directory(temporary, directory)
        finally:
            if temporary.exists():
                for child in temporary.iterdir():
                    child.unlink()
                temporary.rmdir()
        return snapshot_id

    def load(self, kind: str, snapshot_id: str) -> ReferenceBatch:
        """Verify and load one cached reference batch."""

        directory = self.root / kind / snapshot_id
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        parquet_path = directory / "data.parquet"
        if manifest.get("snapshot_id") != snapshot_id:
            raise RuntimeError("reference cache manifest identity mismatch")
        if _sha256_file(parquet_path) != manifest["parquet_sha256"]:
            raise RuntimeError("reference cache Parquet hash mismatch")
        return ReferenceBatch(
            kind,
            pd.read_parquet(parquet_path),
            datetime.fromisoformat(manifest["fetched_at"]),
            tuple(manifest["raw_sha256"]),
            tuple(manifest["warnings"]),
            manifest["metadata"],
        )


def parse_ken_french_daily_zip(body: bytes) -> pd.DataFrame:
    """Parse the official zipped CSV, converting percent values to decimals."""

    try:
        with zipfile.ZipFile(BytesIO(body)) as archive:
            csv_names = [
                name for name in archive.namelist() if name.lower().endswith(".csv")
            ]
            if not csv_names:
                raise ValueError("Ken French ZIP contains no CSV")
            text = archive.read(csv_names[0]).decode("utf-8-sig", errors="replace")
    except zipfile.BadZipFile as error:
        raise ValueError("invalid Ken French ZIP response") from error
    reader = csv.reader(StringIO(text))
    rows: list[dict[str, Any]] = []
    for fields in reader:
        if (
            len(fields) < 5
            or not fields[0].strip().isdigit()
            or len(fields[0].strip()) != 8
        ):
            continue
        try:
            session = datetime.strptime(fields[0].strip(), "%Y%m%d").date()
            values = [float(field.strip()) / 100.0 for field in fields[1:5]]
        except ValueError:
            continue
        event = datetime.combine(session, time(16), tzinfo=NEW_YORK).astimezone(UTC)
        rows.append(
            {
                "session": pd.Timestamp(session),
                "event_time": event,
                # Publication time is not provided; next morning is conservative.
                "available_at": datetime.combine(
                    session + timedelta(days=1), time(8), tzinfo=NEW_YORK
                ).astimezone(UTC),
                "mkt_rf": values[0],
                "smb": values[1],
                "hml": values[2],
                "rf": values[3],
                "market_return": values[0] + values[3],
            }
        )
    if not rows:
        raise ValueError("Ken French CSV contains no daily factor rows")
    return (
        pd.DataFrame(rows).sort_values("session", kind="stable").reset_index(drop=True)
    )


def parse_fred_csv(
    text: str,
    spec: FredSeriesSpec,
    *,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Parse a one-series FRED graph CSV with causal availability timestamps."""

    table = pd.read_csv(StringIO(text))
    if table.shape[1] < 2:
        raise ValueError(f"FRED CSV for {spec.series_id} has no value column")
    date_column = (
        "observation_date" if "observation_date" in table else table.columns[0]
    )
    value_column = spec.series_id if spec.series_id in table else table.columns[1]
    sessions = pd.to_datetime(table[date_column], errors="coerce").dt.normalize()
    values = pd.to_numeric(table[value_column].replace(".", pd.NA), errors="coerce")
    result = pd.DataFrame({"session": sessions, spec.series_id: values}).dropna(
        subset=["session"]
    )
    result = result.loc[
        result["session"].between(pd.Timestamp(start), pd.Timestamp(end))
    ].copy()
    if result.empty:
        raise ValueError(f"FRED CSV for {spec.series_id} has no requested observations")
    local_close = result["session"].map(
        lambda session: datetime.combine(session.date(), time(16), tzinfo=NEW_YORK)
    )
    result[f"{spec.series_id}__event_time"] = pd.to_datetime(local_close, utc=True)
    result[f"{spec.series_id}__available_at"] = (
        result[f"{spec.series_id}__event_time"] + spec.availability_lag
    )
    return result.reset_index(drop=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_directory(source: Path, target: Path, *, attempts: int = 8) -> None:
    """Install a completed reference snapshot with bounded Windows-lock retries."""

    if source.parent != target.parent:
        raise ValueError("reference snapshot rename must remain on one volume")
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except OSError:
            if target.is_dir():
                left = source / "manifest.json"
                right = target / "manifest.json"
                if (
                    left.is_file()
                    and right.is_file()
                    and left.read_bytes() == right.read_bytes()
                ):
                    return
                raise RuntimeError(
                    f"immutable reference target differs at {target}"
                ) from None
            if attempt + 1 == attempts:
                raise
            time_module.sleep(0.05 * (2**attempt))


__all__ = [
    "DGS2",
    "DGS10",
    "FEDFUNDS",
    "FRED_CSV_URL",
    "KEN_FRENCH_DAILY_URL",
    "VIXCLS",
    "FREDCSVSource",
    "FredSeriesSpec",
    "KenFrenchDailyFactorsSource",
    "ReferenceBatch",
    "ReferenceDataCache",
    "parse_fred_csv",
    "parse_ken_french_daily_zip",
]
