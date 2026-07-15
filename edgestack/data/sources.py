"""Daily-bar and quote provider adapters with immutable raw-response hooks.

The public contracts in this module are the shared immutable models from
``edgestack.models``.  Each ``fetch_bars`` call requests exactly one asset and
returns exactly one complete provider series.  :class:`FallbackDailyBarSource`
therefore cannot splice observations from multiple vendors.

Tiingo API documentation: https://www.tiingo.com/documentation/end-of-day
Stooq download endpoint: https://stooq.com/q/d/l/
Yahoo chart endpoint (unofficial): https://query1.finance.yahoo.com/
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from datetime import time as dt_time
from io import StringIO
from types import TracebackType
from typing import Any, Final, Protocol, Self
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from edgestack.models import (
    AssetKey,
    Bar,
    BarRequest,
    DailyBarSource,
    Quote,
    QuoteSource,
    SourceBatch,
    SourceCapabilities,
)

NEW_YORK: Final = ZoneInfo("America/New_York")
BAR_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "exchange",
    "asset_type",
    "session",
    "event_time",
    "available_at",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
    "dividend",
    "split_factor",
    "source",
)


class SourceError(RuntimeError):
    """Base class for a provider or canonicalization failure."""


class AuthenticationError(SourceError):
    """Raised when a provider rejects or lacks required credentials."""


class RateLimitError(SourceError):
    """Raised after provider rate limits remain exhausted after retries."""


class NoDataError(SourceError):
    """Raised when a provider returns no usable observations."""


class PartialSeriesError(SourceError):
    """Raised when a response does not match its single-asset request."""


@dataclass(frozen=True, slots=True)
class RawPayload:
    """Exact provider bytes and sanitized replay metadata.

    API keys and authorization headers must never be included in ``request_url``
    or ``response_headers``.
    """

    source: str
    asset: AssetKey | None
    fetched_at: datetime
    media_type: str
    body: bytes = field(repr=False)
    request_url: str = ""
    status_code: int = 200
    response_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fetched_at.tzinfo is None:
            raise ValueError("RawPayload.fetched_at must be timezone-aware")
        object.__setattr__(self, "response_headers", dict(self.response_headers))

    @property
    def sha256(self) -> str:
        """Return the SHA-256 content address of ``body``."""

        return hashlib.sha256(self.body).hexdigest()


class RawPayloadSink(Protocol):
    """Persistence hook invoked before a parsed provider result is returned."""

    def store(self, payload: RawPayload) -> str:
        """Persist ``payload`` immutably and return its SHA-256 digest."""


class MemoryRawPayloadSink:
    """In-memory sink useful for callers/tests that inspect exact responses."""

    def __init__(self) -> None:
        self.payloads: dict[str, RawPayload] = {}

    def store(self, payload: RawPayload) -> str:
        """Retain ``payload`` under its hash without overwriting a collision."""

        digest = payload.sha256
        existing = self.payloads.get(digest)
        if existing is not None and existing.body != payload.body:
            raise RuntimeError("SHA-256 collision while storing raw payload")
        self.payloads[digest] = payload
        return digest


def bars_to_frame(bars: Sequence[Bar] | SourceBatch) -> pd.DataFrame:
    """Convert immutable bars to the canonical long-form DataFrame schema."""

    observations = bars.bars if isinstance(bars, SourceBatch) else bars
    rows = []
    for bar in observations:
        event = pd.Timestamp(bar.event_time)
        event = (
            event.tz_localize(UTC) if event.tzinfo is None else event.tz_convert(UTC)
        )
        local_event = event.tz_convert(NEW_YORK)
        rows.append(
            {
                "symbol": bar.asset.symbol,
                "exchange": bar.asset.exchange,
                "asset_type": bar.asset.asset_type,
                "session": local_event.tz_localize(None).normalize(),
                "event_time": event,
                "available_at": pd.Timestamp(bar.available_at).tz_convert(UTC),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "adjusted_close": (
                    float(bar.adjusted_close)
                    if bar.adjusted_close is not None
                    else float(bar.close)
                ),
                "dividend": float(bar.dividend),
                "split_factor": float(bar.split_factor),
                "source": bar.source,
            }
        )
    if not rows:
        return pd.DataFrame(columns=BAR_COLUMNS)
    return canonicalize_bar_frame(pd.DataFrame(rows))


def canonicalize_bar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and deterministically order canonical daily bars."""

    missing = set(BAR_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"bar frame is missing canonical columns: {sorted(missing)}")
    result = frame.loc[:, BAR_COLUMNS].copy(deep=True)
    result["symbol"] = result["symbol"].astype("string").str.upper()
    result["session"] = pd.to_datetime(result["session"], errors="raise").dt.normalize()
    for column in ("event_time", "available_at"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    numeric = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjusted_close",
        "dividend",
        "split_factor",
    )
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    if result.duplicated(["symbol", "session"]).any():
        raise ValueError("duplicate symbol/session observations in canonical bars")
    if bool((result["available_at"] <= result["event_time"]).any()):
        raise ValueError("available_at must be strictly later than event_time")
    if bool(
        (result[["open", "high", "low", "close", "adjusted_close"]] <= 0).any(axis=None)
    ):
        raise ValueError("bar prices must be positive")
    if bool((result["volume"] < 0).any()):
        raise ValueError("bar volume cannot be negative")
    invalid_range = (result["low"] > result[["open", "close", "high"]].min(axis=1)) | (
        result["high"] < result[["open", "close", "low"]].max(axis=1)
    )
    if bool(invalid_range.any()):
        raise ValueError("OHLC range invariant violated")
    return result.sort_values(["symbol", "session"], kind="stable").reset_index(
        drop=True
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _datetime_from_unix(seconds: int) -> datetime:
    """Convert Unix seconds without the Windows pre-1970 CRT limitation."""

    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)


def _daily_times(session_value: Any) -> tuple[datetime, datetime]:
    session = pd.Timestamp(session_value)
    if session.tzinfo is not None:
        session = session.tz_convert(NEW_YORK).tz_localize(None)
    local_close = datetime.combine(session.date(), dt_time(16), tzinfo=NEW_YORK)
    event_time = local_close.astimezone(UTC)
    return event_time, event_time + timedelta(minutes=15)


def _number(value: Any, *, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _bar(
    request: BarRequest,
    *,
    session: Any,
    source: str,
    open_: Any,
    high: Any,
    low: Any,
    close: Any,
    volume: Any,
    adjusted_close: Any | None = None,
    dividend: Any = 0.0,
    split_factor: Any = 1.0,
) -> Bar:
    event_time, available_at = _daily_times(session)
    raw_close = _number(close)
    adjusted = _number(adjusted_close, default=raw_close)
    observation = Bar(
        asset=request.asset,
        event_time=event_time,
        available_at=available_at,
        open=_number(open_),
        high=_number(high),
        low=_number(low),
        close=raw_close,
        volume=_number(volume, default=0.0),
        adjusted_close=adjusted,
        dividend=_number(dividend, default=0.0),
        split_factor=_number(split_factor, default=1.0),
        source=source,
    )
    _validate_bar(observation)
    return observation


def _validate_bar(bar: Bar) -> None:
    values = (bar.open, bar.high, bar.low, bar.close, bar.adjusted_close)
    if any(value is None or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("bar prices must be positive and finite")
    if not math.isfinite(bar.volume) or bar.volume < 0:
        raise ValueError("bar volume must be non-negative and finite")
    if bar.low > min(bar.open, bar.close, bar.high) or bar.high < max(
        bar.open, bar.close, bar.low
    ):
        raise ValueError("OHLC range invariant violated")
    if bar.available_at <= bar.event_time:
        raise ValueError("bar availability must follow event time")


class _HttpSource:
    """Async HTTP lifecycle with request spacing and bounded exponential retry."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        raw_sink: RawPayloadSink | None = None,
        timeout: float = 30.0,
        max_attempts: int = 4,
        minimum_interval: float = 0.0,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._raw_sink = raw_sink or MemoryRawPayloadSink()
        self._timeout = timeout
        self._max_attempts = max(1, max_attempts)
        self._minimum_interval = max(0.0, minimum_interval)
        self._rate_lock = asyncio.Lock()
        self._last_request = 0.0
        self._now = now

    async def __aenter__(self) -> Self:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=True
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the internally owned HTTP client."""

        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _pace(self) -> None:
        async with self._rate_lock:
            remaining = self._minimum_interval - (time.monotonic() - self._last_request)
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_request = time.monotonic()

    async def _get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=True
            )
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            await self._pace()
            try:
                response = await self._client.get(url, params=params, headers=headers)
                if response.status_code in {401, 403}:
                    raise AuthenticationError(f"authentication failed for {url}")
                if response.status_code == 429:
                    raise RateLimitError(f"rate limited by {url}")
                if 400 <= response.status_code < 500:
                    raise NoDataError(
                        f"non-retryable provider response {response.status_code} for {url}"
                    )
                response.raise_for_status()
                return response
            except AuthenticationError:
                raise
            except (
                RateLimitError,
                httpx.HTTPStatusError,
                httpx.TimeoutException,
                httpx.TransportError,
            ) as error:
                last_error = error
                if attempt + 1 == self._max_attempts:
                    break
                base = min(8.0, 0.5 * 2**attempt)
                await asyncio.sleep(
                    base + random.Random(attempt).uniform(0.0, base / 4)
                )
        if isinstance(last_error, RateLimitError):
            raise last_error
        raise SourceError(
            f"request failed after {self._max_attempts} attempts: {url}"
        ) from last_error

    def _capture(
        self,
        response: httpx.Response,
        *,
        source: str,
        asset: AssetKey | None,
        fetched_at: datetime,
        redacted_url: str,
    ) -> str:
        allowed_headers = {
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower() in {"content-type", "etag", "last-modified", "date"}
        }
        return self._raw_sink.store(
            RawPayload(
                source=source,
                asset=asset,
                fetched_at=fetched_at,
                media_type=response.headers.get(
                    "content-type", "application/octet-stream"
                ).split(";")[0],
                body=response.content,
                request_url=redacted_url,
                status_code=response.status_code,
                response_headers=allowed_headers,
            )
        )


class TiingoDailyBarSource(_HttpSource):
    """Tiingo authenticated daily raw/adjusted OHLCV adapter."""

    name = "tiingo"
    capabilities = SourceCapabilities(
        name=name,
        daily=True,
        raw_and_adjusted=True,
        corporate_actions=True,
    )

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        if not api_key.strip():
            raise AuthenticationError("Tiingo requires a non-empty API key")
        super().__init__(
            minimum_interval=kwargs.pop("minimum_interval", 72.0), **kwargs
        )
        self._api_key = api_key

    async def fetch_bars(self, request: BarRequest) -> SourceBatch:
        symbol = _vendor_symbol(request.asset.symbol, self.name)
        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        response = await self._get(
            url,
            params={
                "startDate": request.start.isoformat(),
                "endDate": request.end.isoformat(),
                "resampleFreq": "daily",
            },
            headers={"Authorization": f"Token {self._api_key}"},
        )
        fetched_at = self._now()
        digest = self._capture(
            response,
            source=self.name,
            asset=request.asset,
            fetched_at=fetched_at,
            redacted_url=url,
        )
        try:
            records = response.json()
        except json.JSONDecodeError as error:
            raise SourceError(f"Tiingo returned invalid JSON for {symbol}") from error
        if not isinstance(records, list) or not records:
            raise NoDataError(f"Tiingo returned no bars for {symbol}")
        bars = tuple(
            _bar(
                request,
                session=item["date"],
                source=self.name,
                open_=item.get("open"),
                high=item.get("high"),
                low=item.get("low"),
                close=item.get("close"),
                volume=item.get("volume"),
                adjusted_close=item.get("adjClose"),
                dividend=item.get("divCash", 0.0),
                split_factor=item.get("splitFactor", 1.0),
            )
            for item in records
        )
        return SourceBatch(self.name, request, bars, fetched_at, digest)


class StooqDailyBarSource(_HttpSource):
    """No-key Stooq CSV daily history adapter.

    Stooq does not return separate raw/adjusted fields or corporate actions.  The
    provider close is retained as both close and adjusted close and the limitation
    is stamped into every batch.
    """

    name = "stooq"
    capabilities = SourceCapabilities(
        name=name,
        daily=True,
        raw_and_adjusted=False,
        corporate_actions=False,
    )
    _warning = (
        "Stooq CSV exposes one provider-adjusted series and no action table; "
        "raw and adjusted histories cannot be independently reconstructed."
    )

    async def fetch_bars(self, request: BarRequest) -> SourceBatch:
        params = {
            "s": _vendor_symbol(request.asset.symbol, self.name),
            "d1": request.start.strftime("%Y%m%d"),
            "d2": request.end.strftime("%Y%m%d"),
            "i": "d",
        }
        url = "https://stooq.com/q/d/l/"
        response = await self._get(url, params=params)
        fetched_at = self._now()
        digest = self._capture(
            response,
            source=self.name,
            asset=request.asset,
            fetched_at=fetched_at,
            redacted_url=f"{url}?{urlencode(params)}",
        )
        try:
            table = pd.read_csv(StringIO(response.text))
        except (pd.errors.ParserError, UnicodeDecodeError) as error:
            raise SourceError(
                f"Stooq returned invalid CSV for {request.asset.symbol}"
            ) from error
        table.columns = [str(column).strip().lower() for column in table.columns]
        required = {"date", "open", "high", "low", "close", "volume"}
        if table.empty or not required.issubset(table.columns):
            raise NoDataError(
                f"Stooq returned no usable bars for {request.asset.symbol}"
            )
        bars = tuple(
            _bar(
                request,
                session=row.date,
                source=self.name,
                open_=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
                adjusted_close=row.close,
            )
            for row in table.itertuples(index=False)
        )
        return SourceBatch(
            self.name, request, bars, fetched_at, digest, (self._warning,)
        )


class YahooDailyBarSource(_HttpSource):
    """Unofficial Yahoo chart adapter used as a whole-series fallback."""

    name = "yfinance"
    capabilities = SourceCapabilities(
        name=name,
        daily=True,
        raw_and_adjusted=True,
        corporate_actions=True,
        delayed_minutes=15,
    )
    _warning = (
        "SURVIVORSHIP_BIASED: Yahoo Finance is unofficial and resolves "
        "present-day tickers; it may rate-limit or revise history."
    )

    async def fetch_bars(self, request: BarRequest) -> SourceBatch:
        symbol = _vendor_symbol(request.asset.symbol, self.name)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        start_dt = datetime.combine(request.start, dt_time(), tzinfo=UTC)
        end_dt = datetime.combine(
            request.end + timedelta(days=1), dt_time(), tzinfo=UTC
        )
        response = await self._get(
            url,
            params={
                "period1": int(start_dt.timestamp()),
                "period2": int(end_dt.timestamp()),
                "interval": "1d",
                "events": "div,splits",
                "includeAdjustedClose": "true",
            },
            headers={"User-Agent": "Mozilla/5.0"},
        )
        fetched_at = self._now()
        digest = self._capture(
            response,
            source=self.name,
            asset=request.asset,
            fetched_at=fetched_at,
            redacted_url=url,
        )
        result = _yahoo_chart_result(response, request.asset.symbol)
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quote_items = indicators.get("quote") or []
        adjusted_items = indicators.get("adjclose") or []
        if not timestamps or not quote_items:
            raise NoDataError(f"Yahoo returned no bars for {request.asset.symbol}")
        quote_data = quote_items[0]
        adjusted = (
            adjusted_items[0].get("adjclose")
            if adjusted_items
            else quote_data.get("close")
        )
        events = result.get("events") or {}
        dividends = {
            int(key): value for key, value in (events.get("dividends") or {}).items()
        }
        splits = {
            int(key): value for key, value in (events.get("splits") or {}).items()
        }
        bars: list[Bar] = []
        malformed_ohlc_sessions: list[str] = []
        for index, timestamp_value in enumerate(timestamps):
            close = _at(quote_data.get("close"), index)
            if close is None:
                continue
            timestamp = int(timestamp_value)
            split = splits.get(timestamp, {})
            numerator = _number(split.get("numerator"), default=1.0)
            denominator = _number(split.get("denominator"), default=1.0)
            session = _datetime_from_unix(timestamp).date()
            try:
                bar = _bar(
                    request,
                    session=session,
                    source=self.name,
                    open_=_at(quote_data.get("open"), index),
                    high=_at(quote_data.get("high"), index),
                    low=_at(quote_data.get("low"), index),
                    close=close,
                    volume=_at(quote_data.get("volume"), index),
                    adjusted_close=_at(adjusted, index),
                    dividend=_number(
                        dividends.get(timestamp, {}).get("amount"), default=0.0
                    ),
                    split_factor=(numerator / denominator if denominator else 1.0),
                )
            except ValueError as error:
                if str(error) != "OHLC range invariant violated":
                    raise
                malformed_ohlc_sessions.append(session.isoformat())
                continue
            bars.append(bar)
        if not bars:
            raise NoDataError(
                f"Yahoo returned only null bars for {request.asset.symbol}"
            )
        warnings = [self._warning]
        if malformed_ohlc_sessions:
            examples = ",".join(malformed_ohlc_sessions[:10])
            warnings.append(
                "YAHOO_MALFORMED_OHLC_DROPPED: "
                f"{len(malformed_ohlc_sessions)} provider row(s) failed the OHLC "
                f"range invariant; sessions={examples}; raw payload retained."
            )
        return SourceBatch(
            self.name,
            request,
            tuple(bars),
            fetched_at,
            digest,
            tuple(warnings),
        )


# Compatibility alias for callers using the explicit library name.
YFinanceDailyBarSource = YahooDailyBarSource


class FallbackDailyBarSource:
    """Try an ordered source chain for one complete instrument history."""

    capabilities = SourceCapabilities(name="fallback", daily=True)

    def __init__(self, sources: Sequence[DailyBarSource]) -> None:
        if not sources:
            raise ValueError("FallbackDailyBarSource requires at least one source")
        self.sources = tuple(sources)

    async def fetch_bars(self, request: BarRequest) -> SourceBatch:
        """Return the first complete batch and record rejected provider attempts."""

        failures: list[str] = []
        for source in self.sources:
            try:
                batch = await source.fetch_bars(request)
                if batch.request != request or any(
                    bar.asset != request.asset for bar in batch.bars
                ):
                    raise PartialSeriesError(
                        f"{source.capabilities.name} returned observations for another request"
                    )
                if not batch.bars:
                    raise NoDataError(
                        f"{source.capabilities.name} returned an empty series"
                    )
                warning = (
                    "Fallback attempts before selection: " + " | ".join(failures)
                    if failures
                    else ""
                )
                warnings = batch.warnings + ((warning,) if warning else ())
                return SourceBatch(
                    batch.source,
                    batch.request,
                    batch.bars,
                    batch.fetched_at,
                    batch.raw_sha256,
                    warnings,
                )
            except (SourceError, ValueError, KeyError, json.JSONDecodeError) as error:
                failures.append(
                    f"{source.capabilities.name}:{type(error).__name__}:{error}"
                )
        raise NoDataError(
            f"all providers failed for {request.asset.symbol}: {'; '.join(failures)}"
        )


async def fetch_many(
    source: DailyBarSource,
    requests: Sequence[BarRequest],
    *,
    concurrency: int = 4,
) -> tuple[SourceBatch, ...]:
    """Fetch many independent histories with bounded deterministic concurrency."""

    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)

    async def fetch(index: int, request: BarRequest) -> tuple[int, SourceBatch]:
        async with semaphore:
            return index, await source.fetch_bars(request)

    indexed = await asyncio.gather(
        *(fetch(index, request) for index, request in enumerate(requests))
    )
    return tuple(batch for _, batch in sorted(indexed, key=lambda item: item[0]))


class YahooQuoteSource(_HttpSource):
    """Latest unofficial Yahoo one-minute observation, always marked delayed."""

    name = "yfinance"
    capabilities = SourceCapabilities(
        name=name,
        daily=True,
        intraday=True,
        delayed_minutes=15,
    )

    async def fetch_quotes(self, assets: Sequence[AssetKey]) -> tuple[Quote, ...]:
        quotes: list[Quote] = []
        for asset in tuple(dict.fromkeys(assets)):
            symbol = _vendor_symbol(asset.symbol, self.name)
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            response = await self._get(
                url,
                params={"range": "1d", "interval": "1m", "includePrePost": "true"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            received_at = self._now()
            self._capture(
                response,
                source=self.name,
                asset=asset,
                fetched_at=received_at,
                redacted_url=url,
            )
            result = _yahoo_chart_result(response, asset.symbol)
            timestamps = result.get("timestamp") or []
            quote_data = ((result.get("indicators") or {}).get("quote") or [{}])[0]
            closes = quote_data.get("close") or []
            valid = [index for index, value in enumerate(closes) if value is not None]
            if not valid:
                raise NoDataError(f"Yahoo returned no quote for {asset.symbol}")
            index = valid[-1]
            quotes.append(
                Quote(
                    asset=asset,
                    price=float(closes[index]),
                    provider_time=_datetime_from_unix(int(timestamps[index])),
                    received_at=received_at,
                    source=self.name,
                    delayed_minutes=15,
                )
            )
        return tuple(quotes)


class TiingoQuoteSource(_HttpSource):
    """Tiingo IEX last-sale adapter with account-dependent entitlement."""

    name = "tiingo_iex"
    capabilities = SourceCapabilities(name=name, intraday=True)

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        if not api_key.strip():
            raise AuthenticationError("Tiingo requires a non-empty API key")
        super().__init__(**kwargs)
        self._api_key = api_key

    async def fetch_quotes(self, assets: Sequence[AssetKey]) -> tuple[Quote, ...]:
        requested = tuple(dict.fromkeys(assets))
        if not requested:
            return ()
        url = "https://api.tiingo.com/iex/"
        response = await self._get(
            url,
            params={
                "tickers": ",".join(
                    _vendor_symbol(asset.symbol, "tiingo") for asset in requested
                )
            },
            headers={"Authorization": f"Token {self._api_key}"},
        )
        received_at = self._now()
        self._capture(
            response,
            source=self.name,
            asset=None,
            fetched_at=received_at,
            redacted_url=url,
        )
        try:
            records = response.json()
        except json.JSONDecodeError as error:
            raise SourceError("Tiingo returned invalid quote JSON") from error
        by_symbol = {
            str(record.get("ticker", "")).upper(): record for record in records
        }
        quotes: list[Quote] = []
        for asset in requested:
            vendor_symbol = _vendor_symbol(asset.symbol, "tiingo").upper()
            record = by_symbol.get(vendor_symbol)
            if record is None:
                raise PartialSeriesError(f"Tiingo omitted quote for {asset.symbol}")
            price = record.get("last") or record.get("tngoLast") or record.get("mid")
            timestamp = record.get("timestamp") or record.get("quoteTimestamp")
            if price is None or timestamp is None:
                raise NoDataError(
                    f"Tiingo quote for {asset.symbol} has no price/timestamp"
                )
            provider_time = pd.Timestamp(timestamp)
            provider_time = (
                provider_time.tz_localize(UTC)
                if provider_time.tzinfo is None
                else provider_time.tz_convert(UTC)
            )
            quotes.append(
                Quote(
                    asset=asset,
                    price=float(price),
                    provider_time=provider_time.to_pydatetime(),
                    received_at=received_at,
                    source=self.name,
                    delayed_minutes=None,
                )
            )
        return tuple(quotes)


def _vendor_symbol(symbol: str, source: str) -> str:
    canonical = symbol.strip().upper()
    if source == "stooq":
        return canonical.replace(".", "-").lower() + (
            "" if canonical.startswith("^") else ".us"
        )
    if source in {"yfinance", "tiingo", "tiingo_iex"}:
        return canonical.replace(".", "-")
    return canonical


def _yahoo_chart_result(response: httpx.Response, symbol: str) -> Mapping[str, Any]:
    try:
        body = response.json()
    except json.JSONDecodeError as error:
        raise SourceError(f"Yahoo returned invalid JSON for {symbol}") from error
    chart = body.get("chart") or {}
    if chart.get("error"):
        raise NoDataError(f"Yahoo error for {symbol}: {chart['error']}")
    results = chart.get("result") or []
    if not results or not isinstance(results[0], Mapping):
        raise NoDataError(f"Yahoo returned no result for {symbol}")
    return results[0]


def _at(values: Sequence[Any] | None, index: int) -> Any:
    return None if values is None or index >= len(values) else values[index]


__all__ = [
    "BAR_COLUMNS",
    "AuthenticationError",
    "Bar",
    "BarRequest",
    "DailyBarSource",
    "FallbackDailyBarSource",
    "MemoryRawPayloadSink",
    "NoDataError",
    "PartialSeriesError",
    "Quote",
    "QuoteSource",
    "RateLimitError",
    "RawPayload",
    "RawPayloadSink",
    "SourceBatch",
    "SourceCapabilities",
    "SourceError",
    "StooqDailyBarSource",
    "TiingoDailyBarSource",
    "TiingoQuoteSource",
    "YFinanceDailyBarSource",
    "YahooDailyBarSource",
    "YahooQuoteSource",
    "bars_to_frame",
    "canonicalize_bar_frame",
    "fetch_many",
]
