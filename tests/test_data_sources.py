from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime

import httpx
import pytest

from edgestack.data.sources import (
    FallbackDailyBarSource,
    MemoryRawPayloadSink,
    NoDataError,
    StooqDailyBarSource,
    YahooDailyBarSource,
)
from edgestack.models import AssetKey, BarRequest, SourceCapabilities


def test_stooq_adapter_preserves_raw_and_canonicalizes() -> None:
    body = b"Date,Open,High,Low,Close,Volume\n2024-01-02,10,12,9,11,1000\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["s"] == "brk-b.us"
        return httpx.Response(200, content=body, headers={"content-type": "text/csv"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sink = MemoryRawPayloadSink()
    source = StooqDailyBarSource(client=client, raw_sink=sink, minimum_interval=0)
    request = BarRequest(AssetKey("BRK.B"), date(2024, 1, 1), date(2024, 1, 3))
    batch = asyncio.run(source.fetch_bars(request))
    asyncio.run(client.aclose())

    assert batch.request == request
    assert batch.raw_sha256 in sink.payloads
    assert batch.bars[0].close == 11
    assert batch.bars[0].adjusted_close == 11
    assert batch.bars[0].available_at > batch.bars[0].event_time
    assert "raw and adjusted" in batch.warnings[0]


def test_yahoo_uses_adjusted_close_and_actions() -> None:
    timestamp = int(datetime(2024, 1, 2, tzinfo=UTC).timestamp())
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "timestamp": [timestamp],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0],
                                "high": [102.0],
                                "low": [99.0],
                                "close": [100.0],
                                "volume": [1000],
                            }
                        ],
                        "adjclose": [{"adjclose": [50.0]}],
                    },
                    "events": {
                        "dividends": {str(timestamp): {"amount": 1.0}},
                        "splits": {
                            str(timestamp): {"numerator": 2.0, "denominator": 1.0}
                        },
                    },
                }
            ],
        }
    }

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(payload).encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = YahooDailyBarSource(client=client, minimum_interval=0)
    request = BarRequest(AssetKey("TEST"), date(2024, 1, 1), date(2024, 1, 3))
    batch = asyncio.run(source.fetch_bars(request))
    asyncio.run(client.aclose())

    bar = batch.bars[0]
    assert bar.close == 100
    assert bar.adjusted_close == 50
    assert bar.dividend == 1
    assert bar.split_factor == 2
    assert "SURVIVORSHIP_BIASED" in batch.warnings[0]


def test_yahoo_parses_pre_1970_unix_dates_on_windows() -> None:
    timestamp = int(datetime(1962, 1, 2, tzinfo=UTC).timestamp())
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "timestamp": [timestamp],
                    "indicators": {
                        "quote": [
                            {
                                "open": [10.0],
                                "high": [11.0],
                                "low": [9.0],
                                "close": [10.5],
                                "volume": [1000],
                            }
                        ],
                        "adjclose": [{"adjclose": [1.25]}],
                    },
                }
            ],
        }
    }

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(payload).encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = YahooDailyBarSource(client=client, minimum_interval=0)
    request = BarRequest(AssetKey("OLD"), date(1960, 1, 1), date(1965, 1, 1))
    batch = asyncio.run(source.fetch_bars(request))
    asyncio.run(client.aclose())

    assert batch.bars[0].event_time.date() == date(1962, 1, 2)
    assert batch.bars[0].adjusted_close == 1.25


def test_fallback_never_splices_failed_provider() -> None:
    request = BarRequest(AssetKey("TEST"), date(2024, 1, 1), date(2024, 1, 3))

    class Failed:
        capabilities = SourceCapabilities("failed")

        async def fetch_bars(self, _: BarRequest):  # type: ignore[no-untyped-def]
            raise NoDataError("partial history rejected")

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="Date,Open,High,Low,Close,Volume\n2024-01-02,10,12,9,11,1000\n",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stooq = StooqDailyBarSource(client=client, minimum_interval=0)
    batch = asyncio.run(FallbackDailyBarSource((Failed(), stooq)).fetch_bars(request))
    asyncio.run(client.aclose())

    assert {bar.source for bar in batch.bars} == {"stooq"}
    assert "failed:NoDataError" in batch.warnings[-1]


def test_empty_request_source_result_is_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="No data")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = StooqDailyBarSource(client=client, minimum_interval=0)
    request = BarRequest(AssetKey("NOPE"), date(2024, 1, 1), date(2024, 1, 3))
    with pytest.raises(NoDataError):
        asyncio.run(source.fetch_bars(request))
    asyncio.run(client.aclose())
