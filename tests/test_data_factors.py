from __future__ import annotations

import asyncio
import zipfile
from datetime import date
from io import BytesIO

import httpx
import pytest

from edgestack.data.factors import (
    FREDCSVSource,
    FredSeriesSpec,
    KenFrenchDailyFactorsSource,
    ReferenceDataCache,
    parse_ken_french_daily_zip,
)


def _factor_zip() -> bytes:
    content = (
        b"This file was created for testing\n"
        b",Mkt-RF,SMB,HML,RF\n"
        b"19600104,  0.50,  0.10, -0.20, 0.01\n"
        b"19600105, -0.25,  0.05,  0.20, 0.01\n"
        b"Copyright 2026\n"
    )
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("F-F_Research_Data_Factors_daily.csv", content)
    return buffer.getvalue()


def test_ken_french_parser_adds_rf_to_total_market() -> None:
    frame = parse_ken_french_daily_zip(_factor_zip())
    assert frame.loc[0, "mkt_rf"] == pytest.approx(0.005)
    assert frame.loc[0, "market_return"] == pytest.approx(0.0051)
    assert frame.loc[0, "available_at"] > frame.loc[0, "event_time"]


def test_reference_sources_and_parquet_cache(tmp_path) -> None:
    factor_body = _factor_zip()

    async def factor_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=factor_body)

    factor_client = httpx.AsyncClient(transport=httpx.MockTransport(factor_handler))
    factors = KenFrenchDailyFactorsSource(client=factor_client)
    batch = asyncio.run(factors.fetch(date(1960, 1, 1), date(1960, 1, 6)))
    asyncio.run(factor_client.aclose())

    cache = ReferenceDataCache(tmp_path)
    snapshot_id = cache.store(batch)
    restored = cache.load(batch.kind, snapshot_id)
    assert (
        restored.frame["market_return"].tolist()
        == batch.frame["market_return"].tolist()
    )

    async def fred_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["id"] == "TEST"
        return httpx.Response(
            200,
            text="observation_date,TEST\n2024-01-02,12.5\n2024-01-03,.\n",
        )

    fred_client = httpx.AsyncClient(transport=httpx.MockTransport(fred_handler))
    fred = FREDCSVSource(client=fred_client)
    macro = asyncio.run(
        fred.fetch_series((FredSeriesSpec("TEST"),), date(2024, 1, 1), date(2024, 1, 4))
    )
    asyncio.run(fred_client.aclose())
    assert macro.frame.loc[0, "TEST"] == 12.5
    assert "LATEST_VINTAGE_NOT_POINT_IN_TIME" in macro.warnings[0]
