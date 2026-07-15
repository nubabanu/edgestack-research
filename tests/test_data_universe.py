from __future__ import annotations

import asyncio
from datetime import date

import httpx

from edgestack.data.universe import (
    LIQUID_ETFS,
    WikipediaSP500UniverseSource,
    parse_wikipedia_sp500_html,
    reconstruct_membership_intervals,
)

HTML = """
<table id="constituents">
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th>
<th>Headquarters</th><th>Date added</th><th>CIK</th><th>Founded</th></tr>
<tr><td>AAA</td><td>Alpha</td><td>Tech</td><td>Software</td><td>NY</td>
<td>2020-01-02</td><td>0001</td><td>2000</td></tr>
<tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td><td>Insurance</td><td>NE</td>
<td>2010-02-01</td><td>0002</td><td>1900</td></tr>
</table>
<table id="changes">
<tr><th>Effective Date</th><th>Added</th><th>Added Security</th>
<th>Removed</th><th>Removed Security</th><th>Reason</th></tr>
<tr><td>January 2, 2020</td><td>AAA</td><td>Alpha</td><td>OLD</td><td>Old Co</td><td>Change</td></tr>
</table>
"""


def test_wikipedia_parser_and_reverse_membership_hook() -> None:
    current, changes = parse_wikipedia_sp500_html(HTML)
    assert {item.symbol for item in current} == {"AAA", "BRK.B"}
    intervals = reconstruct_membership_intervals(
        current,
        changes,
        start=date(2019, 1, 1),
        end=date(2021, 1, 1),
    )
    old = next(item for item in intervals if item.asset.symbol == "OLD")
    aaa = next(item for item in intervals if item.asset.symbol == "AAA")
    assert old.end == date(2020, 1, 2)
    assert aaa.start == date(2020, 1, 2)


def test_current_snapshot_adds_exactly_nine_etfs_and_bias_warning() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=HTML)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = WikipediaSP500UniverseSource(client=client)
    memberships = asyncio.run(source.memberships(date(2000, 1, 1), date(2024, 1, 1)))
    asyncio.run(client.aclose())
    etfs = {item.asset.symbol for item in memberships if item.asset.asset_type == "etf"}
    assert etfs == set(LIQUID_ETFS)
    assert source.last_snapshot is not None
    assert "SURVIVORSHIP_BIASED" in source.last_snapshot.warnings[0]
