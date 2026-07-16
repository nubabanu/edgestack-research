from __future__ import annotations

import asyncio
from datetime import date

import httpx

from edgestack.data.pit_audit import (
    stooq_member_key,
    summarize_pit_coverage,
    universe_bias_delta,
)
from edgestack.data.universe import (
    LIQUID_ETFS,
    MembershipChange,
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


def test_stooq_member_key_matches_archive_naming() -> None:
    assert stooq_member_key("AAPL") == "aapl.us.txt"
    assert stooq_member_key("BRK.B") == "brk-b.us.txt"


def test_pit_coverage_audit_buckets_removals_by_year() -> None:
    changes = (
        MembershipChange(date(2019, 6, 3), "NEW", "New Co", "GONE", "Gone Co", ""),
        MembershipChange(date(2019, 9, 9), None, None, "LOST", "Lost Co", ""),
        MembershipChange(date(2021, 3, 1), None, None, "KEPT", "Kept Co", ""),
        # An addition-only row and an out-of-window removal contribute nothing.
        MembershipChange(date(2020, 1, 2), "ADD", "Add Co", None, None, ""),
        MembershipChange(date(1990, 1, 2), None, None, "OLD", "Old Co", ""),
    )
    report = summarize_pit_coverage(
        changes,
        {"gone.us.txt", "kept.us.txt"},
        start=date(2015, 1, 1),
        end=date(2022, 1, 1),
    )
    assert report["removed_symbols"] == 3
    assert report["covered_symbols"] == 2
    assert report["per_year"]["2019"] == {
        "removed": 2,
        "covered": 1,
        "coverage_fraction": 0.5,
    }
    assert report["per_year"]["2021"]["coverage_fraction"] == 1.0
    assert report["uncovered"] == [
        {"symbol": "LOST", "removed_on": "2019-09-09"}
    ]


def test_universe_bias_delta_reports_both_tiers_and_the_delta() -> None:
    import numpy as np
    import pandas as pd

    sessions = pd.bdate_range("2020-01-02", periods=120)
    rng = np.random.default_rng(5)
    symbols = [f"S{index:02d}" for index in range(12)]
    rows = []
    for column, symbol in enumerate(symbols):
        closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, len(sessions)))
        rows.append(
            pd.DataFrame(
                {"symbol": symbol, "session": sessions, "adjusted_close": closes}
            )
        )
    bars = pd.concat(rows, ignore_index=True)
    universe = pd.DataFrame(
        {
            "symbol": symbols,
            "asset_type": "equity",
            "start": sessions[0],
            # The last two names are delisted midway; the biased convention
            # keeps them in the panel for the whole window.
            "end": [pd.NaT] * 10 + [sessions[60], sessions[60]],
        }
    )
    report = universe_bias_delta(bars, universe)
    assert report["policy"] == "REPORT_ONLY_BIAS_QUANTIFICATION"
    assert report["symbols"] == 12
    assert report["pit_masked_cells_fraction"] > 0.0
    for name in ("reversal_5d", "momentum_12_1"):
        entry = report["signals"][name]
        assert set(entry) == {"survivorship_biased", "pit_masked", "bias_delta"}
        assert entry["survivorship_biased"]["observations"] > 0


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
