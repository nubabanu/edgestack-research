from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from edgestack.data.calendars import NYSECalendar
from edgestack.data.factors import ReferenceBatch
from edgestack.mobile.models import MobileSnapshot
from edgestack.oil.context import OilContextStore
from edgestack.oil.data import (
    OilReferenceBatch,
    cftc_release_at,
    eia_release_at,
    latest_eia_release_at,
    parse_cftc_cot_json,
    parse_eia_history_xls,
    parse_eia_wpsr_csv,
    signed_price_observations,
)
from edgestack.oil.decision import (
    OilDecisionInputs,
    build_oil_snapshot,
    load_oil_config,
)
from edgestack.oil.ledger import OilLedger
from edgestack.oil.models import OilContext, OilHorizonDecision, OilSnapshot
from edgestack.oil.research import build_oil_streams
from edgestack.oil.risk import size_risk_lanes
from edgestack.oil.scheduler import build_oil_scheduler

NEW_YORK = ZoneInfo("America/New_York")
AS_OF = datetime(2026, 7, 20, 8, 30, tzinfo=NEW_YORK)
SERIES = (
    "DCOILWTICO",
    "DCOILBRENTEU",
    "OVXCLS",
    "DTWEXBGS",
    "WCESTUS1",
    "WCSSTUS1",
)


def _proxy_frame(symbol: str, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    base = np.linspace(45.0, 95.0, len(sessions))
    multiplier = {"USO": 1.0, "BNO": 0.9, "XLE": 1.2, "XOP": 1.1}[symbol]
    close = base * multiplier
    calendar = NYSECalendar()
    event = list(calendar.schedule(sessions.min(), sessions.max())["market_close"])
    return pd.DataFrame(
        {
            "symbol": symbol,
            "exchange": "US",
            "asset_type": "equity",
            "session": sessions,
            "event_time": event,
            "available_at": [item + timedelta(minutes=15) for item in event],
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
            "adjusted_close": close,
            "dividend": 0.0,
            "split_factor": 1.0,
            "source": "fixture",
        }
    )


def _fred_batch(sessions: pd.DatetimeIndex) -> ReferenceBatch:
    calendar = NYSECalendar()
    event = list(calendar.schedule(sessions.min(), sessions.max())["market_close"])
    frame = pd.DataFrame({"session": sessions})
    values = {
        "DCOILWTICO": np.linspace(40.0, 80.0, len(sessions)),
        "DCOILBRENTEU": np.linspace(42.0, 86.0, len(sessions)),
        "OVXCLS": np.full(len(sessions), 20.0),
        "DTWEXBGS": np.linspace(110.0, 90.0, len(sessions)),
        "WCESTUS1": np.linspace(500_000.0, 450_000.0, len(sessions)),
        "WCSSTUS1": np.linspace(60_000.0, 50_000.0, len(sessions)),
    }
    for series_id, series_values in values.items():
        frame[series_id] = series_values
        frame[f"{series_id}__event_time"] = event
        frame[f"{series_id}__available_at"] = [
            item + timedelta(hours=12) for item in event
        ]
    return ReferenceBatch(
        "fred_fixture",
        frame,
        AS_OF.astimezone(UTC),
        tuple(character * 64 for character in "abcdef"),
        metadata={"series": list(SERIES)},
    )


def _inputs() -> OilDecisionInputs:
    sessions = NYSECalendar().sessions(date(2024, 12, 1), date(2026, 7, 17))
    fred = _fred_batch(sessions)
    eia_publication = latest_eia_release_at(AS_OF)
    eia = tuple(
        OilReferenceBatch(
            "eia_wpsr",
            pd.DataFrame({"value": [1.0]}),
            AS_OF.astimezone(UTC),
            (str(index) * 64,),
            metadata={"published_at": eia_publication.isoformat()},
        )
        for index in range(1, 5)
    )
    cftc_frame = pd.DataFrame(
        {
            "report_date": pd.to_datetime(["2026-07-07", "2026-07-14"]),
            "available_at": [
                cftc_release_at(date(2026, 7, 7)),
                cftc_release_at(date(2026, 7, 14)),
            ],
            "managed_money_net": [100_000.0, 105_000.0],
        }
    )
    cftc = OilReferenceBatch(
        "cftc",
        cftc_frame,
        AS_OF.astimezone(UTC),
        ("9" * 64,),
    )
    return OilDecisionInputs(
        {symbol: _proxy_frame(symbol, sessions) for symbol in ("USO", "BNO", "XLE", "XOP")},
        {symbol: symbol.lower()[0] * 64 for symbol in ("USO", "BNO", "XLE", "XOP")},
        fred,
        eia,
        cftc,
    )


def _write_context(root: Path, *, at: datetime = AS_OF) -> None:
    OilContextStore(root / "oil" / "context.json").write(
        OilContext(
            recorded_at=at - timedelta(hours=1),
            expires_at=at + timedelta(hours=10),
            spread_bps=8.0,
            overnight_fee_usd_per_unit=0.01,
            event_risk="NORMAL",
        )
    )


def test_july_20_fixture_is_no_trade_with_complete_counterfactual_lanes(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_context(artifact_root)
    snapshot = build_oil_snapshot(
        _inputs(),
        paper_equity_usd=100_000.0,
        as_of=AS_OF,
        config=load_oil_config(),
        artifact_root=artifact_root,
        persist=False,
    )
    assert snapshot.status == "NO_TRADE"
    assert snapshot.proxy_agreement == "BULLISH"
    assert "FROZEN_JULY_20_BASELINE_NO_TRADE" in snapshot.intraday.active_vetoes
    assert [lane.risk_fraction for lane in snapshot.intraday.lanes] == [
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
    ]
    high_risk = snapshot.intraday.lanes[-1]
    assert "HIGH_RISK_NON_PROMOTABLE" in high_risk.label
    assert high_risk.maximum_planned_loss_usd <= 10_000.0
    assert high_risk.margin_usd <= 50_000.0
    assert high_risk.leverage * high_risk.stressed_move_fraction <= 0.50


def test_risk_sizing_uses_leverage_for_margin_not_account_loss() -> None:
    lanes = size_risk_lanes(
        equity_usd=100_000.0,
        price_usd=100.0,
        atr14_usd=2.0,
        p99_adverse_gap_fraction=0.005,
        spread_bps=2.0,
        overnight_fee_usd_per_unit=0.0,
        holding_nights=0,
    )
    governed, *_, challenge_10 = lanes
    assert governed.maximum_planned_loss_usd == pytest.approx(500.0)
    assert challenge_10.maximum_planned_loss_usd == pytest.approx(10_000.0)
    assert challenge_10.notional_usd == pytest.approx(250_000.0)
    assert challenge_10.leverage == 10.0
    assert challenge_10.margin_usd == pytest.approx(25_000.0)


def test_high_ovx_and_stale_proxy_fail_closed(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_context(artifact_root)
    inputs = _inputs()
    assert inputs.fred is not None
    stressed_frame = inputs.fred.frame.copy(deep=True)
    stressed_frame.loc[stressed_frame.index[-1], "OVXCLS"] = 100.0
    stressed = replace(
        inputs,
        fred=ReferenceBatch(
            inputs.fred.kind,
            stressed_frame,
            inputs.fred.fetched_at,
            inputs.fred.raw_sha256,
            inputs.fred.warnings,
            inputs.fred.metadata,
        ),
    )
    high_ovx = build_oil_snapshot(
        stressed,
        paper_equity_usd=100_000.0,
        as_of=AS_OF,
        config=load_oil_config(),
        artifact_root=artifact_root,
        persist=False,
    )
    assert "OVX_ABOVE_EXPANDING_90TH_PERCENTILE" in high_ovx.intraday.active_vetoes

    stale_bars = dict(inputs.bars)
    stale_bars["BNO"] = stale_bars["BNO"].iloc[:-1].copy()
    stale = build_oil_snapshot(
        replace(inputs, bars=stale_bars),
        paper_equity_usd=100_000.0,
        as_of=AS_OF,
        config=load_oil_config(),
        artifact_root=artifact_root,
        persist=False,
    )
    gate = next(item for item in stale.data_gates if item.name == "PROXY_BARS")
    assert stale.status == "NO_TRADE"
    assert gate.status == "FAIL"

    no_uso_bars = dict(inputs.bars)
    no_uso_bars.pop("USO")
    missing = build_oil_snapshot(
        replace(inputs, bars=no_uso_bars),
        paper_equity_usd=100_000.0,
        as_of=AS_OF,
        config=load_oil_config(),
        artifact_root=artifact_root,
        persist=False,
    )
    assert missing.status == "NO_TRADE"
    assert missing.intraday.reference_price_usd is None
    assert all(lane.status == "UNAVAILABLE" for lane in missing.intraday.lanes)


def test_three_session_swing_never_crosses_a_weekend(tmp_path: Path) -> None:
    wednesday = datetime(2026, 7, 15, 8, 30, tzinfo=NEW_YORK)
    artifact_root = tmp_path / "artifacts"
    _write_context(artifact_root, at=wednesday)
    snapshot = build_oil_snapshot(
        _inputs(),
        paper_equity_usd=100_000.0,
        as_of=wednesday,
        config=load_oil_config(),
        artifact_root=artifact_root,
        persist=False,
    )
    assert "SWING_EXPOSURE_CROSSES_WEEKEND" in snapshot.swing.active_vetoes


def test_challenge_lane_termination_is_irreversible_input() -> None:
    lanes = size_risk_lanes(
        equity_usd=100_000.0,
        equity_by_lane={"CHALLENGE_10": 80_000.0},
        peak_equity_by_lane={"CHALLENGE_10": 100_000.0},
        terminated_lanes={"CHALLENGE_10"},
        price_usd=100.0,
        atr14_usd=2.0,
        p99_adverse_gap_fraction=0.005,
        spread_bps=2.0,
        overnight_fee_usd_per_unit=0.0,
        holding_nights=0,
    )
    assert lanes[-1].status == "TERMINATED"
    assert lanes[-1].notional_usd == 0


def test_signed_spot_price_preserves_negative_wti() -> None:
    event = datetime(2020, 4, 20, 16, 0, tzinfo=NEW_YORK)
    frame = pd.DataFrame(
        {
            "session": [pd.Timestamp("2020-04-20")],
            "DCOILWTICO": [-36.98],
            "DCOILWTICO__event_time": [event],
            "DCOILWTICO__available_at": [event + timedelta(days=7)],
        }
    )
    batch = ReferenceBatch(
        "fred",
        frame,
        event + timedelta(days=8),
        ("a" * 64,),
        metadata={"series": ["DCOILWTICO"]},
    )
    observations = signed_price_observations(batch, series_ids=("DCOILWTICO",))
    assert observations[0].value == -36.98


def test_eia_and_cftc_publication_timestamps_and_parsers() -> None:
    assert eia_release_at(date(2026, 1, 21)) == datetime(
        2026, 1, 22, 12, 0, tzinfo=NEW_YORK
    )
    assert cftc_release_at(date(2026, 7, 14)) == datetime(
        2026, 7, 17, 15, 30, tzinfo=NEW_YORK
    ).astimezone(UTC)
    eia = parse_eia_wpsr_csv(
        b"Series,Current Week,Previous Week\nCrude,1,2\n",
        table_id="table1",
        published_at=datetime(2026, 7, 15, 10, 30, tzinfo=NEW_YORK),
    )
    assert eia.loc[0, "available_at"] == pd.Timestamp(
        datetime(2026, 7, 15, 10, 30, tzinfo=NEW_YORK)
    )
    body = json.dumps(
        [
            {
                "cftc_contract_market_code": "067651",
                "market_and_exchange_names": "CRUDE OIL, LIGHT SWEET - NYMEX",
                "report_date_as_yyyy_mm_dd": "2026-07-14T00:00:00.000",
                "m_money_positions_long_all": "200000",
                "m_money_positions_short_all": "50000",
                "open_interest_all": "1000000",
            }
        ]
    ).encode()
    cftc = parse_cftc_cot_json(body)
    assert cftc.loc[0, "managed_money_net"] == 150_000.0
    assert cftc.loc[0, "available_at"] == pd.Timestamp(
        datetime(2026, 7, 17, 15, 30, tzinfo=NEW_YORK).astimezone(UTC)
    )


def test_eia_history_workbook_uses_holiday_release_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook = pd.DataFrame(
        [
            ["Back to Contents", "description"],
            ["Sourcekey", "WCESTUS1"],
            ["Date", "value"],
            [datetime(2026, 1, 16), 400_000.0],
        ]
    )
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: workbook)
    parsed = parse_eia_history_xls(b"fixture", series_id="WCESTUS1")
    assert parsed.loc[0, "WCESTUS1"] == 400_000.0
    assert parsed.loc[0, "WCESTUS1__available_at"] == pd.Timestamp(
        datetime(2026, 1, 22, 12, 0, tzinfo=NEW_YORK).astimezone(UTC)
    )


def test_research_candidate_count_is_frozen_and_pre_forward() -> None:
    inputs = _inputs()
    config = load_oil_config()
    gross, net, definitions, _ = build_oil_streams(
        config, inputs, date(2026, 7, 20)
    )
    assert gross.shape[1] == net.shape[1] == len(definitions) == 72
    assert net.index.max() < pd.Timestamp("2026-07-20")
    assert all(item["outcome_proxy"] == "USO" for item in definitions.values())


def test_context_expiry_ledger_idempotency_and_mobile_schema(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_context(artifact_root)
    store = OilContextStore(artifact_root / "oil" / "context.json")
    assert store.read(at=AS_OF) is not None
    assert store.read(at=AS_OF - timedelta(hours=2)) is None
    assert store.read(at=datetime(2026, 7, 21, tzinfo=UTC)) is None
    snapshot = build_oil_snapshot(
        _inputs(),
        paper_equity_usd=100_000.0,
        as_of=AS_OF,
        config=load_oil_config(),
        artifact_root=artifact_root,
        persist=False,
    )
    ledger = OilLedger(artifact_root / "oil" / "forward.sqlite")
    assert ledger.record_snapshot(snapshot)
    assert not ledger.record_snapshot(snapshot)
    assert ledger.scorecard()["decisions"] == {"NO_TRADE": 1}
    oil_property = MobileSnapshot.model_json_schema()["properties"]["oil"]
    assert "anyOf" in oil_property


def test_proxy_lifecycle_records_pessimistic_stop_without_an_open_position(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_context(artifact_root)
    baseline = build_oil_snapshot(
        _inputs(),
        paper_equity_usd=100_000.0,
        as_of=AS_OF,
        config=load_oil_config(),
        artifact_root=artifact_root,
        persist=False,
    )
    intraday_payload = baseline.intraday.model_dump(mode="json")
    intraday_payload.update(status="PAPER_LONG", active_vetoes=[])
    intraday = OilHorizonDecision.model_validate(intraday_payload)
    snapshot_payload = baseline.model_dump(mode="json")
    snapshot_payload.update(
        decision_id="oil-stop-fixture",
        status="PAPER_LONG",
        intraday=intraday.model_dump(mode="json"),
    )
    snapshot = OilSnapshot.model_validate(snapshot_payload)
    ledger = OilLedger(artifact_root / "oil" / "forward.sqlite")
    assert ledger.record_snapshot(snapshot)

    uso = _inputs().bars["USO"].copy()
    previous = float(uso["adjusted_close"].iloc[-1])
    close_at = NYSECalendar().close_time(date(2026, 7, 20))
    new_bar = uso.iloc[-1:].copy()
    new_bar.loc[:, "session"] = pd.Timestamp("2026-07-20")
    new_bar.loc[:, "event_time"] = close_at
    new_bar.loc[:, "available_at"] = close_at + timedelta(minutes=15)
    new_bar.loc[:, "open"] = previous
    new_bar.loc[:, "high"] = previous * 1.01
    new_bar.loc[:, "low"] = previous * 0.80
    new_bar.loc[:, "close"] = previous * 0.90
    new_bar.loc[:, "adjusted_close"] = previous * 0.90
    uso = pd.concat([uso, new_bar], ignore_index=True)
    inserted = ledger.reconcile_proxy_bars(
        uso,
        at=datetime(2026, 7, 20, 17, 0, tzinfo=NEW_YORK),
        stop_slippage_bps=15.0,
    )
    assert inserted == 15
    assert ledger.scorecard()["events"] == {"EXIT": 5, "FILL": 5, "MARK": 5}
    assert not ledger.has_open_position()


def test_scheduler_declares_all_fixed_refreshes() -> None:
    scheduler = build_oil_scheduler(lambda _: None)
    assert {job.id for job in scheduler.get_jobs()} == {
        "oil_pre_open_eligibility",
        "oil_intraday_refresh",
        "oil_post_eia_refresh",
        "oil_swing_snapshot",
    }
