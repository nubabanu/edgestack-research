"""Fail-closed USO paper decisions with causal oil references and risk lanes."""

from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from edgestack.data.cache import ContentAddressedRawStore
from edgestack.data.calendars import NYSECalendar
from edgestack.data.factors import FREDCSVSource, ReferenceBatch
from edgestack.data.sources import (
    FallbackDailyBarSource,
    StooqDailyBarSource,
    YahooDailyBarSource,
    bars_to_frame,
)
from edgestack.models import AssetKey, BarRequest
from edgestack.oil.context import OilContextStore
from edgestack.oil.data import (
    OIL_FRED_SPECS,
    CftcCotSource,
    EiaHistorySource,
    EiaWpsrSource,
    OilReferenceBatch,
    eia_release_at,
    latest_eia_release_at,
)
from edgestack.oil.ledger import OilLedger
from edgestack.oil.models import (
    OilDataGate,
    OilHorizonDecision,
    OilSnapshot,
)
from edgestack.oil.risk import size_risk_lanes, unavailable_risk_lanes
from edgestack.provenance import canonical_sha256

NEW_YORK = ZoneInfo("America/New_York")
PROXY_SYMBOLS = ("USO", "BNO", "XLE", "XOP")
REQUIRED_GATES = {
    "PROXY_BARS",
    "EIA_WPSR",
    "CFTC_POSITIONING",
    "FRED_REFERENCES",
    "OPERATOR_CONTEXT",
}


@dataclass(frozen=True, slots=True)
class OilDecisionInputs:
    """Prepared causal inputs, injectable for deterministic replay tests."""

    bars: dict[str, pd.DataFrame] = field(repr=False)
    bar_hashes: dict[str, str]
    fred: ReferenceBatch | None = field(default=None, repr=False)
    eia: tuple[OilReferenceBatch, ...] = field(default=(), repr=False)
    cftc: OilReferenceBatch | None = field(default=None, repr=False)
    eia_history: OilReferenceBatch | None = field(default=None, repr=False)
    warnings: tuple[str, ...] = ()


def load_oil_config(path: str | Path = "configs/oil-paper-v1.yaml") -> dict[str, Any]:
    """Load and enforce the non-negotiable paper/governance configuration."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("oil configuration must be a mapping")
    config = cast(dict[str, Any], payload)
    if config.get("paper_only") is not True or config.get("outcome_proxy") != "USO":
        raise ValueError("oil v1 must remain paper-only with USO as outcome proxy")
    risk = cast(dict[str, Any], config.get("risk", {}))
    if float(risk.get("governed_fraction", -1)) != 0.005:
        raise ValueError("oil governed risk must remain 0.5%")
    if list(risk.get("challenge_fractions", [])) != [0.01, 0.02, 0.05, 0.1]:
        raise ValueError("oil challenge lanes must remain fixed at 1/2/5/10%")
    return config


async def fetch_oil_inputs(
    *,
    as_of: datetime,
    artifact_root: str | Path = "artifacts",
    config: dict[str, Any] | None = None,
) -> OilDecisionInputs:
    """Fetch free proxy/reference inputs while retaining every response byte."""

    if as_of.tzinfo is None:
        raise ValueError("oil input time must be timezone-aware")
    resolved = config or load_oil_config()
    data = cast(dict[str, Any], resolved["data"])
    start = date.fromisoformat(str(data["start"]))
    end = as_of.astimezone(NEW_YORK).date()
    raw_store = ContentAddressedRawStore(Path(artifact_root) / "oil" / "raw")
    bar_source = FallbackDailyBarSource(
        (
            StooqDailyBarSource(raw_sink=raw_store),
            YahooDailyBarSource(raw_sink=raw_store),
        )
    )
    requests = tuple(
        BarRequest(AssetKey(symbol), start, end, adjusted=True)
        for symbol in PROXY_SYMBOLS
    )
    warnings: list[str] = []
    bars: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    batches = await asyncio.gather(
        *(bar_source.fetch_bars(request) for request in requests),
        return_exceptions=True,
    )
    for request, batch in zip(requests, batches, strict=True):
        if isinstance(batch, BaseException):
            warnings.append(
                f"{request.asset.symbol}_PROXY_FETCH_FAILED:"
                f"{type(batch).__name__}:{batch}"
            )
        else:
            symbol = batch.request.asset.symbol
            bars[symbol] = bars_to_frame(batch)
            hashes[symbol] = batch.raw_sha256
            warnings.extend(batch.warnings)

    fred_source = FREDCSVSource(raw_sink=raw_store)
    eia_source = EiaWpsrSource(raw_sink=raw_store)
    eia_history_source = EiaHistorySource(raw_sink=raw_store)
    cftc_source = CftcCotSource(raw_sink=raw_store)
    eia_publication = latest_eia_release_at(as_of)

    async def current_wpsr(table_id: str) -> OilReferenceBatch:
        # ir.eia.gov exposes the current report, not archived publication
        # vintages. Never relabel today's bytes as an old historical release.
        now = datetime.now(UTC)
        replay = as_of.astimezone(UTC)
        if replay < now - timedelta(days=2) or replay > now + timedelta(days=7):
            raise RuntimeError("CURRENT_WPSR_CANNOT_SERVE_HISTORICAL_REPLAY")
        return await eia_source.fetch_table(table_id, published_at=eia_publication)

    tasks = (
        fred_source.fetch_series(OIL_FRED_SPECS, start, end),
        current_wpsr("table1"),
        current_wpsr("table4"),
        current_wpsr("table9"),
        current_wpsr("table11"),
        cftc_source.fetch_wti(start=start, end=end),
        eia_history_source.fetch_series(),
    )
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for name, result in zip(
        (
            "FRED",
            "EIA_TABLE1",
            "EIA_TABLE4",
            "EIA_TABLE9",
            "EIA_TABLE11",
            "CFTC",
            "EIA_HISTORY",
        ),
        results,
        strict=True,
    ):
        if isinstance(result, BaseException):
            warnings.append(f"{name}_FETCH_FAILED:{type(result).__name__}:{result}")
    fred = results[0] if isinstance(results[0], ReferenceBatch) else None
    eia = tuple(item for item in results[1:5] if isinstance(item, OilReferenceBatch))
    cftc = results[5] if isinstance(results[5], OilReferenceBatch) else None
    eia_history = results[6] if isinstance(results[6], OilReferenceBatch) else None
    if fred is not None:
        warnings.extend(fred.warnings)
    if eia_history is not None:
        warnings.extend(eia_history.warnings)
    return OilDecisionInputs(
        bars,
        hashes,
        fred,
        eia,
        cftc,
        eia_history,
        tuple(warnings),
    )


def _causal_frame(frame: pd.DataFrame, moment: datetime) -> pd.DataFrame:
    result = frame.copy(deep=True)
    if "available_at" in result:
        available = pd.to_datetime(result["available_at"], utc=True, errors="coerce")
        result = result.loc[available <= pd.Timestamp(moment)]
    return result.sort_values("session", kind="stable").reset_index(drop=True)


def _expected_completed_session(moment: datetime, calendar: NYSECalendar) -> date:
    local = moment.astimezone(NEW_YORK)
    if calendar.is_session(local.date()) and moment >= calendar.close_time(local.date()):
        return local.date()
    return calendar.previous_session(local.date()).date()


def _atr_and_gap(frame: pd.DataFrame) -> tuple[float, float, float]:
    if len(frame) < 30:
        raise ValueError("USO requires at least 30 causal daily bars")
    close = frame["close"].astype(float)
    adjusted = frame["adjusted_close"].astype(float)
    factor = adjusted / close
    adjusted_open = frame["open"].astype(float) * factor
    adjusted_high = frame["high"].astype(float) * factor
    adjusted_low = frame["low"].astype(float) * factor
    previous = adjusted.shift(1)
    true_range = pd.concat(
        [
            adjusted_high - adjusted_low,
            (adjusted_high - previous).abs(),
            (adjusted_low - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = float(true_range.ewm(alpha=1.0 / 14.0, adjust=False).mean().iloc[-1])
    adverse_gap = (1.0 - adjusted_open / previous).clip(lower=0).dropna()
    p99 = float(adverse_gap.quantile(0.99)) if not adverse_gap.empty else 0.0
    latest_up_gap_atr = max(0.0, float(adjusted_open.iloc[-1] - previous.iloc[-1])) / atr14
    if not all(math.isfinite(item) for item in (atr14, p99, latest_up_gap_atr)):
        raise ValueError("USO ATR/gap statistics are not finite")
    return atr14, p99, latest_up_gap_atr


def _proxy_state(
    bars: dict[str, pd.DataFrame], moment: datetime
) -> tuple[str, dict[str, pd.DataFrame], dict[str, float]]:
    causal: dict[str, pd.DataFrame] = {}
    momentum: dict[str, float] = {}
    for symbol in PROXY_SYMBOLS:
        frame = bars.get(symbol)
        if frame is None:
            continue
        usable = _causal_frame(frame, moment)
        if len(usable) < 21:
            continue
        causal[symbol] = usable
        close = usable["adjusted_close"].astype(float)
        momentum[symbol] = float(close.iloc[-1] / close.iloc[-6] - 1.0)
    if len(momentum) != len(PROXY_SYMBOLS):
        return "DATA_UNAVAILABLE", causal, momentum
    positives = sum(value > 0 for value in momentum.values())
    agreement = "BULLISH" if positives >= 3 else "BEARISH" if positives <= 1 else "MIXED"
    return agreement, causal, momentum


def _fred_state(
    batch: ReferenceBatch | None, moment: datetime
) -> tuple[bool, float | None, float | None, float | None, str]:
    if batch is None:
        return False, None, None, None, "FRED reference fetch unavailable"
    frame = batch.frame.copy(deep=True)
    required = ("OVXCLS", "DTWEXBGS")
    for series in required:
        if series not in frame or f"{series}__available_at" not in frame:
            return False, None, None, None, f"FRED reference missing {series}"
    ovx_mask = pd.to_datetime(frame["OVXCLS__available_at"], utc=True) <= pd.Timestamp(moment)
    ovx = frame.loc[ovx_mask, ["session", "OVXCLS"]].dropna()
    dollar_mask = pd.to_datetime(frame["DTWEXBGS__available_at"], utc=True) <= pd.Timestamp(moment)
    dollar = frame.loc[dollar_mask, ["session", "DTWEXBGS"]].dropna()
    if len(ovx) < 253 or len(dollar) < 21:
        return False, None, None, None, "FRED history is too short for causal gates"
    ovx_value = float(ovx["OVXCLS"].iloc[-1])
    ovx_p90 = float(ovx["OVXCLS"].iloc[:-1].expanding(min_periods=252).quantile(0.90).iloc[-1])
    dollar_change = float(dollar["DTWEXBGS"].iloc[-1] / dollar["DTWEXBGS"].iloc[-21] - 1.0)
    if pd.Timestamp(ovx["session"].iloc[-1]).date() < moment.astimezone(NEW_YORK).date() - timedelta(days=7):
        return False, ovx_value, ovx_p90, dollar_change, "OVX observation is stale"
    return True, ovx_value, ovx_p90, dollar_change, "OVX and dollar are causally available"


def _crosses_weekend(entry: date, exit_: date) -> bool:
    cursor = entry
    while cursor < exit_:
        cursor += timedelta(days=1)
        if cursor.weekday() == 5:
            return True
    return False


def _eia_window(moment: datetime) -> bool:
    local = moment.astimezone(NEW_YORK)
    nominal = local.date() - timedelta(days=(local.weekday() - 2) % 7)
    release = eia_release_at(nominal)
    return release - timedelta(minutes=15) <= local <= release + timedelta(minutes=45)


def _gate(
    name: str,
    status: str,
    reason: str,
    *,
    as_of: datetime | None = None,
    hashes: tuple[str, ...] = (),
) -> OilDataGate:
    return OilDataGate(
        name=name,
        status=status,
        reason=reason,
        as_of=as_of,
        raw_sha256=hashes,
    )


def _freeze_status(
    artifact_root: Path, campaign_id: str, horizon: str
) -> tuple[bool, str]:
    path = artifact_root / "campaigns" / campaign_id / "oil" / "freeze.json"
    if not path.is_file():
        return False, "historical diagnostics are not promoted; a forward freeze is absent"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rules = payload.get("rules", {}) if isinstance(payload, dict) else {}
    rule = rules.get(horizon) if isinstance(rules, dict) else None
    if not isinstance(rule, dict):
        return False, f"no frozen {horizon.lower()} rule"
    return True, f"forward-only frozen rule {rule.get('candidate_id', 'unknown')}"


def _write_latest(path: Path, snapshot: OilSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = snapshot.model_dump_json(indent=2).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def build_oil_snapshot(
    inputs: OilDecisionInputs,
    *,
    paper_equity_usd: float,
    as_of: datetime,
    config: dict[str, Any] | None = None,
    artifact_root: str | Path = "artifacts",
    persist: bool = True,
) -> OilSnapshot:
    """Build and optionally append one paper-only snapshot; never place an order."""

    if as_of.tzinfo is None:
        raise ValueError("oil decision time must be timezone-aware")
    if paper_equity_usd <= 0:
        raise ValueError("paper equity must be positive")
    resolved = config or load_oil_config()
    campaign_id = str(resolved["campaign_id"])
    root = Path(artifact_root).resolve()
    decision_cfg = cast(dict[str, Any], resolved["decision"])
    calendar = NYSECalendar()
    agreement, causal_bars, _momentum = _proxy_state(inputs.bars, as_of)
    expected = _expected_completed_session(as_of, calendar)
    proxy_ok = len(causal_bars) == len(PROXY_SYMBOLS)
    latest_session: date | None = None
    if proxy_ok:
        latest_sessions = {
            pd.Timestamp(frame["session"].iloc[-1]).date() for frame in causal_bars.values()
        }
        proxy_ok = len(latest_sessions) == 1
        if proxy_ok:
            latest_session = next(iter(latest_sessions))
            proxy_ok = latest_session >= expected
    proxy_reason = (
        f"all four proxies causally complete through {latest_session}"
        if proxy_ok
        else f"proxy series are missing, stale, or disagree on session; expected {expected}"
    )
    gates: list[OilDataGate] = [
        _gate(
            "PROXY_BARS",
            "PASS" if proxy_ok else "FAIL",
            proxy_reason,
            as_of=as_of,
            hashes=tuple(inputs.bar_hashes.get(symbol, "") for symbol in PROXY_SYMBOLS if inputs.bar_hashes.get(symbol)),
        )
    ]
    fred_ok, ovx, ovx_p90, dollar_change, fred_reason = _fred_state(inputs.fred, as_of)
    gates.append(
        _gate(
            "FRED_REFERENCES",
            "PASS" if fred_ok else "FAIL",
            fred_reason,
            as_of=as_of,
            hashes=inputs.fred.raw_sha256 if inputs.fred else (),
        )
    )
    eia_hashes = tuple(digest for batch in inputs.eia for digest in batch.raw_sha256)
    eia_ok = len(inputs.eia) == 4 and all(
        datetime.fromisoformat(str(batch.metadata["published_at"])) <= as_of.astimezone(UTC)
        for batch in inputs.eia
    )
    gates.append(
        _gate(
            "EIA_WPSR",
            "PASS" if eia_ok else "FAIL",
            "four official WPSR tables are causally available" if eia_ok else "complete causal WPSR tables unavailable",
            as_of=latest_eia_release_at(as_of) if eia_ok else None,
            hashes=eia_hashes,
        )
    )
    cftc_ok = False
    cftc_at: datetime | None = None
    if inputs.cftc is not None and not inputs.cftc.frame.empty:
        available = pd.to_datetime(inputs.cftc.frame["available_at"], utc=True)
        usable = inputs.cftc.frame.loc[available <= pd.Timestamp(as_of)]
        if not usable.empty:
            cftc_at = pd.Timestamp(usable["available_at"].iloc[-1]).to_pydatetime()
            cftc_ok = as_of.astimezone(UTC) - cftc_at <= timedelta(days=14)
    gates.append(
        _gate(
            "CFTC_POSITIONING",
            "PASS" if cftc_ok else "FAIL",
            "latest Tuesday WTI observation is available after Friday 15:30 ET" if cftc_ok else "causal CFTC WTI observation unavailable or stale",
            as_of=cftc_at,
            hashes=inputs.cftc.raw_sha256 if inputs.cftc else (),
        )
    )
    context = OilContextStore(root / "oil" / "context.json").read(at=as_of)
    context_ok = context is not None
    gates.append(
        _gate(
            "OPERATOR_CONTEXT",
            "PASS" if context_ok else "FAIL",
            "unexpired spread, financing, and event-risk context supplied" if context_ok else "daily operator context is missing or expired",
            as_of=context.recorded_at if context else None,
        )
    )
    gates.extend(
        (
            _gate("EXACT_ETORO_HISTORY", "DATA_UNAVAILABLE", "USO proxy cannot reproduce eToro rolling-CFD basis"),
            _gate("FUTURES_CURVE", "DATA_UNAVAILABLE", "no free point-in-time futures-curve history is used in v1"),
            _gate("VALIDATED_INTRADAY", "DATA_UNAVAILABLE", "daily proxy bars expose open/close anchors, not validated hour-level execution"),
        )
    )

    uso: pd.DataFrame | None = None
    raw_uso = inputs.bars.get("USO")
    if raw_uso is not None:
        candidate_uso = _causal_frame(raw_uso, as_of)
        if len(candidate_uso) >= 30:
            uso = candidate_uso
    reference_price: float | None = None
    atr14: float | None = None
    p99_gap = 0.0
    latest_gap_atr = 0.0
    if uso is not None:
        reference_price = float(uso["adjusted_close"].iloc[-1])
        atr14, p99_gap, latest_gap_atr = _atr_and_gap(uso)
        signal_session = pd.Timestamp(uso["session"].iloc[-1]).date()
    else:
        signal_session = expected
    entry_session = calendar.next_session(signal_session).date()
    sessions = calendar.sessions(entry_session, entry_session + timedelta(days=12))
    swing_exit = sessions[3].date()

    ledger = OilLedger(root / "oil" / "forward.sqlite") if persist else None
    current_equity: dict[str, float] = {}
    peaks: dict[str, float] = {}
    daily_loss: dict[str, float] = {}
    open_risk: dict[str, float] = {}
    terminated: set[str] = set()
    if ledger is not None:
        if uso is not None:
            costs = cast(dict[str, Any], resolved["costs"])
            ledger.reconcile_proxy_bars(
                uso,
                at=as_of,
                stop_slippage_bps=float(costs["pessimistic_stop_slippage_bps"]),
            )
        current_equity, peaks = ledger.lane_state(initial_equity_usd=paper_equity_usd)
        daily_loss, open_risk = ledger.risk_usage(at=as_of)
        terminated = ledger.terminated_challenge_lanes(
            initial_equity_usd=paper_equity_usd
        )
    spread = context.spread_bps if context else float(decision_cfg["maximum_spread_bps"]) + 1.0
    fee = context.overnight_fee_usd_per_unit if context else 0.0
    if reference_price is not None and atr14 is not None:
        intraday_lanes = size_risk_lanes(
            equity_usd=paper_equity_usd,
            equity_by_lane=current_equity,
            peak_equity_by_lane=peaks,
            daily_loss_by_lane=daily_loss,
            open_risk_by_lane=open_risk,
            terminated_lanes=terminated,
            price_usd=reference_price,
            atr14_usd=atr14,
            p99_adverse_gap_fraction=p99_gap,
            spread_bps=spread,
            overnight_fee_usd_per_unit=fee,
            holding_nights=0,
        )
        swing_lanes = size_risk_lanes(
            equity_usd=paper_equity_usd,
            equity_by_lane=current_equity,
            peak_equity_by_lane=peaks,
            daily_loss_by_lane=daily_loss,
            open_risk_by_lane=open_risk,
            terminated_lanes=terminated,
            price_usd=reference_price,
            atr14_usd=atr14,
            p99_adverse_gap_fraction=p99_gap,
            spread_bps=spread,
            overnight_fee_usd_per_unit=fee,
            holding_nights=(swing_exit - entry_session).days,
        )
    else:
        missing_reason = "USO price/ATR inputs unavailable; no lane can be sized"
        intraday_lanes = unavailable_risk_lanes(
            equity_usd=paper_equity_usd,
            equity_by_lane=current_equity,
            peak_equity_by_lane=peaks,
            terminated_lanes=terminated,
            reason=missing_reason,
        )
        swing_lanes = intraday_lanes

    common_vetoes: list[str] = []
    for gate in gates:
        if gate.name in REQUIRED_GATES and gate.status != "PASS":
            common_vetoes.append(f"{gate.name}:{gate.reason}")
    if context is not None and context.event_risk != "NORMAL":
        common_vetoes.append(f"EVENT_RISK_{context.event_risk}")
    if context is not None and context.spread_bps > float(decision_cfg["maximum_spread_bps"]):
        common_vetoes.append("ABNORMAL_SPREAD")
    if _eia_window(as_of):
        common_vetoes.append("EIA_RELEASE_WINDOW_10_15_TO_11_15_ET")
    if latest_gap_atr > float(decision_cfg["maximum_entry_gap_atr"]):
        common_vetoes.append("ENTRY_GAP_ABOVE_1_5_ATR")
    if fred_ok and ovx is not None and ovx_p90 is not None and ovx > ovx_p90:
        common_vetoes.append("OVX_ABOVE_EXPANDING_90TH_PERCENTILE")
    if agreement != "BULLISH":
        common_vetoes.append(f"PROXY_AGREEMENT_{agreement}")
    if ledger is not None and ledger.has_open_position():
        common_vetoes.append("ONE_OIL_POSITION_ALREADY_OPEN")
    baseline = {date.fromisoformat(str(item)) for item in decision_cfg.get("baseline_no_trade_sessions", [])}
    if entry_session in baseline:
        common_vetoes.append("FROZEN_JULY_20_BASELINE_NO_TRADE")

    adjusted = uso["adjusted_close"].astype(float) if uso is not None else pd.Series(dtype=float)
    trend = len(adjusted) >= 61 and adjusted.iloc[-1] > adjusted.iloc[-61:].mean()
    momentum_long = len(adjusted) >= 21 and adjusted.iloc[-1] > adjusted.iloc[-21]
    dollar_support = dollar_change is not None and dollar_change <= 0
    composite_long = sum((trend, momentum_long, agreement == "BULLISH", dollar_support)) >= 3
    if not composite_long:
        common_vetoes.append("FIXED_COMPOSITE_LONG_SIGNAL_ABSENT")

    intraday_frozen, intraday_evidence = _freeze_status(root, campaign_id, "INTRADAY")
    swing_frozen, swing_evidence = _freeze_status(root, campaign_id, "SWING_3D")
    intraday_specific = list(common_vetoes)
    swing_specific = list(common_vetoes)
    if _crosses_weekend(entry_session, swing_exit):
        swing_specific.append("SWING_EXPOSURE_CROSSES_WEEKEND")
    if not intraday_specific and not swing_specific and intraday_frozen and swing_frozen:
        intraday_specific.append("MUTUALLY_EXCLUSIVE_SWING_3D_SELECTED")
    intraday_vetoes = tuple(dict.fromkeys(intraday_specific))
    swing_vetoes = tuple(dict.fromkeys(swing_specific))

    def status(vetoes: tuple[str, ...], frozen: bool) -> str:
        if vetoes:
            return "NO_TRADE"
        return "PAPER_LONG" if frozen else "WATCH"

    intraday_status = status(intraday_vetoes, intraday_frozen)
    swing_status = status(swing_vetoes, swing_frozen)
    intraday = OilHorizonDecision(
        horizon="INTRADAY",
        status=intraday_status,
        evidence_status="FORWARD_TRACKING" if intraday_frozen else "FORWARD_REQUIRED",
        signal_session=signal_session.isoformat(),
        planned_entry=f"{entry_session.isoformat()} USO open",
        planned_exit=f"{entry_session.isoformat()} USO close",
        reference_price_usd=reference_price,
        atr14_usd=atr14,
        p99_adverse_gap_fraction=p99_gap,
        active_vetoes=intraday_vetoes,
        reasons=(intraday_evidence, "USO open-to-close; daily bars do not justify a finer clock time"),
        lanes=intraday_lanes,
    )
    swing = OilHorizonDecision(
        horizon="SWING_3D",
        status=swing_status,
        evidence_status="FORWARD_TRACKING" if swing_frozen else "FORWARD_REQUIRED",
        signal_session=signal_session.isoformat(),
        planned_entry=f"{entry_session.isoformat()} USO open",
        planned_exit=f"{swing_exit.isoformat()} USO close after three sessions",
        reference_price_usd=reference_price,
        atr14_usd=atr14,
        p99_adverse_gap_fraction=p99_gap,
        active_vetoes=swing_vetoes,
        reasons=(swing_evidence, "three-session swing is the preregistered default"),
        lanes=swing_lanes,
    )
    overall = "PAPER_LONG" if "PAPER_LONG" in {intraday_status, swing_status} else "WATCH" if "WATCH" in {intraday_status, swing_status} else "NO_TRADE"
    identity = canonical_sha256(
        {
            "campaign": campaign_id,
            "as_of": as_of.astimezone(UTC).isoformat(),
            "market": signal_session.isoformat(),
            "equity": paper_equity_usd,
            "intraday": intraday.model_dump(mode="json"),
            "swing": swing.model_dump(mode="json"),
        }
    )
    snapshot = OilSnapshot(
        campaign_id=campaign_id,
        decision_id=f"oil-{signal_session.isoformat()}-{identity[:16]}",
        generated_at=as_of,
        market_as_of=signal_session.isoformat(),
        status=overall,
        watermark="PAPER_ONLY_NOT_AN_ORDER_HIGH_RISK_CHALLENGE_IS_NON_PROMOTABLE",
        basis_warning=str(resolved["basis_warning"]),
        proxy_agreement=agreement,
        data_gates=tuple(gates),
        intraday=intraday,
        swing=swing,
        provenance_warnings=tuple(dict.fromkeys(inputs.warnings)),
    )
    if persist:
        assert ledger is not None
        ledger.record_snapshot(snapshot)
        _write_latest(root / "oil" / "latest.json", snapshot)
    return snapshot


__all__ = [
    "OilDecisionInputs",
    "build_oil_snapshot",
    "fetch_oil_inputs",
    "load_oil_config",
]
