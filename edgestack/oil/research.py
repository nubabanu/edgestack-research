"""Preregistered 72-rule oil diagnostic and forward-freeze boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from edgestack.data.calendars import NYSECalendar
from edgestack.data.factors import ReferenceBatch
from edgestack.edges._study_common import evaluate_family
from edgestack.oil.data import OilReferenceBatch, signed_price_observations
from edgestack.oil.decision import (
    OilDecisionInputs,
    fetch_oil_inputs,
    load_oil_config,
)
from edgestack.provenance import canonical_sha256
from edgestack.storage.artifacts import ArtifactStore

SIGNALS = (
    "trend_20",
    "trend_60",
    "trend_200",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "reversal_1",
    "reversal_5",
    "brent_wti_relative_20",
    "energy_equity_confirmation_5",
    "ovx_below_expanding_p90",
    "dollar_down_20",
    "eia_crude_draw",
    "eia_cushing_draw",
    "cftc_managed_money_rising",
    "weekday_tuesday",
    "composite_trend_confirmation",
    "composite_full",
)
HORIZONS: Mapping[str, int] = {
    "INTRADAY": 1,
    "SWING_1D": 1,
    "SWING_3D": 3,
    "SWING_5D": 5,
}


def _indexed_adjusted(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    result["session"] = pd.to_datetime(result["session"]).dt.normalize()
    result = result.sort_values("session", kind="stable").set_index("session")
    factor = result["adjusted_close"].astype(float) / result["close"].astype(float)
    result["adjusted_open"] = result["open"].astype(float) * factor
    return result


def _decision_times(index: pd.DatetimeIndex) -> pd.DataFrame:
    calendar = NYSECalendar()
    schedule = calendar.schedule(index.min(), index.max()).reindex(index)
    return pd.DataFrame(
        {
            "session": index,
            "decision_at": pd.to_datetime(schedule["market_close"], utc=True).to_numpy(),
        }
    ).sort_values("decision_at")


def _align_fred(
    batch: ReferenceBatch | None,
    series_id: str,
    sessions: pd.DatetimeIndex,
) -> pd.Series:
    if batch is None or series_id not in batch.frame:
        return pd.Series(np.nan, index=sessions, dtype=float)
    availability = f"{series_id}__available_at"
    observations = batch.frame.loc[
        batch.frame[series_id].notna(), [series_id, availability]
    ].copy()
    observations["available_at"] = pd.to_datetime(observations[availability], utc=True)
    observations = observations.sort_values("available_at")
    decisions = _decision_times(sessions)
    decisions["decision_at"] = pd.to_datetime(decisions["decision_at"], utc=True)
    aligned = pd.merge_asof(
        decisions,
        observations[["available_at", series_id]],
        left_on="decision_at",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    return pd.Series(
        aligned[series_id].to_numpy(dtype=float), index=aligned["session"], dtype=float
    ).reindex(sessions)


def _align_cftc(
    batch: OilReferenceBatch | None, sessions: pd.DatetimeIndex
) -> pd.Series:
    if batch is None or batch.frame.empty:
        return pd.Series(np.nan, index=sessions, dtype=float)
    observations = batch.frame[["available_at", "managed_money_net"]].dropna().copy()
    observations["available_at"] = pd.to_datetime(observations["available_at"], utc=True)
    observations = observations.sort_values("available_at")
    decisions = _decision_times(sessions)
    decisions["decision_at"] = pd.to_datetime(decisions["decision_at"], utc=True)
    aligned = pd.merge_asof(
        decisions,
        observations,
        left_on="decision_at",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    return pd.Series(
        aligned["managed_money_net"].to_numpy(dtype=float),
        index=aligned["session"],
        dtype=float,
    ).reindex(sessions)


def _align_eia_history(
    batch: OilReferenceBatch | None,
    series_id: str,
    sessions: pd.DatetimeIndex,
) -> pd.Series:
    if batch is None or series_id not in batch.frame:
        return pd.Series(np.nan, index=sessions, dtype=float)
    availability = f"{series_id}__available_at"
    observations = batch.frame.loc[
        batch.frame[series_id].notna(), [series_id, availability]
    ].copy()
    observations["available_at"] = pd.to_datetime(observations[availability], utc=True)
    observations = observations.sort_values("available_at")
    decisions = _decision_times(sessions)
    decisions["decision_at"] = pd.to_datetime(decisions["decision_at"], utc=True)
    aligned = pd.merge_asof(
        decisions,
        observations[["available_at", series_id]],
        left_on="decision_at",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    return pd.Series(
        aligned[series_id].to_numpy(dtype=float),
        index=aligned["session"],
        dtype=float,
    ).reindex(sessions)


def _features(inputs: OilDecisionInputs, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    uso = _indexed_adjusted(inputs.bars["USO"]).reindex(sessions)
    close = uso["adjusted_close"].astype(float)
    xle = _indexed_adjusted(inputs.bars["XLE"])["adjusted_close"].reindex(sessions)
    xop = _indexed_adjusted(inputs.bars["XOP"])["adjusted_close"].reindex(sessions)
    wti = _align_fred(inputs.fred, "DCOILWTICO", sessions)
    brent = _align_fred(inputs.fred, "DCOILBRENTEU", sessions)
    ovx = _align_fred(inputs.fred, "OVXCLS", sessions)
    dollar = _align_fred(inputs.fred, "DTWEXBGS", sessions)
    crude = _align_eia_history(inputs.eia_history, "WCESTUS1", sessions)
    cushing = _align_eia_history(
        inputs.eia_history, "W_EPC0_SAX_YCUOK_MBBL", sessions
    )
    # Synthetic/legacy fixtures can still supply the old reference columns;
    # live research uses the official hash-retained EIA workbooks above.
    if crude.isna().all():
        crude = _align_fred(inputs.fred, "WCESTUS1", sessions)
    if cushing.isna().all():
        cushing = _align_fred(inputs.fred, "WCSSTUS1", sessions)
    cot = _align_cftc(inputs.cftc, sessions)
    positive_spot = (wti > 0) & (brent > 0)
    relative = (brent.pct_change(20, fill_method=None) > wti.pct_change(20, fill_method=None)) & positive_spot
    energy = (xle.pct_change(5, fill_method=None) > 0) & (
        xop.pct_change(5, fill_method=None) > 0
    )
    ovx_p90 = ovx.shift(1).expanding(min_periods=252).quantile(0.90)
    feature = pd.DataFrame(index=sessions)
    feature["trend_20"] = close > close.rolling(20, min_periods=20).mean()
    feature["trend_60"] = close > close.rolling(60, min_periods=60).mean()
    feature["trend_200"] = close > close.rolling(200, min_periods=200).mean()
    feature["momentum_5"] = close.pct_change(5, fill_method=None) > 0
    feature["momentum_20"] = close.pct_change(20, fill_method=None) > 0
    feature["momentum_60"] = close.pct_change(60, fill_method=None) > 0
    feature["reversal_1"] = close.pct_change(fill_method=None) < 0
    feature["reversal_5"] = close.pct_change(5, fill_method=None) < 0
    feature["brent_wti_relative_20"] = relative
    feature["energy_equity_confirmation_5"] = energy
    feature["ovx_below_expanding_p90"] = ovx < ovx_p90
    feature["dollar_down_20"] = dollar.pct_change(20, fill_method=None) < 0
    feature["eia_crude_draw"] = crude.diff() < 0
    feature["eia_cushing_draw"] = cushing.diff() < 0
    feature["cftc_managed_money_rising"] = cot.diff() > 0
    feature["weekday_tuesday"] = sessions.weekday == 1
    feature["composite_trend_confirmation"] = feature["trend_60"] & energy
    feature["composite_full"] = (
        feature["momentum_20"]
        & relative
        & energy
        & feature["ovx_below_expanding_p90"]
        & feature["dollar_down_20"]
    )
    return feature.fillna(False).astype(bool)


def build_oil_streams(
    config: Mapping[str, Any],
    inputs: OilDecisionInputs,
    end_exclusive: date,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], pd.Series]:
    """Build all 72 causal open-to-close/one/three/five-session trials."""

    uso = _indexed_adjusted(inputs.bars["USO"])
    uso = uso.loc[uso.index < pd.Timestamp(end_exclusive)]
    sessions = pd.DatetimeIndex(uso.index)
    feature = _features(inputs, sessions)
    open_ = uso["adjusted_open"].astype(float)
    close = uso["adjusted_close"].astype(float)
    cost = float(cast(Mapping[str, Any], config["costs"])["research_round_trip_bps"]) / 10_000.0
    gross_streams: dict[str, pd.Series] = {}
    net_streams: dict[str, pd.Series] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for signal in SIGNALS:
        active = feature[signal].to_numpy(dtype=bool)
        for horizon, exit_offset in HORIZONS.items():
            output = pd.Series(np.nan, index=sessions, dtype=float)
            for index in range(len(sessions) - exit_offset):
                if active[index]:
                    entry = float(open_.iloc[index + 1])
                    exit_price = float(close.iloc[index + exit_offset])
                    if entry > 0 and exit_price > 0:
                        output.iloc[index + exit_offset] = exit_price / entry - 1.0
            trial_id = f"oil|{horizon}|{signal}"
            gross_streams[trial_id] = output
            net_streams[trial_id] = output - cost
            definitions[trial_id] = {
                "signal": signal,
                "horizon": horizon,
                "holding_sessions": exit_offset,
                "entry": "NEXT_USO_OPEN",
                "exit": "USO_CLOSE",
                "outcome_proxy": "USO",
                "basis": "NOT_ETORO_ROLLING_WTI_CFD",
            }
    gross = pd.DataFrame(gross_streams)
    net = pd.DataFrame(net_streams)
    declared = int(cast(Mapping[str, Any], config["research"])["declared_real_candidates"])
    if len(definitions) != declared or declared != len(SIGNALS) * len(HORIZONS):
        raise RuntimeError("oil candidate enumeration differs from preregistration")
    benchmark = close.pct_change(fill_method=None)
    return gross, net, definitions, benchmark


def _write_governance(
    *,
    root: Path,
    config: Mapping[str, Any],
    result_path: Path,
) -> Path:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    trials = {str(item["trial_id"]): item for item in result["trials"]}
    survivors = [str(item) for item in result["survivors"]]
    rules: dict[str, dict[str, Any]] = {}
    if result.get("preholdout_pass") is True:
        for horizon in ("INTRADAY", "SWING_3D"):
            eligible = [item for item in survivors if f"|{horizon}|" in item]
            if eligible:
                selected = min(
                    eligible,
                    key=lambda item: (
                        -float(trials[item].get("max_drawdown") or -1.0),
                        -float(trials[item].get("net_mean_bp") or -1e9),
                        item,
                    ),
                )
                rules[horizon] = {
                    "candidate_id": selected,
                    "definition": {
                        "signal": trials[selected]["signal"],
                        "entry": "NEXT_USO_OPEN",
                        "holding_sessions": trials[selected]["holding_sessions"],
                    },
                    "historical_status": "DIAGNOSTIC_ONLY",
                    "promotion_status": "FORWARD_REQUIRED",
                }
    research = cast(Mapping[str, Any], config["research"])
    governance = {
        "campaign_id": config["campaign_id"],
        "historical_status": "DIAGNOSTIC_ONLY_ALREADY_INSPECTED",
        "forward_start": research["forward_start"],
        "forward_end": research["forward_end"],
        "freeze_limit_per_horizon_family": 1,
        "rules": rules,
        "ten_percent_lane": "HIGH_RISK_NON_PROMOTABLE",
        "research_result": str(result_path),
    }
    governance["freeze_sha256"] = canonical_sha256(governance)
    artifact = ArtifactStore(root / "artifacts" / "campaigns" / str(config["campaign_id"]))
    return artifact.write_json("oil/freeze.json", governance)


def run_oil_research(
    config_path: str | Path = "configs/oil-paper-v1.yaml",
    *,
    root: str | Path = ".",
) -> Path:
    """Fetch, evaluate, and freeze no more than one rule per horizon family."""

    base = Path(root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = base / path
    config = load_oil_config(path)
    research = cast(Mapping[str, Any], config["research"])
    forward_start = date.fromisoformat(str(research["forward_start"]))
    fetch_time = datetime.combine(
        forward_start, time.min, tzinfo=ZoneInfo("America/New_York")
    )
    inputs = asyncio.run(
        fetch_oil_inputs(
            as_of=fetch_time,
            artifact_root=base / "artifacts",
            config=config,
        )
    )
    if (
        set(inputs.bars) != {"USO", "BNO", "XLE", "XOP"}
        or inputs.fred is None
        or inputs.eia_history is None
    ):
        raise RuntimeError(
            "oil research cannot proceed without complete proxy/FRED/EIA history"
        )
    gross, net, definitions, benchmark = build_oil_streams(config, inputs, forward_start)

    def rebuild(end_exclusive: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        rebuilt_gross, rebuilt_net, _, _ = build_oil_streams(
            config, inputs, end_exclusive
        )
        return rebuilt_gross, rebuilt_net

    result_path = evaluate_family(
        campaign_id=str(config["campaign_id"]),
        config_path=path,
        root=base,
        net=net,
        gross=gross,
        definitions=definitions,
        accounting_family_size=int(research["accounting_candidates"]),
        forward_start=forward_start,
        rebuild=rebuild,
        benchmark=benchmark,
    )
    if inputs.fred is not None:
        signed = signed_price_observations(inputs.fred)
        rows = [
            {
                "series_id": item.series_id,
                "session": item.session,
                "event_time": item.event_time,
                "available_at": item.available_at,
                "value": item.value,
                "source": item.source,
                "raw_sha256": item.raw_sha256,
            }
            for item in signed
        ]
        ArtifactStore(base / "artifacts" / "campaigns" / str(config["campaign_id"])).write_parquet(
            "oil/signed_spot_observations.parquet", pd.DataFrame(rows)
        )
    return _write_governance(root=base, config=config, result_path=result_path)


__all__ = ["HORIZONS", "SIGNALS", "build_oil_streams", "run_oil_research"]
