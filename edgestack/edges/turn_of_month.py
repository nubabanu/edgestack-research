"""Preregistered SPY turn-of-month study and single-use holdout ceremony.

The strategy has one rule and no fitted parameters.  Preholdout and holdout
commands use mutually exclusive Parquet predicates.  The holdout command
consumes its SQLite authorization before reading any holdout return.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml

from edgestack.backtest.costs import (
    CostAssumptions,
    CostModel,
    MarketContext,
    TradeIntent,
)
from edgestack.data.calendars import NYSECalendar
from edgestack.disclaimer import DISCLAIMER
from edgestack.models import GateResult, GateStatus, HoldoutFreezeManifest
from edgestack.pipeline.holdout import HoldoutGuard
from edgestack.provenance import (
    canonical_sha256,
    sha256_file,
    source_tree_sha256,
)
from edgestack.stats.bootstrap import stationary_bootstrap_ci
from edgestack.stats.tests import hac_mean_test
from edgestack.storage.catalog import Catalog


@dataclass(frozen=True, slots=True)
class TurnOfMonthEpisode:
    """One fully completed, non-overlapping turn-of-month position."""

    entry_session: date
    first_exposure_session: date
    exit_session: date
    entry_adjusted_close: float
    exit_adjusted_close: float
    gross_return: float
    baseline_cost_bps: float
    net_return: float
    net_return_4x_cost: float


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("edge configuration must be a mapping")
    return cast(dict[str, Any], payload)


def _cost_model(config: Mapping[str, Any]) -> CostModel:
    costs = cast(Mapping[str, Any], config["costs"])
    strategy = cast(Mapping[str, Any], config["strategy"])
    return CostModel(
        CostAssumptions(
            portfolio_capital=float(strategy["allocation_usd"]),
            commission_per_side=float(costs["commission_per_side_usd"]),
            etf_full_spread_bps=float(costs["etf_full_spread_bps"]),
            base_slippage_bps=float(costs["base_slippage_bps_per_fill"]),
            impact_coefficient_bps=float(costs["impact_coefficient_bps"]),
            max_impact_bps=float(costs["impact_cap_bps_per_fill"]),
            turnover_penalty_bps=float(
                costs["selection_penalty_bps_per_100pct_one_way_turnover"]
            ),
        )
    )


def load_bars(
    path: str | Path,
    *,
    symbol: str,
    start: date | None = None,
    end_exclusive: date | None = None,
) -> pd.DataFrame:
    """Read only the requested symbol/date region from canonical Parquet."""

    filters: list[tuple[str, str, object]] = [("symbol", "=", symbol)]
    if start is not None:
        filters.append(("session", ">=", datetime.combine(start, time.min)))
    if end_exclusive is not None:
        filters.append(("session", "<", datetime.combine(end_exclusive, time.min)))
    table = pq.read_table(
        path,
        columns=[
            "symbol",
            "session",
            "close",
            "adjusted_close",
            "volume",
            "source",
        ],
        filters=filters,
    )
    frame = table.to_pandas()
    if frame.empty:
        raise ValueError(f"no {symbol} bars in requested interval")
    frame["session"] = (
        pd.to_datetime(frame["session"]).dt.tz_localize(None).dt.normalize()
    )
    frame = frame.sort_values("session").drop_duplicates("session", keep=False)
    if frame["session"].duplicated().any():
        raise ValueError("duplicate session rows remain after canonical load")
    return cast(pd.DataFrame, frame.reset_index(drop=True))


def _side_cost_bps(
    model: CostModel,
    *,
    allocation_usd: float,
    adv_dollars: float,
) -> float:
    estimate = model.estimate(
        TradeIntent(
            order_dollars=allocation_usd,
            holding_days=0.0,
            fills=1,
            one_way_turnover=1.0,
            order_type="MOC",
        ),
        MarketContext(adv_dollars=adv_dollars, asset_type="etf"),
    )
    return estimate.total_bps


def build_turn_of_month_episodes(
    bars: pd.DataFrame,
    *,
    allocation_usd: float = 10_000.0,
    adv_floor_usd: float = 1_000_000.0,
    cost_model: CostModel | None = None,
) -> tuple[TurnOfMonthEpisode, ...]:
    """Construct exact last-one/first-three episodes from an NYSE calendar.

    Entry is the close immediately before the final session of a month.  Thus
    the four exposed close-to-close returns are the final session and the first
    three sessions of the next month.  Only episodes whose entry and exit bars
    are present are returned; an incomplete boundary month is never relabeled.
    """

    required = {"session", "close", "adjusted_close", "volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing required fields: {sorted(missing)}")
    if allocation_usd <= 0.0 or adv_floor_usd <= 0.0:
        raise ValueError("allocation and ADV floor must be positive")
    frame = bars.copy()
    frame["session"] = (
        pd.to_datetime(frame["session"]).dt.tz_localize(None).dt.normalize()
    )
    frame = frame.sort_values("session")
    if frame["session"].duplicated().any():
        raise ValueError("bars must have unique sessions")
    for column in ("close", "adjusted_close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if (frame["adjusted_close"] <= 0.0).any() or frame["adjusted_close"].isna().any():
        raise ValueError("adjusted closes must be finite and positive")

    first = cast(pd.Timestamp, frame["session"].iloc[0])
    last = cast(pd.Timestamp, frame["session"].iloc[-1])
    calendar = NYSECalendar()
    sessions = calendar.sessions(first.date(), last.date())
    locations = {session: position for position, session in enumerate(sessions)}
    indexed = frame.set_index("session", drop=False)
    model = cost_model or CostModel(CostAssumptions(portfolio_capital=allocation_usd))

    output: list[TurnOfMonthEpisode] = []
    periods = pd.period_range(first.to_period("M"), last.to_period("M"), freq="M")
    for period in periods[:-1]:
        month_sessions = sessions[sessions.to_period("M") == period]
        next_period = period + 1
        following = sessions[sessions.to_period("M") == next_period]
        if month_sessions.empty or len(following) < 3:
            continue
        first_exposure = month_sessions[-1]
        position = locations[first_exposure]
        if position == 0:
            continue
        entry = sessions[position - 1]
        exit_session = following[2]
        required_sessions = (
            entry,
            first_exposure,
            following[0],
            following[1],
            exit_session,
        )
        if any(session not in indexed.index for session in required_sessions):
            continue
        entry_row = indexed.loc[entry]
        exit_row = indexed.loc[exit_session]
        entry_price = float(entry_row["adjusted_close"])
        exit_price = float(exit_row["adjusted_close"])
        gross_factor = exit_price / entry_price
        entry_adv = max(float(entry_row["close"] * entry_row["volume"]), adv_floor_usd)
        exit_adv = max(float(exit_row["close"] * exit_row["volume"]), adv_floor_usd)
        entry_cost = _side_cost_bps(
            model, allocation_usd=allocation_usd, adv_dollars=entry_adv
        )
        exit_cost = _side_cost_bps(
            model, allocation_usd=allocation_usd, adv_dollars=exit_adv
        )
        total_cost = entry_cost + exit_cost

        baseline_entry_fraction = entry_cost / 10_000.0
        baseline_exit_fraction = exit_cost / 10_000.0
        four_x_entry_fraction = 4.0 * entry_cost / 10_000.0
        four_x_exit_fraction = 4.0 * exit_cost / 10_000.0
        baseline_net = (
            gross_factor
            * (1.0 - baseline_entry_fraction)
            * (1.0 - baseline_exit_fraction)
            - 1.0
        )
        four_x_net = (
            gross_factor * (1.0 - four_x_entry_fraction) * (1.0 - four_x_exit_fraction)
            - 1.0
        )

        output.append(
            TurnOfMonthEpisode(
                entry_session=entry.date(),
                first_exposure_session=first_exposure.date(),
                exit_session=exit_session.date(),
                entry_adjusted_close=entry_price,
                exit_adjusted_close=exit_price,
                gross_return=gross_factor - 1.0,
                baseline_cost_bps=total_cost,
                net_return=baseline_net,
                net_return_4x_cost=four_x_net,
            )
        )
    return tuple(output)


def evaluate_episode_sample(
    episodes: Sequence[TurnOfMonthEpisode],
    *,
    config: Mapping[str, Any],
    include_preholdout_gates: bool,
) -> dict[str, Any]:
    """Evaluate a completed episode sample without fitting any parameter."""

    if not episodes:
        raise ValueError("at least one completed episode is required")
    net = np.asarray([episode.net_return for episode in episodes], dtype=float)
    gross = np.asarray([episode.gross_return for episode in episodes], dtype=float)
    four_x = np.asarray(
        [episode.net_return_4x_cost for episode in episodes], dtype=float
    )
    hac = hac_mean_test(net, alternative="greater")
    annual = (
        pd.DataFrame(
            {
                "year": [episode.exit_session.year for episode in episodes],
                "net": net,
            }
        )
        .groupby("year", sort=True)["net"]
        .apply(lambda values: float(np.prod(1.0 + values) - 1.0))
    )
    result: dict[str, Any] = {
        "episodes": len(episodes),
        "start_entry": episodes[0].entry_session.isoformat(),
        "end_exit": episodes[-1].exit_session.isoformat(),
        "gross_mean": float(gross.mean()),
        "net_mean": float(net.mean()),
        "median_net": float(np.median(net)),
        "hit_rate": float(np.mean(net > 0.0)),
        "net_standard_deviation": float(net.std(ddof=1)) if len(net) > 1 else math.nan,
        "hac": hac.as_dict(),
        "four_x_cost_net_mean": float(four_x.mean()),
        "average_baseline_roundtrip_cost_bps": float(
            np.mean([episode.baseline_cost_bps for episode in episodes])
        ),
        "positive_year_fraction": float(np.mean(annual.to_numpy() > 0.0)),
        "annual_returns": {str(year): value for year, value in annual.items()},
    }
    if not include_preholdout_gates:
        result["holdout_pass"] = bool(result["net_mean"] > 0.0)
        return result

    validation = cast(Mapping[str, Any], config["preholdout_validation"])
    bootstrap = stationary_bootstrap_ci(
        net,
        statistic="mean",
        confidence=0.95,
        n_resamples=int(validation["bootstrap_draws"]),
        average_block_length=float(validation["stationary_average_block_episodes"]),
        seed=int(validation["bootstrap_seed"]),
    )
    era_metrics: dict[str, dict[str, Any]] = {}
    for era in cast(Sequence[Mapping[str, Any]], validation["fixed_eras"]):
        era_start = date.fromisoformat(str(era["start"]))
        era_end = date.fromisoformat(str(era["end"]))
        selected = np.asarray(
            [
                episode.net_return
                for episode in episodes
                if era_start <= episode.exit_session <= era_end
            ],
            dtype=float,
        )
        if selected.size == 0:
            raise ValueError(f"fixed era {era['name']} has no observations")
        era_metrics[str(era["name"])] = {
            "episodes": int(selected.size),
            "net_mean": float(selected.mean()),
            "hit_rate": float(np.mean(selected > 0.0)),
        }
    recent = era_metrics["RECENT_CONFIRMATION"]["net_mean"]
    prior_median = float(
        np.median(
            [
                era_metrics["PRE_PUBLICATION"]["net_mean"],
                era_metrics["POST_PUBLICATION_1"]["net_mean"],
            ]
        )
    )
    recent_ratio = float(recent / prior_median) if prior_median > 0.0 else -math.inf
    checks = {
        "minimum_episodes": len(episodes) >= int(validation["minimum_episodes"]),
        "hac_t": hac.t_stat > float(validation["full_sample_one_sided_hac_t_minimum"]),
        "bootstrap_lower_positive": bootstrap.lower > 0.0,
        "every_fixed_era_positive": all(
            metric["net_mean"] > 0.0 for metric in era_metrics.values()
        ),
        "positive_year_fraction": result["positive_year_fraction"]
        > float(validation["positive_year_fraction_minimum"]),
        "recent_effect_retained": recent_ratio
        >= float(validation["recent_to_prior_era_median_minimum"]),
        "four_x_cost_positive": result["four_x_cost_net_mean"] > 0.0,
    }
    result.update(
        {
            "bootstrap_mean_95": asdict(bootstrap),
            "fixed_eras": era_metrics,
            "recent_to_prior_era_median": recent_ratio,
            "checks": checks,
            "preholdout_pass": all(checks.values()),
        }
    )
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    return sha256_file(path)


def _artifact_root(root: Path, campaign_id: str) -> Path:
    return root / "artifacts" / "campaigns" / campaign_id


def _parent_tom_pass(root: Path, config: Mapping[str, Any]) -> tuple[bool, str]:
    relative = Path(
        cast(Mapping[str, Any], config["data"])["replication_evidence_path"]
    )
    path = root / relative
    evidence = json.loads(path.read_text(encoding="utf-8"))
    checks = evidence.get("checks", [])
    selected = [item for item in checks if item.get("name") == "turn_of_month"]
    return bool(len(selected) == 1 and selected[0].get("passed") is True), sha256_file(
        path
    )


def run_preholdout(config_path: str | Path, *, root: str | Path = ".") -> Path:
    """Run and persist validation while making a holdout read impossible."""

    base = Path(root).resolve()
    config_file = (base / config_path).resolve()
    config = _load_config(config_file)
    data = cast(Mapping[str, Any], config["data"])
    cutoff = date.fromisoformat(str(data["preholdout_end_exclusive"]))
    bars = load_bars(
        base / str(data["bars_path"]),
        symbol=str(data["symbol"]),
        end_exclusive=cutoff,
    )
    if cast(pd.Timestamp, bars["session"].max()).date() >= cutoff:
        raise RuntimeError("preholdout loader exposed a holdout session")
    episodes = build_turn_of_month_episodes(
        bars,
        allocation_usd=float(
            cast(Mapping[str, Any], config["strategy"])["allocation_usd"]
        ),
        adv_floor_usd=float(cast(Mapping[str, Any], config["costs"])["adv_floor_usd"]),
        cost_model=_cost_model(config),
    )
    result = evaluate_episode_sample(
        episodes, config=config, include_preholdout_gates=True
    )
    parent_pass, parent_sha = _parent_tom_pass(base, config)
    result["checks"]["parent_turn_of_month_replication"] = parent_pass
    result["preholdout_pass"] = bool(result["preholdout_pass"] and parent_pass)
    result.update(
        {
            "campaign_id": config["campaign_id"],
            "status": "PASS" if result["preholdout_pass"] else "FAIL",
            "data_boundary": f"session < {cutoff.isoformat()}",
            "parent_replication_sha256": parent_sha,
            "config_sha256": sha256_file(config_file),
            "disclaimer": DISCLAIMER,
        }
    )
    artifact = _artifact_root(base, str(config["campaign_id"]))
    _write_json(artifact / "preholdout" / "result.json", result)
    pd.DataFrame(asdict(episode) for episode in episodes).to_csv(
        artifact / "preholdout" / "episodes.csv", index=False
    )
    print(DISCLAIMER)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "episodes",
                    "net_mean",
                    "hac",
                    "bootstrap_mean_95",
                    "fixed_eras",
                    "four_x_cost_net_mean",
                )
            },
            indent=2,
        )
    )
    return artifact / "preholdout" / "result.json"


def freeze(config_path: str | Path, *, root: str | Path = ".") -> Path:
    """Seal the exact passing rule, source, costs, and data identity."""

    base = Path(root).resolve()
    config_file = (base / config_path).resolve()
    config = _load_config(config_file)
    campaign_id = str(config["campaign_id"])
    artifact = _artifact_root(base, campaign_id)
    result_path = artifact / "preholdout" / "result.json"
    preholdout = json.loads(result_path.read_text(encoding="utf-8"))
    if preholdout.get("preholdout_pass") is not True:
        raise RuntimeError("cannot freeze a strategy that failed preholdout validation")
    data = cast(Mapping[str, Any], config["data"])
    strategy = cast(Mapping[str, Any], config["strategy"])
    data_manifest_path = base / str(data["data_manifest_path"])
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    bars_path = base / str(data["bars_path"])
    universe_path = base / str(data["universe_path"])
    edge_identity = {
        "strategy": strategy,
        "data_symbol": data["symbol"],
        "return_field": data["return_field"],
        "holdout_start": data["holdout_start"],
        "holdout_end": data["holdout_end"],
    }
    edge_id = canonical_sha256(edge_identity)
    identity = {
        "campaign_id": campaign_id,
        "edge_id": edge_id,
        "config_sha256": sha256_file(config_file),
        "preholdout_result_sha256": sha256_file(result_path),
        "bars_sha256": sha256_file(bars_path),
        "universe_sha256": sha256_file(universe_path),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "source_tree_sha256": source_tree_sha256(base),
        "lock_sha256": sha256_file(base / "uv.lock"),
    }
    freeze_id = canonical_sha256(identity)
    manifest = HoldoutFreezeManifest(
        campaign_id=campaign_id,
        freeze_id=freeze_id,
        frozen_at=datetime.now(UTC),
        edge_ids=(edge_id,),
        specs_sha256=identity["config_sha256"],
        stack_sha256=canonical_sha256({"edges": [edge_id], "weights": [1.0]}),
        overlay_sha256=canonical_sha256({"enabled": []}),
        cost_sha256=canonical_sha256(config["costs"]),
        config_sha256=identity["config_sha256"],
        bars_sha256=identity["bars_sha256"],
        universe_sha256=identity["universe_sha256"],
        data_manifest_sha256=identity["data_manifest_sha256"],
        source_tree_sha256=identity["source_tree_sha256"],
        lock_sha256=identity["lock_sha256"],
        model_mapping_sha256=canonical_sha256(edge_identity),
        data_snapshot_id=str(data_manifest["snapshot_id"]),
    )
    payload = asdict(manifest)
    payload["preholdout_result_sha256"] = identity["preholdout_result_sha256"]
    payload["disclaimer"] = DISCLAIMER
    freeze_path = artifact / "freeze" / "manifest.json"
    freeze_sha = _write_json(freeze_path, payload)

    catalog = Catalog(base / "artifacts" / "edgestack.sqlite")
    existing = catalog.campaign(campaign_id)
    campaign_manifest = {
        "campaign_id": campaign_id,
        "parent_campaign_id": config["parent_campaign_id"],
        "config_sha256": identity["config_sha256"],
        "data_snapshot_id": manifest.data_snapshot_id,
        "holdout_start": data["holdout_start"],
        "holdout_end": data["holdout_end"],
        "strategy": strategy["name"],
    }
    if existing is None:
        catalog.create_campaign(campaign_id, campaign_manifest)
    elif existing != campaign_manifest:
        raise RuntimeError("registered campaign identity does not match freeze")
    catalog.record_gate(
        GateResult(
            campaign_id=campaign_id,
            phase="edge_preholdout",
            status=GateStatus.PASS,
            checked_at=datetime.now(UTC),
            summary="single preregistered SPY turn-of-month rule passed every frozen preholdout check",
            evidence={"result_sha256": identity["preholdout_result_sha256"]},
        )
    )
    catalog.record_artifact(campaign_id, "edge_freeze", freeze_sha, freeze_path)
    print(DISCLAIMER)
    print(
        json.dumps(
            {
                "freeze_id": freeze_id,
                "manifest": str(freeze_path),
                "sha256": freeze_sha,
            },
            indent=2,
        )
    )
    return freeze_path


def _load_verified_freeze(
    base: Path, config: Mapping[str, Any]
) -> tuple[HoldoutFreezeManifest, Mapping[str, Any]]:
    campaign_id = str(config["campaign_id"])
    path = _artifact_root(base, campaign_id) / "freeze" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = cast(Mapping[str, Any], config["data"])
    expected = {
        "config_sha256": sha256_file(base / "configs" / "spy-tom-edge-v1.yaml"),
        "bars_sha256": sha256_file(base / str(data["bars_path"])),
        "universe_sha256": sha256_file(base / str(data["universe_path"])),
        "data_manifest_sha256": sha256_file(base / str(data["data_manifest_path"])),
        "source_tree_sha256": source_tree_sha256(base),
        "lock_sha256": sha256_file(base / "uv.lock"),
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise RuntimeError(f"freeze identity mismatch: {', '.join(mismatches)}")
    fields = {key: payload[key] for key in HoldoutFreezeManifest.__dataclass_fields__}
    fields["frozen_at"] = datetime.fromisoformat(str(fields["frozen_at"]))
    fields["edge_ids"] = tuple(fields["edge_ids"])
    return HoldoutFreezeManifest(**fields), payload


def run_holdout(config_path: str | Path, *, root: str | Path = ".") -> Path:
    """Consume the one authorization, evaluate, seal, and thereafter replay."""

    base = Path(root).resolve()
    config = _load_config(base / config_path)
    campaign_id = str(config["campaign_id"])
    catalog = Catalog(base / "artifacts" / "edgestack.sqlite")
    result_path = _artifact_root(base, campaign_id) / "holdout" / "result.json"
    access = catalog.holdout_access(campaign_id)
    if access is not None:
        if access.result_sha256 is None or not result_path.exists():
            raise RuntimeError(
                "holdout authorization was consumed but no sealed result is available"
            )
        if sha256_file(result_path) != access.result_sha256:
            raise RuntimeError("stored holdout result does not match its sealed hash")
        print(DISCLAIMER)
        print(result_path.read_text(encoding="utf-8"))
        return result_path

    catalog.require_passed(campaign_id, ["edge_preholdout"])
    freeze_manifest, freeze_payload = _load_verified_freeze(base, config)
    data = cast(Mapping[str, Any], config["data"])
    strategy = cast(Mapping[str, Any], config["strategy"])
    start = date.fromisoformat(str(data["holdout_start"]))
    end = date.fromisoformat(str(data["holdout_end"]))
    guard = HoldoutGuard(catalog)
    with guard.authorize(freeze_manifest):
        bars = load_bars(
            base / str(data["bars_path"]),
            symbol=str(data["symbol"]),
            start=start,
            end_exclusive=end.fromordinal(end.toordinal() + 1),
        )
        if cast(pd.Timestamp, bars["session"].min()).date() < start:
            raise RuntimeError("holdout loader crossed its lower boundary")
        episodes = build_turn_of_month_episodes(
            bars,
            allocation_usd=float(strategy["allocation_usd"]),
            adv_floor_usd=float(
                cast(Mapping[str, Any], config["costs"])["adv_floor_usd"]
            ),
            cost_model=_cost_model(config),
        )
        result = evaluate_episode_sample(
            episodes, config=config, include_preholdout_gates=False
        )
        result.update(
            {
                "campaign_id": campaign_id,
                "freeze_id": freeze_manifest.freeze_id,
                "freeze_manifest_sha256": sha256_file(
                    _artifact_root(base, campaign_id) / "freeze" / "manifest.json"
                ),
                "status": "PASS" if result["holdout_pass"] else "FAIL",
                "data_boundary": f"{start.isoformat()} <= session <= {end.isoformat()}",
                "overlays": "NOT_APPLICABLE_NONE_ENABLED",
                "second_evaluation": "REJECTED_REPLAY_ONLY",
                "disclaimer": DISCLAIMER,
            }
        )
        result_sha = _write_json(result_path, result)
        pd.DataFrame(asdict(episode) for episode in episodes).to_csv(
            result_path.parent / "episodes.csv", index=False
        )
        guard.complete(campaign_id, result_sha)
        catalog.record_artifact(
            campaign_id, "edge_holdout_result", result_sha, result_path
        )
        catalog.record_gate(
            GateResult(
                campaign_id=campaign_id,
                phase="edge_holdout",
                status=GateStatus.PASS if result["holdout_pass"] else GateStatus.FAIL,
                checked_at=datetime.now(UTC),
                summary=(
                    "frozen SPY turn-of-month holdout net mean is positive"
                    if result["holdout_pass"]
                    else "frozen SPY turn-of-month holdout net mean is nonpositive"
                ),
                evidence={
                    "result_sha256": result_sha,
                    "freeze_id": freeze_payload["freeze_id"],
                },
            )
        )
    print(DISCLAIMER)
    print(json.dumps(result, indent=2))
    return result_path


def next_trade(config_path: str | Path, *, root: str | Path = ".") -> dict[str, Any]:
    """Return the next calendar-known entry/exit only after holdout success."""

    base = Path(root).resolve()
    config = _load_config(base / config_path)
    campaign_id = str(config["campaign_id"])
    catalog = Catalog(base / "artifacts" / "edgestack.sqlite")
    catalog.require_passed(campaign_id, ["edge_preholdout", "edge_holdout"])
    data = cast(Mapping[str, Any], config["data"])
    as_of = date.fromisoformat(str(data["holdout_end"]))
    calendar = NYSECalendar()
    sessions = calendar.sessions(as_of, date(as_of.year + 1, 1, 31))
    periods = sessions.to_period("M")
    for period in periods.unique():
        current = sessions[periods == period]
        following_period = period + 1
        following = sessions[periods == following_period]
        if current.empty or len(following) < 3:
            continue
        first_exposure = current[-1]
        prior = sessions[sessions < first_exposure]
        if prior.empty:
            continue
        entry = prior[-1]
        if entry.date() <= as_of:
            continue
        exit_session = following[2]
        output = {
            "state": "WAIT",
            "symbol": data["symbol"],
            "direction": "LONG",
            "entry_session": entry.date().isoformat(),
            "entry_order": "MOC",
            "first_exposure_session": first_exposure.date().isoformat(),
            "exit_session": exit_session.date().isoformat(),
            "exit_order": "MOC",
            "maximum_allocation_usd": cast(Mapping[str, Any], config["strategy"])[
                "allocation_usd"
            ],
            "sizing": cast(Mapping[str, Any], config["action_policy"])["size_rule"],
            "stop": cast(Mapping[str, Any], config["action_policy"])["stop_overlay"],
            "disclaimer": DISCLAIMER,
        }
        print(DISCLAIMER)
        print(json.dumps(output, indent=2))
        return output
    raise RuntimeError("no complete future turn-of-month episode found")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preholdout", "freeze", "holdout", "next-trade")
    )
    parser.add_argument("--config", default="configs/spy-tom-edge-v1.yaml")
    parser.add_argument("--root", default=".")
    arguments = parser.parse_args(argv)
    if arguments.command == "preholdout":
        run_preholdout(arguments.config, root=arguments.root)
    elif arguments.command == "freeze":
        freeze(arguments.config, root=arguments.root)
    elif arguments.command == "holdout":
        run_holdout(arguments.config, root=arguments.root)
    else:
        next_trade(arguments.config, root=arguments.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
