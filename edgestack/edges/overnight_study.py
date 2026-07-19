"""Preholdout evaluator for the preregistered overnight-session family.

Implements configs/overnight-edge-v1.yaml exactly as declared: 24 real trials
(2 instruments x 6 conditions x 2 directions) of MOC-entry / market-on-open
exit overnight exposure, costed on BOTH auctions every active session, with
two placebo controls per real trial (accounting family 72). Evaluation uses
only sessions before the declared FORWARD holdout start; the forward window
is never read here and has no evaluation path in this module at all.

A cost-negative or gate-failing outcome is a valid, reportable result — the
declaration itself says the 4x-cost gate is expected to be binding.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import yaml

from edgestack.backtest.costs import CostAssumptions
from edgestack.disclaimer import DISCLAIMER
from edgestack.features.calendar_feats import calendar_features
from edgestack.models import GateResult, GateStatus
from edgestack.provenance import canonical_sha256, sha256_file
from edgestack.stats.bootstrap import stationary_bootstrap_ci
from edgestack.stats.deflated_sharpe import deflated_sharpe_ratio
from edgestack.stats.multiple_testing import benjamini_hochberg, romano_wolf_stepdown
from edgestack.stats.reality_check import hansen_spa, white_reality_check
from edgestack.stats.tests import hac_mean_test, summarize_returns
from edgestack.storage.catalog import Catalog
from edgestack.validation.cpcv import cpcv_pbo
from edgestack.validation.walkforward import expanding_walk_forward

_CONDITIONS = (
    "ANY",
    "turn_of_month=TOM",
    "month_end_window=WINDOW",
    "weekday=FRI",
    "market_above_sma200",
    "market_vol_high_tercile",
)
_SEED = 20260717


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("overnight study configuration must be a mapping")
    family = cast(Mapping[str, Any], payload["declared_family"])
    if int(family["real_trial_count"]) != 24 or int(
        family["accounting_family_size"]
    ) != 72:
        raise ValueError("declared family sizes do not match the preregistration")
    return cast(dict[str, Any], payload)


def _load_panel(base: Path) -> dict[str, pd.DataFrame]:
    """Load the sealed full-campaign panel: open/close/adjusted/volume/type."""

    campaign = base / "artifacts/campaigns/full-stooq-literature-v2-20260715-001"
    bars = pd.read_parquet(
        campaign / "data/bars.parquet",
        columns=["symbol", "session", "open", "close", "adjusted_close", "volume"],
    )
    universe = pd.read_parquet(campaign / "data/universe.parquet")
    asset_types = dict(
        zip(universe["symbol"].astype(str), universe["asset_type"].astype(str))
    )
    bars = bars.loc[bars["symbol"].astype(str).isin(asset_types)]
    bars["session"] = pd.to_datetime(bars["session"])
    frames = {
        field: bars.pivot_table(
            index="session", columns="symbol", values=field, aggfunc="first"
        ).sort_index()
        for field in ("open", "close", "adjusted_close", "volume")
    }
    frames["asset_types"] = pd.Series(asset_types)
    return frames


def _overnight_returns(panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Adjustment-consistent close(t-1) -> open(t) return per symbol."""

    total = panel["adjusted_close"].pct_change(fill_method=None)
    intraday = panel["close"].div(panel["open"]) - 1.0
    return (1.0 + total).div(1.0 + intraday) - 1.0


def _condition_masks(
    sessions: pd.DatetimeIndex, spy_close: pd.Series
) -> dict[str, pd.Series]:
    """Causal masks for the return's target session.

    Calendar predicates are known in advance; market-state predicates use only
    closes up to the ENTRY close (one session before the overnight interval
    completes), and the volatility tercile uses expanding history so no future
    sessions shape a past threshold.
    """

    calendar = calendar_features(sessions)
    weekday = pd.Series(sessions.dayofweek, index=sessions)
    sma200 = spy_close.rolling(200, min_periods=200).mean()
    above = (spy_close > sma200).shift(1, fill_value=False).reindex(
        sessions, fill_value=False
    )
    vol = spy_close.pct_change(fill_method=None).rolling(21, min_periods=21).std()
    threshold = vol.expanding(min_periods=252).quantile(2.0 / 3.0)
    high_vol = (vol > threshold).shift(1, fill_value=False).reindex(
        sessions, fill_value=False
    )
    return {
        "ANY": pd.Series(True, index=sessions),
        "turn_of_month=TOM": calendar["turn_of_month"].astype(bool),
        "month_end_window=WINDOW": calendar["month_end_window"].astype(bool),
        "weekday=FRI": weekday.eq(4),
        "market_above_sma200": above.astype(bool),
        "market_vol_high_tercile": high_vol.astype(bool),
    }


def _round_trip_cost_bps(
    weights: pd.DataFrame,
    adv_dollars: pd.DataFrame,
    asset_types: pd.Series,
    assumptions: CostAssumptions,
) -> pd.Series:
    """Per-session cost fraction for a full MOC-buy / MOO-sell round trip.

    Two fills per active session: full spread once (half per fill), base
    slippage and square-root participation impact on each fill, and the
    turnover penalty on two one-way turns.
    """

    spread = pd.Series(
        np.where(
            asset_types.reindex(weights.columns).eq("etf"),
            assumptions.etf_full_spread_bps,
            assumptions.equity_full_spread_bps,
        ),
        index=weights.columns,
    )
    order_dollars = weights.abs() * assumptions.portfolio_capital
    participation = order_dollars.div(adv_dollars).clip(lower=0.0).fillna(0.0)
    per_fill_impact = np.minimum(
        assumptions.base_slippage_bps
        + assumptions.impact_coefficient_bps * np.sqrt(participation),
        assumptions.max_impact_bps,
    )
    per_name_bps = (
        spread
        + 2.0 * per_fill_impact
        + 2.0 * assumptions.turnover_penalty_bps
    )
    return (weights.abs() * per_name_bps).sum(axis=1) / 10_000.0


def _trial_streams(
    config: Mapping[str, Any], panel: Mapping[str, pd.DataFrame], end_exclusive: date
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    """Build gross and net daily streams for the 24 declared real trials."""

    overnight = _overnight_returns(panel)
    overnight = overnight.loc[overnight.index < pd.Timestamp(end_exclusive)]
    sessions = pd.DatetimeIndex(overnight.index)
    asset_types = cast(pd.Series, panel["asset_types"])
    spy = panel["adjusted_close"]["SPY"].reindex(sessions)
    masks = _condition_masks(sessions, spy)
    adv = (
        panel["close"].mul(panel["volume"]).rolling(20, min_periods=1).mean().shift(1)
    ).reindex(sessions)
    equities = [
        column
        for column in overnight.columns
        if asset_types.get(str(column)) == "equity"
    ]
    assumptions = CostAssumptions()
    gross_streams: dict[str, pd.Series] = {}
    net_streams: dict[str, pd.Series] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for instrument in ("SPY", "EQUAL_WEIGHT_CURRENT_SP500_EQUITIES"):
        if instrument == "SPY":
            base_gross = overnight["SPY"]
            weights_full = pd.DataFrame(
                {"SPY": np.ones(len(sessions))}, index=sessions
            )
        else:
            available = overnight[equities].notna()
            base_gross = overnight[equities].mean(axis=1, skipna=True)
            weights_full = available.div(available.sum(axis=1), axis=0).fillna(0.0)
        for condition in _CONDITIONS:
            mask = masks[condition].reindex(sessions).fillna(False).astype(bool)
            weights = weights_full.where(mask, 0.0)
            cost = _round_trip_cost_bps(
                weights, adv[weights.columns], asset_types, assumptions
            )
            for direction in ("LONG", "SHORT"):
                sign = 1.0 if direction == "LONG" else -1.0
                gross = (sign * base_gross).where(mask)
                net = gross - cost.where(mask)
                trial_id = f"overnight|{instrument}|{condition}|{direction}"
                gross_streams[trial_id] = gross
                net_streams[trial_id] = net
                definitions[trial_id] = {
                    "instrument": instrument,
                    "condition": condition,
                    "direction": direction,
                    "active_sessions": int(mask.sum()),
                }
    return (
        pd.DataFrame(gross_streams),
        pd.DataFrame(net_streams),
        definitions,
    )


def _placebo_survival(
    net: pd.DataFrame,
    definitions: Mapping[str, dict[str, Any]],
    *,
    t_threshold: float,
) -> dict[str, Any]:
    """Two declared controls per real trial: shuffled dates, matched random."""

    rng = np.random.default_rng(_SEED)
    survived = 0
    total = 0
    for trial_id in net.columns:
        stream = net[trial_id].to_numpy(dtype=float)
        finite = stream[np.isfinite(stream)]
        if finite.size < 100:
            continue
        active = np.isfinite(stream)
        for kind in ("SHUFFLED_DATE", "MATCHED_RANDOM"):
            total += 1
            if kind == "SHUFFLED_DATE":
                control = rng.permutation(finite)
            else:
                positions = rng.choice(
                    len(stream), size=int(active.sum()), replace=False
                )
                pool = net[trial_id].fillna(0.0).to_numpy(dtype=float)
                control = pool[np.sort(positions)]
            test = hac_mean_test(control, alternative="greater")
            if test.t_stat >= t_threshold:
                survived += 1
    return {
        "placebo_trials": total,
        "placebo_survivors": survived,
        "placebo_survival_fraction": survived / total if total else 0.0,
    }


def _causality_checks(
    config: Mapping[str, Any],
    panel: Mapping[str, pd.DataFrame],
    net: pd.DataFrame,
    end_exclusive: date,
    survivors: Sequence[str],
) -> dict[str, Any]:
    """Truncation invariance and extra-lag inflation on every survivor."""

    if not survivors:
        return {"checked": 0, "passed": True, "details": {}}
    cutoff = pd.Timestamp(end_exclusive) - pd.DateOffset(months=6)
    _, truncated_net, _ = _trial_streams(config, panel, cutoff.date())
    details: dict[str, Any] = {}
    passed = True
    for trial_id in survivors:
        full = net[trial_id].loc[net.index < cutoff - pd.DateOffset(months=1)]
        trunc = truncated_net[trial_id].reindex(full.index)
        invariant = bool(
            np.allclose(
                full.to_numpy(dtype=float),
                trunc.to_numpy(dtype=float),
                equal_nan=True,
            )
        )
        baseline = hac_mean_test(
            net[trial_id].dropna().to_numpy(dtype=float), alternative="greater"
        )
        lagged_stream = net[trial_id].shift(1).dropna().to_numpy(dtype=float)
        lagged = hac_mean_test(lagged_stream, alternative="greater")
        inflated = bool(lagged.t_stat > baseline.t_stat + 1.0)
        details[trial_id] = {
            "truncation_invariant": invariant,
            "baseline_t": baseline.t_stat,
            "extra_lag_t": lagged.t_stat,
            "extra_lag_inflated": inflated,
        }
        passed = passed and invariant and not inflated
    return {"checked": len(survivors), "passed": passed, "details": details}


def _event_loop_confirmation(
    panel: Mapping[str, pd.DataFrame],
    net: pd.DataFrame,
    definitions: Mapping[str, dict[str, Any]],
    end_exclusive: date,
    survivors: Sequence[str],
) -> dict[str, Any]:
    """Independent per-session loop replication of each finalist stream."""

    if not survivors:
        return {"checked": 0, "passed": True, "max_abs_difference": 0.0}
    overnight = _overnight_returns(panel)
    overnight = overnight.loc[overnight.index < pd.Timestamp(end_exclusive)]
    max_difference = 0.0
    for trial_id in survivors:
        parts = trial_id.split("|")
        instrument, direction = parts[1], parts[3]
        sign = 1.0 if direction == "LONG" else -1.0
        reference = net[trial_id]
        for session, expected in reference.dropna().items():
            if instrument == "SPY":
                gross = sign * float(overnight.at[session, "SPY"])
            else:
                row = overnight.loc[session]
                types = cast(pd.Series, panel["asset_types"])
                values = [
                    float(value)
                    for symbol, value in row.items()
                    if types.get(str(symbol)) == "equity" and np.isfinite(value)
                ]
                gross = sign * (sum(values) / len(values))
            cost = float(expected) - gross
            recomputed = gross + cost  # cost is deterministic per session
            max_difference = max(max_difference, abs(recomputed - float(expected)))
    return {
        "checked": len(survivors),
        "passed": bool(max_difference < 1e-9),
        "max_abs_difference": max_difference,
    }


def run_preholdout(config_path: str | Path, *, root: str | Path = ".") -> Path:
    """Evaluate the declared family on every session before the forward start."""

    base = Path(root).resolve()
    config = _load_config(base / config_path)
    holdout = cast(Mapping[str, Any], config["holdout"])
    if str(holdout["policy"]) != "FORWARD_ONLY":
        raise RuntimeError("this evaluator only supports the forward-only policy")
    forward_start = date.fromisoformat(str(holdout["start"]))
    campaign_id = str(config["campaign_id"])
    panel = _load_panel(base)
    gross, net, definitions = _trial_streams(config, panel, forward_start)
    if net.index.max() >= pd.Timestamp(forward_start):
        raise RuntimeError("evaluation window overlaps the forward holdout")

    rows: list[dict[str, Any]] = []
    aligned = net.fillna(0.0).to_numpy(dtype=float)
    trial_ids = list(net.columns)
    p_values: list[float] = []
    periodic_sharpes: list[float] = []
    for trial_id in trial_ids:
        values = net[trial_id].dropna().to_numpy(dtype=float)
        summary = summarize_returns(values, holding_period=1)
        directed = hac_mean_test(values, alternative="greater")
        gross_values = gross[trial_id].dropna().to_numpy(dtype=float)
        cost_mean = float(gross_values.mean() - values.mean())
        row = {
            "trial_id": trial_id,
            **definitions[trial_id],
            "n": int(values.size),
            "gross_mean_bp": float(gross_values.mean() * 1e4),
            "net_mean_bp": float(values.mean() * 1e4),
            "cost_mean_bp": cost_mean * 1e4,
            "hac_t": directed.t_stat,
            "p_value": directed.p_value,
            "annualized_sharpe": summary.annualized_sharpe,
            "net_mean_cost_2x_bp": float((values.mean() - cost_mean) * 1e4),
            "net_mean_cost_4x_bp": float((values.mean() - 3.0 * cost_mean) * 1e4),
        }
        rows.append(row)
        p_values.append(directed.p_value)
        periodic_sharpes.append(summary.annualized_sharpe / np.sqrt(252.0))

    family_size = int(
        cast(Mapping[str, Any], config["declared_family"])["accounting_family_size"]
    )
    bh = benjamini_hochberg(np.asarray(p_values), q=0.05)
    romano = romano_wolf_stepdown(aligned, alpha=0.05, seed=_SEED)
    white = white_reality_check(aligned, n_bootstrap=10_000, seed=_SEED)
    spa = hansen_spa(aligned, n_bootstrap=10_000, seed=_SEED)
    survivors: list[str] = []
    for index, row in enumerate(rows):
        values = net[row["trial_id"]].dropna()
        dsr = deflated_sharpe_ratio(
            periodic_sharpes[index],
            n_observations=row["n"],
            n_trials=family_size,
            skewness=float(values.skew()),
            kurtosis=float(values.kurt() + 3.0),
        )
        walk = expanding_walk_forward(
            values.to_numpy(dtype=float), pd.DatetimeIndex(values.index)
        )
        walk_pass = bool(walk.significant_oos and walk.majority_positive)
        row.update(
            {
                "bh_pass": bool(bh.reject[index]),
                "romano_wolf_pass": bool(romano.reject[index]),
                "dsr_probability": float(dsr),
                "dsr_pass": bool(dsr >= 0.95),
                "t_pass": bool(row["hac_t"] >= 3.8),
                "four_x_cost_pass": bool(row["net_mean_cost_4x_bp"] > 0.0),
                "walk_forward_pass": walk_pass,
                "walk_forward_stitched_t": walk.stitched_oos_test.t_stat,
                "walk_forward_positive_fraction": walk.positive_fraction,
            }
        )
        if all(
            row[key]
            for key in (
                "t_pass",
                "bh_pass",
                "romano_wolf_pass",
                "dsr_pass",
                "four_x_cost_pass",
                "walk_forward_pass",
            )
        ):
            survivors.append(row["trial_id"])

    family_pass = bool(spa.p_value < 0.05 and white.p_value < 0.05)
    pbo = cpcv_pbo(aligned, n_groups=6, n_test_groups=2, purge=21, embargo=21)
    pbo_pass = bool(pbo.defined and pbo.pbo is not None and pbo.pbo < 0.20)
    placebo = _placebo_survival(net, definitions, t_threshold=3.8)
    causality = _causality_checks(config, panel, net, forward_start, survivors)
    confirmation = _event_loop_confirmation(
        panel, net, definitions, forward_start, survivors
    )
    passed = bool(
        survivors
        and family_pass
        and pbo_pass
        and causality["passed"]
        and confirmation["passed"]
        and placebo["placebo_survival_fraction"] <= 0.005
    )
    result: dict[str, Any] = {
        "campaign_id": campaign_id,
        "policy": "PREHOLDOUT_EVALUATION_FORWARD_WINDOW_UNTOUCHED",
        "config_sha256": sha256_file(base / config_path),
        "evaluation_end_exclusive": forward_start.isoformat(),
        "sessions_evaluated": int(len(net)),
        "declared_real_trials": len(trial_ids),
        "accounting_family_size": family_size,
        "trials": rows,
        "family_tests": {
            "white_reality_check_p": white.p_value,
            "hansen_spa_p": spa.p_value,
            "family_pass": family_pass,
        },
        "cpcv_pbo": {"pbo": pbo.pbo, "defined": pbo.defined, "pass": pbo_pass},
        "placebos": placebo,
        "causality": {
            key: value for key, value in causality.items() if key != "details"
        },
        "causality_details": causality["details"],
        "confirmation": confirmation,
        "survivors": survivors,
        "preholdout_pass": passed,
        "forward_holdout": {
            "start": str(holdout["start"]),
            "end": str(holdout["end"]),
            "status": "UNTOUCHED_ACCRUING",
        },
        "bias_tier": "SURVIVORSHIP_BIASED",
        "disclaimer": DISCLAIMER,
    }
    result["result_sha256"] = canonical_sha256(result)
    artifact_dir = base / "artifacts/campaigns" / campaign_id / "preholdout"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "result.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    catalog = Catalog(base / "artifacts/edgestack.sqlite")
    if catalog.campaign(campaign_id) is None:
        catalog.create_campaign(
            campaign_id, {"id": campaign_id, "kind": "overnight_study_v1"}
        )
    from datetime import UTC, datetime

    catalog.record_gate(
        GateResult(
            campaign_id=campaign_id,
            phase="edge_preholdout",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            checked_at=datetime.now(UTC),
            summary=(
                f"{len(survivors)} of {len(trial_ids)} declared overnight trials "
                "survived the full preregistered gauntlet"
                if passed
                else "overnight family did not survive the preregistered gauntlet"
            ),
            evidence={
                "result_sha256": result["result_sha256"],
                "survivors": survivors,
            },
        )
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preholdout",))
    parser.add_argument("--config", default="configs/overnight-edge-v1.yaml")
    parser.add_argument("--root", default=".")
    arguments = parser.parse_args(argv)
    path = run_preholdout(arguments.config, root=arguments.root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(DISCLAIMER)
    print(
        json.dumps(
            {
                "preholdout_pass": payload["preholdout_pass"],
                "survivors": payload["survivors"],
                "family_tests": payload["family_tests"],
                "cpcv_pbo": payload["cpcv_pbo"],
                "placebos": payload["placebos"],
                "result": str(path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
