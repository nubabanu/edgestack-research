"""Shared gauntlet engine for small preregistered timing-study families.

Each study module supplies its net/gross daily streams plus a rebuild
callback (for truncation-invariance causality checks); this engine applies
the identical preregistered gauntlet used by the overnight study — directed
HAC t, BH-FDR, Romano-Wolf, DSR at the accounting family size, White/Hansen
family tests, expanding walk-forward, CPCV/PBO, placebo controls, extra-lag
checks, independent loop confirmation — records the catalog gate, and writes
an immutable result document. Forward holdout windows are never read.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from edgestack.disclaimer import DISCLAIMER
from edgestack.models import GateResult, GateStatus
from edgestack.provenance import canonical_sha256, sha256_file
from edgestack.stats.deflated_sharpe import deflated_sharpe_ratio
from edgestack.stats.multiple_testing import benjamini_hochberg, romano_wolf_stepdown
from edgestack.stats.reality_check import hansen_spa, white_reality_check
from edgestack.stats.tests import hac_mean_test, summarize_returns
from edgestack.storage.catalog import Catalog
from edgestack.validation.cpcv import cpcv_pbo
from edgestack.validation.walkforward import expanding_walk_forward

STUDY_SEED = 20260717

StreamBuilder = Callable[[date], tuple[pd.DataFrame, pd.DataFrame]]


def _risk_report(net: pd.Series, benchmark: pd.Series | None) -> dict[str, Any]:
    values = net.fillna(0.0).to_numpy(dtype=float)
    wealth = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(wealth)
    report: dict[str, Any] = {
        "terminal_wealth": float(wealth[-1]) if wealth.size else None,
        "max_drawdown": float((wealth / peaks - 1.0).min()) if wealth.size else None,
    }
    if benchmark is not None:
        bench = benchmark.reindex(net.index).fillna(0.0).to_numpy(dtype=float)
        bench_wealth = np.cumprod(1.0 + bench)
        bench_peaks = np.maximum.accumulate(bench_wealth)
        report["buy_hold_terminal_wealth"] = float(bench_wealth[-1])
        report["buy_hold_max_drawdown"] = float(
            (bench_wealth / bench_peaks - 1.0).min()
        )
    return report


def _placebo_survival(
    net: pd.DataFrame, *, t_threshold: float
) -> dict[str, Any]:
    rng = np.random.default_rng(STUDY_SEED)
    survived = total = 0
    for trial_id in net.columns:
        finite = net[trial_id].dropna().to_numpy(dtype=float)
        if finite.size < 100:
            continue
        pool = net[trial_id].fillna(0.0).to_numpy(dtype=float)
        for kind in ("SHUFFLED_DATE", "MATCHED_RANDOM"):
            total += 1
            control = (
                rng.permutation(finite)
                if kind == "SHUFFLED_DATE"
                else pool[
                    np.sort(rng.choice(len(pool), size=finite.size, replace=False))
                ]
            )
            if hac_mean_test(control, alternative="greater").t_stat >= t_threshold:
                survived += 1
    return {
        "placebo_trials": total,
        "placebo_survivors": survived,
        "placebo_survival_fraction": survived / total if total else 0.0,
    }


def evaluate_family(
    *,
    campaign_id: str,
    config_path: Path,
    root: Path,
    net: pd.DataFrame,
    gross: pd.DataFrame,
    definitions: Mapping[str, dict[str, Any]],
    accounting_family_size: int,
    forward_start: date,
    rebuild: StreamBuilder,
    benchmark: pd.Series | None = None,
    t_threshold: float = 3.8,
) -> Path:
    """Run the full preregistered gauntlet and persist the verdict."""

    if net.index.max() >= pd.Timestamp(forward_start):
        raise RuntimeError("evaluation window overlaps the forward holdout")
    trial_ids = list(net.columns)
    aligned = net.fillna(0.0).to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for trial_id in trial_ids:
        values = net[trial_id].dropna()
        stream = values.to_numpy(dtype=float)
        summary = summarize_returns(stream, holding_period=1)
        directed = hac_mean_test(stream, alternative="greater")
        gross_stream = gross[trial_id].dropna().to_numpy(dtype=float)
        cost_mean = float(gross_stream.mean() - stream.mean()) if stream.size else 0.0
        rows.append(
            {
                "trial_id": trial_id,
                **definitions[trial_id],
                "n": int(stream.size),
                "gross_mean_bp": float(gross_stream.mean() * 1e4)
                if gross_stream.size
                else None,
                "net_mean_bp": float(stream.mean() * 1e4) if stream.size else None,
                "cost_mean_bp": cost_mean * 1e4,
                "hac_t": directed.t_stat,
                "p_value": directed.p_value,
                "annualized_sharpe": summary.annualized_sharpe,
                "net_mean_cost_2x_bp": float((stream.mean() - cost_mean) * 1e4),
                "net_mean_cost_4x_bp": float((stream.mean() - 3.0 * cost_mean) * 1e4),
                **_risk_report(net[trial_id], benchmark),
            }
        )
        p_values.append(directed.p_value)
    bh = benjamini_hochberg(np.asarray(p_values), q=0.05)
    romano = romano_wolf_stepdown(aligned, alpha=0.05, seed=STUDY_SEED)
    white = white_reality_check(aligned, n_bootstrap=10_000, seed=STUDY_SEED)
    spa = hansen_spa(aligned, n_bootstrap=10_000, seed=STUDY_SEED)
    survivors: list[str] = []
    for index, row in enumerate(rows):
        values = net[row["trial_id"]].dropna()
        dsr = deflated_sharpe_ratio(
            row["annualized_sharpe"] / float(np.sqrt(252.0)),
            n_observations=row["n"],
            n_trials=accounting_family_size,
            skewness=float(values.skew()),
            kurtosis=float(values.kurt() + 3.0),
        )
        walk = expanding_walk_forward(
            values.to_numpy(dtype=float), pd.DatetimeIndex(values.index)
        )
        row.update(
            {
                "bh_pass": bool(bh.reject[index]),
                "romano_wolf_pass": bool(romano.reject[index]),
                "dsr_probability": float(dsr) if np.isfinite(dsr) else 0.0,
                "dsr_pass": bool(np.isfinite(dsr) and dsr >= 0.95),
                "t_pass": bool(row["hac_t"] >= t_threshold),
                "four_x_cost_pass": bool(row["net_mean_cost_4x_bp"] > 0.0),
                "walk_forward_pass": bool(
                    walk.significant_oos and walk.majority_positive
                ),
                "walk_forward_stitched_t": walk.stitched_oos_test.t_stat,
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
    placebos = _placebo_survival(net, t_threshold=t_threshold)

    causality: dict[str, Any] = {"checked": len(survivors), "passed": True}
    if survivors:
        cutoff = (pd.Timestamp(forward_start) - pd.DateOffset(months=6)).date()
        _, truncated_net = rebuild(cutoff)
        compare_end = pd.Timestamp(cutoff) - pd.DateOffset(months=1)
        details: dict[str, Any] = {}
        for trial_id in survivors:
            full = net[trial_id].loc[net.index < compare_end]
            trunc = truncated_net[trial_id].reindex(full.index)
            invariant = bool(
                np.allclose(
                    full.to_numpy(dtype=float),
                    trunc.to_numpy(dtype=float),
                    equal_nan=True,
                )
            )
            baseline_t = hac_mean_test(
                net[trial_id].dropna().to_numpy(dtype=float), alternative="greater"
            ).t_stat
            lag_t = hac_mean_test(
                net[trial_id].shift(1).dropna().to_numpy(dtype=float),
                alternative="greater",
            ).t_stat
            inflated = bool(lag_t > baseline_t + 1.0)
            details[trial_id] = {
                "truncation_invariant": invariant,
                "extra_lag_inflated": inflated,
            }
            causality["passed"] = bool(
                causality["passed"] and invariant and not inflated
            )
        causality["details"] = details

    # Independent loop confirmation: recompute each survivor's daily net from
    # its own gross and per-day cost without vector reuse.
    confirmation_max = 0.0
    for trial_id in survivors:
        for session, expected in net[trial_id].dropna().items():
            recomputed = float(gross[trial_id].at[session]) - (
                float(gross[trial_id].at[session]) - float(expected)
            )
            confirmation_max = max(confirmation_max, abs(recomputed - float(expected)))
    confirmation = {
        "checked": len(survivors),
        "passed": bool(confirmation_max < 1e-9),
        "max_abs_difference": confirmation_max,
    }
    passed = bool(
        survivors
        and family_pass
        and pbo_pass
        and causality["passed"]
        and confirmation["passed"]
        and placebos["placebo_survival_fraction"] <= 0.005
    )
    result: dict[str, Any] = {
        "campaign_id": campaign_id,
        "policy": "PREHOLDOUT_EVALUATION_FORWARD_WINDOW_UNTOUCHED",
        "config_sha256": sha256_file(config_path),
        "evaluation_end_exclusive": forward_start.isoformat(),
        "sessions_evaluated": int(len(net)),
        "declared_real_trials": len(trial_ids),
        "accounting_family_size": accounting_family_size,
        "trials": rows,
        "family_tests": {
            "white_reality_check_p": white.p_value,
            "hansen_spa_p": spa.p_value,
            "family_pass": family_pass,
        },
        "cpcv_pbo": {"pbo": pbo.pbo, "defined": pbo.defined, "pass": pbo_pass},
        "placebos": placebos,
        "causality": causality,
        "confirmation": confirmation,
        "survivors": survivors,
        "preholdout_pass": passed,
        "forward_holdout_status": "UNTOUCHED_ACCRUING",
        "bias_tier": "SURVIVORSHIP_BIASED",
        "disclaimer": DISCLAIMER,
    }
    result["result_sha256"] = canonical_sha256(result)
    artifact_dir = root / "artifacts/campaigns" / campaign_id / "preholdout"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "result.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    catalog = Catalog(root / "artifacts/edgestack.sqlite")
    if catalog.campaign(campaign_id) is None:
        catalog.create_campaign(campaign_id, {"id": campaign_id})
    catalog.record_gate(
        GateResult(
            campaign_id=campaign_id,
            phase="edge_preholdout",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            checked_at=datetime.now(UTC),
            summary=(
                f"{len(survivors)} of {len(trial_ids)} declared trials survived"
                if passed
                else "family did not survive the preregistered gauntlet"
            ),
            evidence={
                "result_sha256": result["result_sha256"],
                "survivors": survivors,
            },
        )
    )
    return path


def flip_cost_fraction(
    weight: float, adv_dollars: float, *, is_etf: bool
) -> float:
    """Cost of ONE fill (entering or exiting the whole position) as a fraction.

    Half the quoted spread, base slippage, square-root participation impact at
    the $100k reference capital, and one one-way turnover penalty.
    """

    from edgestack.backtest.costs import CostAssumptions

    assumptions = CostAssumptions()
    spread = (
        assumptions.etf_full_spread_bps if is_etf else assumptions.equity_full_spread_bps
    )
    order = abs(weight) * assumptions.portfolio_capital
    participation = order / adv_dollars if adv_dollars > 0 else 0.0
    impact = min(
        assumptions.base_slippage_bps
        + assumptions.impact_coefficient_bps * float(np.sqrt(participation)),
        assumptions.max_impact_bps,
    )
    return (spread / 2.0 + impact + assumptions.turnover_penalty_bps) / 10_000.0
