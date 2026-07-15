"""Frozen empirical replication checks for pipeline validation.

These functions evaluate preregistered definitions and thresholds. A failure is
evidence that promotion must stop; none of the functions retune a sample or
tolerance in response to the observed data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as student_t  # type: ignore[import-untyped]

from edgestack.features.calendar_feats import turn_of_month
from edgestack.features.sessions import decompose_sessions
from edgestack.stats._types import BoolArray, FloatArray
from edgestack.stats.tests import (
    automatic_newey_west_lag,
    hac_mean_test,
    newey_west_long_run_variance,
)


@dataclass(frozen=True, slots=True)
class MeanDifference:
    """HAC difference between event and comparison means."""

    event_mean: float
    comparison_mean: float
    difference: float
    standard_error: float
    t_stat: float
    p_value_one_sided_greater: float
    event_count: int
    comparison_count: int


def event_mean_difference(
    returns: FloatArray | pd.Series,
    event_mask: BoolArray | pd.Series,
    *,
    lags: int | None = None,
) -> MeanDifference:
    """Estimate a HAC event-minus-rest mean difference.

    The influence-function representation retains calendar spacing and serial
    dependence, unlike comparing two compressed event arrays as if iid.
    """

    values = np.asarray(returns, dtype=float)
    mask = np.asarray(event_mask, dtype=bool)
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError(
            "returns and event_mask must be aligned one-dimensional arrays"
        )
    finite = np.isfinite(values)
    event = finite & mask
    rest = finite & ~mask
    event_count = int(event.sum())
    rest_count = int(rest.sum())
    if event_count < 2 or rest_count < 2:
        return MeanDifference(
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            event_count,
            rest_count,
        )
    event_mean = float(values[event].mean())
    rest_mean = float(values[rest].mean())
    difference = event_mean - rest_mean
    sample = values[finite]
    sample_event = mask[finite]
    probability = float(sample_event.mean())
    influence = np.where(
        sample_event,
        (sample - event_mean) / probability,
        -(sample - rest_mean) / (1.0 - probability),
    )
    selected_lags = automatic_newey_west_lag(len(sample)) if lags is None else lags
    selected_lags = min(max(selected_lags, 0), len(sample) - 1)
    long_run_variance = newey_west_long_run_variance(influence, selected_lags)
    standard_error = math.sqrt(long_run_variance / len(sample))
    if standard_error == 0.0:
        t_stat = math.copysign(math.inf, difference) if difference else 0.0
    else:
        t_stat = difference / standard_error
    p_value = float(student_t.sf(t_stat, df=len(sample) - 1))
    return MeanDifference(
        event_mean,
        rest_mean,
        difference,
        standard_error,
        t_stat,
        p_value,
        event_count,
        rest_count,
    )


@dataclass(frozen=True, slots=True)
class ReplicationCheck:
    """One immutable gate outcome with diagnostics."""

    name: str
    passed: bool
    statistic: float | None
    threshold: str
    details: Mapping[str, float | int | str | bool]
    citation: str
    limitation: str | None = None


@dataclass(frozen=True, slots=True)
class ReplicationSuiteResult:
    """All six checks and aggregate all-must-pass gate."""

    checks: tuple[ReplicationCheck, ...]

    @property
    def passed(self) -> bool:
        """Return true only when all frozen checks pass."""

        return len(self.checks) == 6 and all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[str, ...]:
        """Names of failed replication checks."""

        return tuple(check.name for check in self.checks if not check.passed)

    @property
    def executed(self) -> bool:
        """Return true only when all six checks produced numerical evidence."""

        return len(self.checks) == 6 and all(
            check.statistic is not None and not math.isnan(float(check.statistic))
            for check in self.checks
        )


def replicate_turn_of_month(market_returns: pd.Series) -> ReplicationCheck:
    """Replicate McConnell and Xu's last-through-third-session TOM premium."""

    if not isinstance(market_returns.index, pd.DatetimeIndex):
        raise TypeError("market_returns needs a DatetimeIndex of exchange sessions")
    mask = turn_of_month(market_returns.index).to_numpy()
    result = event_mean_difference(market_returns.to_numpy(), mask)
    passed = bool(result.difference > 0.0 and result.t_stat > 2.0)
    return ReplicationCheck(
        "turn_of_month",
        passed,
        result.t_stat,
        "TOM minus rest > 0 with one-sided HAC t > 2",
        {
            "tom_mean": result.event_mean,
            "rest_mean": result.comparison_mean,
            "difference": result.difference,
            "t_stat": result.t_stat,
            "tom_days": result.event_count,
        },
        "McConnell & Xu (2008)",
    )


def replicate_pre_fomc(
    spy_close_to_close_returns: pd.Series,
    fomc_dates: pd.DatetimeIndex | list[object],
) -> ReplicationCheck:
    """Test the frozen 1994-2013 daily FOMC proxy.

    Daily prior-close-to-meeting-close includes post-announcement return and is
    therefore explicitly not the exact Lucca-Moench intraday 24-hour window.
    """

    series = spy_close_to_close_returns.sort_index()
    historical = series.loc[
        (series.index >= "1994-01-01") & (series.index <= "2013-12-31")
    ]
    dates = pd.DatetimeIndex(fomc_dates).normalize()
    historical_index = pd.DatetimeIndex(historical.index)
    selected = historical_index.normalize().isin(dates)
    event_returns = historical.to_numpy(dtype=float)[selected]
    test = hac_mean_test(event_returns, lags=0, alternative="greater")
    passed = bool(test.mean > 0.0 and test.t_stat > 2.0)
    post = series.loc[series.index >= "2014-01-01"]
    post_index = pd.DatetimeIndex(post.index)
    post_values = post.to_numpy(dtype=float)[post_index.normalize().isin(dates)]
    post_mean = (
        float(np.nanmean(post_values)) if np.isfinite(post_values).any() else math.nan
    )
    return ReplicationCheck(
        "pre_fomc_daily_proxy",
        passed,
        test.t_stat,
        "1994-2013 proxy mean > 0 with t > 2",
        {
            "historical_mean": test.mean,
            "historical_t": test.t_stat,
            "historical_events": test.n_observations,
            "post_2013_mean": post_mean,
        },
        "Lucca & Moench (2015)",
        limitation="prior close to meeting-day close includes post-announcement contamination",
    )


def replicate_overnight_dominance(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    minimum_total_share: float = 0.75,
) -> ReplicationCheck:
    """Check SPY and QQQ overnight log return dominance."""

    details: dict[str, float | int | str | bool] = {}
    symbol_passes: list[bool] = []
    margins: list[float] = []
    for symbol in ("SPY", "QQQ"):
        if symbol not in bars_by_symbol:
            raise ValueError(f"missing {symbol} bars")
        bars = bars_by_symbol[symbol]
        sessions = decompose_sessions(bars["open"], bars["close"], log=True)
        overnight = float(np.nansum(np.asarray(sessions.overnight, dtype=float)))
        intraday = float(np.nansum(np.asarray(sessions.intraday, dtype=float)))
        total = overnight + intraday
        share = overnight / total if total > 0.0 else math.nan
        symbol_pass = bool(overnight > intraday and share >= minimum_total_share)
        symbol_passes.append(symbol_pass)
        margins.append(overnight - intraday)
        details.update(
            {
                f"{symbol}_overnight_log": overnight,
                f"{symbol}_intraday_log": intraday,
                f"{symbol}_overnight_share": share,
            }
        )
    return ReplicationCheck(
        "overnight_dominance",
        all(symbol_passes),
        min(margins),
        f"SPY and QQQ overnight > intraday and >= {minimum_total_share:.0%} of total log return",
        details,
        "Lou, Polk & Skouras (2019)",
    )


def replicate_momentum(momentum_spread_returns: pd.Series) -> ReplicationCheck:
    """Check positive long-run momentum and a >=20% rolling-63-day 2009 crash."""

    values = momentum_spread_returns.astype(float)
    mean = float(values.mean())
    rolling = (1.0 + values).rolling(63, min_periods=63).apply(np.prod, raw=True) - 1.0
    crash_2009 = rolling.loc[
        (rolling.index >= "2009-01-01") & (rolling.index <= "2009-12-31")
    ]
    minimum = float(crash_2009.min()) if crash_2009.notna().any() else math.nan
    passed = bool(mean > 0.0 and math.isfinite(minimum) and minimum <= -0.20)
    return ReplicationCheck(
        "momentum",
        passed,
        mean,
        "long-run mean > 0 and 2009 rolling-63-session loss <= -20%",
        {
            "mean": mean,
            "worst_2009_63_session_return": minimum,
            "observations": int(values.notna().sum()),
        },
        "Jegadeesh & Titman (1993)",
    )


def replicate_short_term_reversal(
    gross_returns: FloatArray | pd.Series,
    net_returns: FloatArray | pd.Series,
) -> ReplicationCheck:
    """Check gross reversal and preregistered transaction-cost damage."""

    gross = np.asarray(gross_returns, dtype=float)
    net = np.asarray(net_returns, dtype=float)
    if gross.shape != net.shape:
        raise ValueError("gross and net reversal returns must be aligned")
    gross_mean = float(np.nanmean(gross))
    net_mean = float(np.nanmean(net))
    damaged = net_mean <= 0.0 or net_mean <= 0.5 * gross_mean
    passed = bool(gross_mean > 0.0 and damaged)
    return ReplicationCheck(
        "short_term_reversal",
        passed,
        net_mean,
        "gross mean > 0 and net mean <= 0 or <=50% of gross",
        {
            "gross_mean": gross_mean,
            "net_mean": net_mean,
            "net_to_gross": net_mean / gross_mean if gross_mean else math.nan,
        },
        "Jegadeesh (1990); Novy-Marx & Velikov (2016)",
    )


def replicate_monday_effect(market_returns: pd.Series) -> ReplicationCheck:
    """Check an early negative Monday premium and its post-1990 death."""

    series = market_returns.sort_index().astype(float)

    def period_test(period: pd.Series) -> MeanDifference:
        monday = pd.DatetimeIndex(period.index).dayofweek == 0
        return event_mean_difference(period.to_numpy(), monday)

    early = period_test(series.loc[series.index < "1975-01-01"])
    late = period_test(series.loc[series.index >= "1990-01-01"])
    passed = bool(
        early.difference < 0.0
        and early.t_stat < -2.0
        and abs(late.t_stat) < 1.96
        and abs(late.difference) < 0.0005
    )
    return ReplicationCheck(
        "monday_effect_death",
        passed,
        late.t_stat,
        "pre-1975 Monday-minus-other t < -2; post-1990 |t| < 1.96 and |mean| < 5 bps/day",
        {
            "early_difference": early.difference,
            "early_t": early.t_stat,
            "late_difference": late.difference,
            "late_t": late.t_stat,
        },
        "French (1980), decay replication",
    )


def run_replication_suite(
    *,
    market_returns: pd.Series,
    spy_returns: pd.Series,
    fomc_dates: pd.DatetimeIndex | list[object],
    bars_by_symbol: Mapping[str, pd.DataFrame],
    momentum_returns: pd.Series,
    reversal_gross: FloatArray | pd.Series,
    reversal_net: FloatArray | pd.Series,
) -> ReplicationSuiteResult:
    """Run all six frozen checks; the caller must stop promotion on failure."""

    return ReplicationSuiteResult(
        (
            replicate_turn_of_month(market_returns),
            replicate_pre_fomc(spy_returns, fomc_dates),
            replicate_overnight_dominance(bars_by_symbol),
            replicate_momentum(momentum_returns),
            replicate_short_term_reversal(reversal_gross, reversal_net),
            replicate_monday_effect(market_returns),
        )
    )


turn_of_month_check = replicate_turn_of_month
pre_fomc_check = replicate_pre_fomc
overnight_check = replicate_overnight_dominance
momentum_check = replicate_momentum
reversal_check = replicate_short_term_reversal
monday_check = replicate_monday_effect
