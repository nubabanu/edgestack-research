"""Rolling-window stability and preregistered decay classification."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from edgestack.models import DecayClass
from edgestack.stats._types import DateArray, FloatArray
from edgestack.stats.tests import hac_mean_test


@dataclass(frozen=True, slots=True)
class DecayPoint:
    """One trailing-window effect estimate."""

    window_end: pd.Timestamp
    window_start: pd.Timestamp
    n_observations: int
    mean: float
    t_stat: float


@dataclass(frozen=True, slots=True)
class DecayResult:
    """Rolling/fixed evidence, stability score, class, and optional death date."""

    points: tuple[DecayPoint, ...]
    stability_score: float
    classification: DecayClass
    death_date: pd.Timestamp | None
    recent_to_prior_ratio: float | None
    reason: str
    fixed_points: tuple[DecayPoint, ...] = ()
    same_sign_periods: int = 0
    eligible_periods: int = 0


def rolling_decay(
    returns: FloatArray | list[float],
    dates: pd.DatetimeIndex | DateArray | list[object],
    *,
    window_years: int = 5,
    step_months: int = 12,
    holding_period: int = 1,
    minimum_observations: int = 100,
) -> tuple[DecayPoint, ...]:
    """Compute trailing calendar-window HAC means and t-statistics."""

    values = np.asarray(returns, dtype=float)
    index = pd.DatetimeIndex(dates)
    if values.ndim != 1 or len(values) != len(index):
        raise ValueError("returns and dates must be aligned one-dimensional data")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("dates must be sorted and unique")
    if window_years < 1 or step_months < 1:
        raise ValueError("window_years and step_months must be positive")
    if len(index) == 0:
        return ()
    end = index[0] + pd.DateOffset(years=window_years)
    points: list[DecayPoint] = []
    while end <= index[-1] + pd.Timedelta(days=1):
        start = end - pd.DateOffset(years=window_years)
        selected = np.flatnonzero(
            (index >= start) & (index < end) & np.isfinite(values)
        )
        if selected.size >= minimum_observations:
            test = hac_mean_test(values[selected], holding_period=holding_period)
            points.append(
                DecayPoint(
                    index[selected[-1]],
                    index[selected[0]],
                    int(selected.size),
                    test.mean,
                    test.t_stat,
                )
            )
        end += pd.DateOffset(months=step_months)
    return tuple(points)


def fixed_decay(
    returns: FloatArray | list[float],
    dates: pd.DatetimeIndex | DateArray | list[object],
    *,
    window_years: int = 5,
    holding_period: int = 1,
    minimum_observations: int = 100,
) -> tuple[DecayPoint, ...]:
    """Compute non-overlapping, full-length calendar-window HAC estimates.

    Fixed windows are anchored to the first eligible observation.  An incomplete
    trailing window is intentionally excluded so it cannot receive the same
    weight as a complete five-year period in the frozen stability fraction.
    """

    values, index = _aligned_inputs(returns, dates)
    if window_years < 1:
        raise ValueError("window_years must be positive")
    if not index.size:
        return ()
    start = index[0]
    last_exclusive = index[-1] + pd.Timedelta(days=1)
    points: list[DecayPoint] = []
    while start + pd.DateOffset(years=window_years) <= last_exclusive:
        end = start + pd.DateOffset(years=window_years)
        selected = np.flatnonzero(
            (index >= start) & (index < end) & np.isfinite(values)
        )
        if selected.size >= minimum_observations:
            test = hac_mean_test(values[selected], holding_period=holding_period)
            points.append(
                DecayPoint(
                    index[selected[-1]],
                    index[selected[0]],
                    int(selected.size),
                    test.mean,
                    test.t_stat,
                )
            )
        start = end
    return tuple(points)


def classify_decay(
    points: tuple[DecayPoint, ...] | list[DecayPoint],
    *,
    fixed_points: tuple[DecayPoint, ...] | list[DecayPoint] = (),
    expected_sign: int = 1,
    stability_min: float = 0.75,
    regime_interaction_adjusted_p: float | None = None,
    active_regime_t: float | None = None,
) -> DecayResult:
    """Classify STABLE, DECAYING, DEAD, or REGIME_DEPENDENT.

    DEAD has precedence over full-sample or old significance. Regime dependence
    requires both an adjusted interaction p-value below .05 and active-regime
    directed t above 2.
    """

    curve = tuple(points)
    fixed = tuple(fixed_points)
    if expected_sign not in (-1, 1):
        raise ValueError("expected_sign must be +1 or -1")
    if not curve:
        return DecayResult(
            curve,
            math.nan,
            DecayClass.INSUFFICIENT,
            None,
            None,
            "no eligible rolling windows",
            fixed,
            0,
            len(fixed),
        )
    directed_means = np.asarray([point.mean * expected_sign for point in curve])
    directed_t = np.asarray([point.t_stat * expected_sign for point in curve])
    fixed_directed_means = np.asarray(
        [point.mean * expected_sign for point in fixed], dtype=float
    )
    all_period_means = np.concatenate((directed_means, fixed_directed_means))
    finite_periods = np.isfinite(all_period_means)
    eligible_periods = int(finite_periods.sum())
    same_sign_periods = int(np.count_nonzero(all_period_means[finite_periods] > 0.0))
    stability = same_sign_periods / eligible_periods if eligible_periods else math.nan
    if len(curve) >= 3:
        earlier_significant = directed_t[:-2] > 3.0
        if np.any(earlier_significant):
            earlier_median = float(np.median(directed_means[:-2][earlier_significant]))
            recent_dead = bool(
                np.all(np.abs(directed_t[-2:]) < 1.0)
                and np.all(np.abs(directed_means[-2:]) <= 0.25 * abs(earlier_median))
            )
            if recent_dead:
                death_date = curve[-2].window_end
                return DecayResult(
                    curve,
                    stability,
                    DecayClass.DEAD,
                    death_date,
                    (
                        float(
                            np.median(np.abs(directed_means[-2:])) / abs(earlier_median)
                        )
                        if earlier_median
                        else 0.0
                    ),
                    "two most recent windows are economically and statistically dead",
                    fixed,
                    same_sign_periods,
                    eligible_periods,
                )
    prior = directed_means[:-1]
    finite_prior = prior[np.isfinite(prior)]
    prior_median = float(np.median(finite_prior)) if finite_prior.size else math.nan
    ratio = (
        float(directed_means[-1] / prior_median)
        if math.isfinite(prior_median) and prior_median > 0.0
        else None
    )
    if (
        regime_interaction_adjusted_p is not None
        and active_regime_t is not None
        and regime_interaction_adjusted_p < 0.05
        and active_regime_t * expected_sign > 2.0
    ):
        return DecayResult(
            curve,
            stability,
            DecayClass.REGIME_DEPENDENT,
            None,
            ratio,
            "adjusted interaction is significant and active regime is positive",
            fixed,
            same_sign_periods,
            eligible_periods,
        )
    if (
        stability >= stability_min
        and directed_means[-1] > 0.0
        and ratio is not None
        and ratio >= 0.5
    ):
        classification = DecayClass.STABLE
        reason = "same-sign stability and recent magnitude both pass"
    elif directed_means[-1] > 0.0 and ratio is not None and ratio < 0.5:
        classification = DecayClass.DECAYING
        reason = "recent effect remains positive but is below half its prior median"
    else:
        classification = DecayClass.INSUFFICIENT
        reason = "trajectory does not meet a live-eligible decay class"
    return DecayResult(
        curve,
        stability,
        classification,
        None,
        ratio,
        reason,
        fixed,
        same_sign_periods,
        eligible_periods,
    )


def analyze_decay(
    returns: FloatArray | list[float],
    dates: pd.DatetimeIndex | DateArray | list[object],
    *,
    window_years: int = 5,
    step_months: int = 12,
    holding_period: int = 1,
    minimum_observations: int = 100,
    expected_sign: int = 1,
    stability_min: float = 0.75,
    regime_interaction_adjusted_p: float | None = None,
    active_regime_t: float | None = None,
) -> DecayResult:
    """Compute and classify a decay curve with shared keyword policy."""

    points = rolling_decay(
        returns,
        dates,
        window_years=window_years,
        step_months=step_months,
        holding_period=holding_period,
        minimum_observations=minimum_observations,
    )
    fixed_points = fixed_decay(
        returns,
        dates,
        window_years=window_years,
        holding_period=holding_period,
        minimum_observations=minimum_observations,
    )
    return classify_decay(
        points,
        fixed_points=fixed_points,
        expected_sign=expected_sign,
        stability_min=stability_min,
        regime_interaction_adjusted_p=regime_interaction_adjusted_p,
        active_regime_t=active_regime_t,
    )


def _aligned_inputs(
    returns: FloatArray | list[float],
    dates: pd.DatetimeIndex | DateArray | list[object],
) -> tuple[FloatArray, pd.DatetimeIndex]:
    """Validate and return aligned inputs shared by decay estimators."""

    values: FloatArray = np.asarray(returns, dtype=np.float64)
    index = pd.DatetimeIndex(dates)
    if values.ndim != 1 or len(values) != len(index):
        raise ValueError("returns and dates must be aligned one-dimensional data")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("dates must be sorted and unique")
    return values, index
