"""Expanding-window, calendar-aware walk-forward validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from edgestack.stats._types import DateArray, FloatArray, IntArray
from edgestack.stats.tests import HACTestResult, hac_mean_test


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One expanding train and non-overlapping forward test window."""

    fold: int
    train_indices: IntArray
    test_indices: IntArray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def expanding_window_splits(
    dates: pd.DatetimeIndex | DateArray | list[object],
    *,
    min_train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
) -> tuple[WalkForwardFold, ...]:
    """Create calendar-year expanding splits on actual observation dates."""

    index = pd.DatetimeIndex(dates)
    if index.hasnans or not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("dates must be sorted, unique, and non-missing")
    if min_train_years < 1 or test_years < 1 or step_years < 1:
        raise ValueError("year lengths must be positive")
    if len(index) == 0:
        return ()
    candidate_start = index[0] + pd.DateOffset(years=min_train_years)
    folds: list[WalkForwardFold] = []
    fold_number = 0
    while candidate_start <= index[-1]:
        test_stop = candidate_start + pd.DateOffset(years=test_years)
        train_indices = np.flatnonzero(index < candidate_start)
        test_indices = np.flatnonzero((index >= candidate_start) & (index < test_stop))
        if train_indices.size and test_indices.size:
            folds.append(
                WalkForwardFold(
                    fold=fold_number,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_start=index[train_indices[0]],
                    train_end=index[train_indices[-1]],
                    test_start=index[test_indices[0]],
                    test_end=index[test_indices[-1]],
                )
            )
            fold_number += 1
        candidate_start += pd.DateOffset(years=step_years)
    return tuple(folds)


@dataclass(frozen=True, slots=True)
class WalkForwardEvidence:
    """Inference for one train/test split."""

    fold: WalkForwardFold
    train: HACTestResult
    test: HACTestResult


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Stitched out-of-sample evidence over all expanding folds."""

    folds: tuple[WalkForwardEvidence, ...]
    stitched_oos_returns: FloatArray
    stitched_oos_test: HACTestResult
    positive_fraction: float
    majority_positive: bool
    significant_oos: bool


def expanding_walk_forward(
    returns: FloatArray | list[float],
    dates: pd.DatetimeIndex | DateArray | list[object] | None = None,
    *,
    min_train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
    holding_period: int = 1,
    oos_t_threshold: float = 2.0,
    required_positive_fraction: float = 0.5,
) -> WalkForwardResult:
    """Evaluate a fixed edge on expanding train and forward test windows.

    This function never selects parameters within a fold. Parameter selection,
    if needed, must be supplied as a separately preregistered outer workflow.
    """

    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if dates is None:
        # A synthetic business-day index makes the observation-count API useful
        # in unit tests without weakening calendar behavior in real campaigns.
        index = pd.bdate_range("2000-01-03", periods=len(values))
    else:
        index = pd.DatetimeIndex(dates)
    if len(index) != len(values):
        raise ValueError("dates and returns must be aligned")
    splits = expanding_window_splits(
        index,
        min_train_years=min_train_years,
        test_years=test_years,
        step_years=step_years,
    )
    evidence: list[WalkForwardEvidence] = []
    stitched = np.full(len(values), np.nan, dtype=float)
    test_means: list[float] = []
    for split in splits:
        train_test = hac_mean_test(
            values[split.train_indices], holding_period=holding_period
        )
        forward_test = hac_mean_test(
            values[split.test_indices], holding_period=holding_period
        )
        evidence.append(WalkForwardEvidence(split, train_test, forward_test))
        # Step < test horizon can overlap. First-seen ownership keeps stitched
        # OOS observations unique and prevents significance inflation.
        unassigned = ~np.isfinite(stitched[split.test_indices])
        stitched[split.test_indices[unassigned]] = values[
            split.test_indices[unassigned]
        ]
        test_means.append(forward_test.mean)
    finite_stitched = stitched[np.isfinite(stitched)]
    stitched_test = hac_mean_test(finite_stitched, holding_period=holding_period)
    positive_fraction = (
        float(np.mean(np.asarray(test_means) > 0.0)) if test_means else math.nan
    )
    return WalkForwardResult(
        folds=tuple(evidence),
        stitched_oos_returns=stitched,
        stitched_oos_test=stitched_test,
        positive_fraction=positive_fraction,
        majority_positive=bool(
            test_means and positive_fraction > required_positive_fraction
        ),
        significant_oos=bool(stitched_test.t_stat > oos_t_threshold),
    )


walk_forward = expanding_walk_forward
