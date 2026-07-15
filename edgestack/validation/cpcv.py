"""Combinatorial purged cross-validation and PBO.

PBO follows Bailey et al. (2016): select the best in-sample alternative for each
split, rank its out-of-sample performance among all alternatives, transform the
relative rank to a logit, and report the fraction at or below zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from edgestack.stats._types import FloatArray, IntArray
from edgestack.stats.tests import annualized_sharpe


@dataclass(frozen=True, slots=True)
class CPCVSplit:
    """One combinatorial test-group choice with purged training indices."""

    test_groups: tuple[int, ...]
    train_indices: IntArray
    test_indices: IntArray


def combinatorial_purged_splits(
    n_observations: int,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge: int = 21,
    embargo: int = 21,
) -> tuple[CPCVSplit, ...]:
    """Enumerate all purged/embargoed group combinations.

    ``purge`` removes training labels immediately before every contiguous test
    group; ``embargo`` removes observations immediately after it.
    """

    if n_observations < n_groups:
        raise ValueError("n_observations must be at least n_groups")
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("n_test_groups must be in [1, n_groups)")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo cannot be negative")
    groups = tuple(
        np.asarray(group, dtype=np.int64)
        for group in np.array_split(np.arange(n_observations), n_groups)
    )
    output: list[CPCVSplit] = []
    all_indices = np.arange(n_observations, dtype=np.int64)
    for selected in combinations(range(n_groups), n_test_groups):
        test_indices = np.concatenate([groups[group] for group in selected])
        test_mask = np.zeros(n_observations, dtype=bool)
        test_mask[test_indices] = True
        forbidden = test_mask.copy()
        # Merge adjacent selected groups so their shared boundary does not
        # receive meaningless purge/embargo treatment.
        selected_set = set(selected)
        starts = [group for group in selected if group - 1 not in selected_set]
        ends = [group for group in selected if group + 1 not in selected_set]
        for start_group, end_group in zip(starts, ends, strict=True):
            start = int(groups[start_group][0])
            stop = int(groups[end_group][-1]) + 1
            forbidden[max(0, start - purge) : start] = True
            forbidden[stop : min(n_observations, stop + embargo)] = True
        train_indices = all_indices[~forbidden]
        output.append(CPCVSplit(tuple(selected), train_indices, test_indices))
    return tuple(output)


@dataclass(frozen=True, slots=True)
class PBOResult:
    """Probability-of-backtest-overfitting evidence for a candidate family."""

    pbo: float | None
    logits: FloatArray
    selected_strategies: IntArray
    oos_rank_percentiles: FloatArray
    n_splits: int
    defined: bool
    reason: str = ""


def probability_backtest_overfitting(
    train_performance: FloatArray,
    test_performance: FloatArray,
) -> PBOResult:
    """Compute PBO from aligned split-by-strategy performance matrices."""

    train = np.asarray(train_performance, dtype=float)
    test = np.asarray(test_performance, dtype=float)
    if train.ndim != 2 or train.shape != test.shape:
        raise ValueError(
            "train and test performance must have equal (split, strategy) shape"
        )
    n_splits, n_strategies = train.shape
    if n_strategies < 2 or n_splits == 0:
        return PBOResult(
            None,
            np.array([]),
            np.array([], dtype=int),
            np.array([]),
            n_splits,
            False,
            "PBO needs at least two alternatives and one split",
        )
    if np.any(~np.isfinite(train)) or np.any(~np.isfinite(test)):
        raise ValueError("performance matrices must be finite")
    selected = np.argmax(train, axis=1)
    percentiles = np.empty(n_splits, dtype=float)
    for split, strategy in enumerate(selected):
        value = test[split, strategy]
        # Mid-rank handles ties deterministically. Rank 1 is worst, N is best.
        lower = np.count_nonzero(test[split] < value)
        equal = np.count_nonzero(test[split] == value)
        rank = lower + (equal + 1.0) / 2.0
        percentiles[split] = rank / n_strategies
    epsilon = np.finfo(float).eps
    clipped = np.clip(percentiles, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped))
    pbo = float(np.mean(logits <= 0.0))
    return PBOResult(pbo, logits, selected, percentiles, n_splits, True)


def cpcv_pbo(
    returns: FloatArray,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge: int = 21,
    embargo: int = 21,
    periods_per_year: float = 252.0,
) -> PBOResult:
    """Compute CPCV Sharpe matrices and PBO from date-by-strategy returns."""

    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("returns must be (dates, strategies)")
    splits = combinatorial_purged_splits(
        matrix.shape[0],
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge=purge,
        embargo=embargo,
    )
    train = np.empty((len(splits), matrix.shape[1]), dtype=float)
    test = np.empty_like(train)
    for row, split in enumerate(splits):
        for strategy in range(matrix.shape[1]):
            train[row, strategy] = annualized_sharpe(
                matrix[split.train_indices, strategy], periods_per_year=periods_per_year
            )
            test[row, strategy] = annualized_sharpe(
                matrix[split.test_indices, strategy], periods_per_year=periods_per_year
            )
    # Constant streams yield NaN/inf Sharpes. PBO should be explicitly
    # unavailable rather than silently assigning arbitrary ranks.
    if np.any(~np.isfinite(train)) or np.any(~np.isfinite(test)):
        return PBOResult(
            None,
            np.array([]),
            np.array([], dtype=int),
            np.array([]),
            len(splits),
            False,
            "non-finite fold performance",
        )
    return probability_backtest_overfitting(train, test)


combinatorial_purged_kfold = combinatorial_purged_splits
