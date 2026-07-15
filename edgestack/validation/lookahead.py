"""Hard causal checks and a synthetic shift-sensitivity diagnostic."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from edgestack.backtest.engine import lag_positions
from edgestack.models import ensure_fill_after_signal
from edgestack.stats._types import FloatArray
from edgestack.stats.tests import hac_mean_test


def _equal_prefix(left: object, right: object, length: int) -> bool:
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        return bool(left.iloc[:length].equals(right.iloc[:length]))
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        return bool(left.iloc[:length].equals(right.iloc[:length]))
    first = np.asarray(left)[:length]
    second = np.asarray(right)[:length]
    return bool(np.allclose(first, second, equal_nan=True))


def assert_prefix_invariant(
    feature: Callable[[object], object],
    data: object,
    *,
    prefix_length: int,
    future_mutator: Callable[[object, int], object],
) -> None:
    """Assert mutating every future input cannot alter past feature values."""

    if prefix_length < 1:
        raise ValueError("prefix_length must be positive")
    baseline = feature(data)
    mutated_data = future_mutator(data, prefix_length)
    mutated = feature(mutated_data)
    if not _equal_prefix(baseline, mutated, prefix_length):
        raise AssertionError("feature changed before the future-mutation boundary")


def assert_truncation_invariant(
    feature: Callable[[object], object], data: object, *, prefix_length: int
) -> None:
    """Assert full-history and truncated-history feature prefixes match."""

    baseline = feature(data)
    truncated_data: object
    if isinstance(data, (pd.Series, pd.DataFrame)):
        truncated_data = data.iloc[:prefix_length].copy()
    else:
        truncated_data = np.asarray(data)[:prefix_length].copy()
    truncated = feature(truncated_data)
    if not _equal_prefix(baseline, truncated, prefix_length):
        raise AssertionError("feature is not prefix invariant")


def assert_available_at(
    available_at: Sequence[datetime], decision_times: Sequence[datetime]
) -> None:
    """Reject joins containing data unavailable at the corresponding decision."""

    if len(available_at) != len(decision_times):
        raise ValueError("timestamp sequences must be aligned")
    violations = [
        index
        for index, (known, decision) in enumerate(
            zip(available_at, decision_times, strict=True)
        )
        if known > decision
    ]
    if violations:
        raise AssertionError(f"future information at rows {violations[:10]}")


def assert_fills_after_signals(
    signal_times: Sequence[datetime], fill_times: Sequence[datetime]
) -> None:
    """Apply the strict global execution-time invariant to every fill."""

    if len(signal_times) != len(fill_times):
        raise ValueError("signal and fill times must be aligned")
    for signal_time, fill_time in zip(signal_times, fill_times, strict=True):
        ensure_fill_after_signal(signal_time, fill_time)


@dataclass(frozen=True, slots=True)
class ShiftCollapseResult:
    """Synthetic one-bar-alpha shift diagnostic."""

    baseline_t: float
    extra_lag_t: float
    collapsed: bool


def shift_and_collapse(
    signal: FloatArray,
    returns: FloatArray,
    *,
    baseline_lag: int = 1,
    significant_t: float = 3.0,
    noise_t: float = 1.96,
) -> ShiftCollapseResult:
    """Verify a deliberately one-bar alpha disappears with one extra signal lag."""

    values = np.asarray(signal, dtype=float)
    realized = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.shape != realized.shape:
        raise ValueError("signal and returns must be aligned one-dimensional arrays")
    baseline = lag_positions(values, execution_lag=baseline_lag) * realized
    shifted = lag_positions(values, execution_lag=baseline_lag + 1) * realized
    baseline_t = hac_mean_test(baseline).t_stat
    shifted_t = hac_mean_test(shifted).t_stat
    collapsed = bool(baseline_t > significant_t and abs(shifted_t) < noise_t)
    return ShiftCollapseResult(baseline_t, shifted_t, collapsed)


lookahead_detector = assert_prefix_invariant
