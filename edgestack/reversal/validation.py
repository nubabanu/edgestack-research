"""Purged panel splits and cross-sectional ranking diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from edgestack.stats._types import FloatArray, IntArray
from edgestack.validation.walkforward import expanding_window_splits


@dataclass(frozen=True, slots=True)
class PurgedPanelFold:
    """Row indices for one label-overlap-safe chronological panel fold."""

    fold: int
    train_indices: IntArray
    test_indices: IntArray
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purged_rows: int


@dataclass(frozen=True, slots=True)
class RankingDiagnostics:
    """Out-of-sample cross-sectional ranking evidence."""

    mean_ic: float
    median_ic: float
    positive_ic_fraction: float
    top_bottom_spread: float
    dates: int
    observations: int


def purged_panel_splits(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    label_end_column: str = "label_end",
    min_train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
    purge_sessions: int = 5,
) -> tuple[PurgedPanelFold, ...]:
    """Create expanding panel folds and remove overlapping training labels."""

    if purge_sessions < 0:
        raise ValueError("purge_sessions cannot be negative")
    dates = pd.to_datetime(frame[date_column])
    label_end = pd.to_datetime(frame[label_end_column])
    if dates.isna().any() or label_end.isna().any():
        raise ValueError("date and label_end columns must be complete")
    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
    date_folds = expanding_window_splits(
        unique_dates,
        min_train_years=min_train_years,
        test_years=test_years,
        step_years=step_years,
    )
    output: list[PurgedPanelFold] = []
    for date_fold in date_folds:
        test_dates = unique_dates[date_fold.test_indices]
        train_dates = unique_dates[date_fold.train_indices]
        test_start = pd.Timestamp(test_dates[0])
        if purge_sessions:
            allowed_train_dates = train_dates[:-purge_sessions]
        else:
            allowed_train_dates = train_dates
        initial_train = dates.isin(train_dates)
        safe_train = dates.isin(allowed_train_dates) & (label_end < test_start)
        test_mask = dates.isin(test_dates)
        train_indices = np.flatnonzero(safe_train.to_numpy()).astype(np.int64)
        test_indices = np.flatnonzero(test_mask.to_numpy()).astype(np.int64)
        if train_indices.size == 0 or test_indices.size == 0:
            continue
        output.append(
            PurgedPanelFold(
                date_fold.fold,
                train_indices,
                test_indices,
                pd.Timestamp(dates.iloc[train_indices].max()),
                test_start,
                pd.Timestamp(test_dates[-1]),
                int(initial_train.sum() - safe_train.sum()),
            )
        )
    return tuple(output)


def ranking_diagnostics(
    realized: FloatArray,
    predicted: FloatArray,
    dates: pd.Series | pd.DatetimeIndex,
    *,
    top_k: int = 5,
) -> RankingDiagnostics:
    """Measure date-level Spearman IC and predicted top-minus-bottom spread."""

    actual = np.asarray(realized, dtype=float)
    forecast = np.asarray(predicted, dtype=float)
    date_values = pd.DatetimeIndex(dates)
    if (
        actual.ndim != 1
        or actual.shape != forecast.shape
        or len(date_values) != len(actual)
    ):
        raise ValueError("realized, predicted, and dates must be aligned vectors")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    frame = pd.DataFrame(
        {"date": date_values, "realized": actual, "predicted": forecast}
    ).dropna()
    information_coefficients: list[float] = []
    spreads: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        if len(group) < max(3, 2 * top_k):
            continue
        coefficient = spearmanr(group["predicted"], group["realized"]).statistic
        if np.isfinite(coefficient):
            information_coefficients.append(float(coefficient))
        ordered = group.sort_values("predicted", kind="stable")
        bottom = float(ordered["realized"].head(top_k).mean())
        top = float(ordered["realized"].tail(top_k).mean())
        spreads.append(top - bottom)
    ic = np.asarray(information_coefficients, dtype=float)
    spread = np.asarray(spreads, dtype=float)
    return RankingDiagnostics(
        float(ic.mean()) if ic.size else float("nan"),
        float(np.median(ic)) if ic.size else float("nan"),
        float(np.mean(ic > 0.0)) if ic.size else float("nan"),
        float(spread.mean()) if spread.size else float("nan"),
        int(max(ic.size, spread.size)),
        len(frame),
    )
