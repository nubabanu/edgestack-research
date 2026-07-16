"""Exchange-session calendar features.

The functions work on an explicit sequence of trading sessions. No civil-day
arithmetic is used for turn-of-month or event proximity, preventing holiday and
weekend look-ahead mistakes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def _sessions_index(sessions: Sequence[object] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    index = (
        sessions.copy()
        if isinstance(sessions, pd.DatetimeIndex)
        else pd.DatetimeIndex(list(sessions))
    )
    if index.hasnans:
        raise ValueError("sessions cannot contain NaT")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("sessions must be sorted and unique")
    return index


def day_of_week(sessions: Sequence[object] | pd.DatetimeIndex) -> pd.Series:
    """Return Monday=0 through Friday=4 for each exchange session."""

    index = _sessions_index(sessions)
    return pd.Series(index.dayofweek, index=index, name="weekday", dtype="int8")


def month(sessions: Sequence[object] | pd.DatetimeIndex) -> pd.Series:
    """Return calendar month number 1 through 12."""

    index = _sessions_index(sessions)
    return pd.Series(index.month, index=index, name="month", dtype="int8")


def session_of_month(sessions: Sequence[object] | pd.DatetimeIndex) -> pd.Series:
    """Return zero-based trading-session position within each month."""

    index = _sessions_index(sessions)
    frame = pd.DataFrame(index=index)
    values = (
        frame.groupby([index.year, index.month]).cumcount().to_numpy(dtype=np.int16)
    )
    return pd.Series(values, index=index, name="session_of_month")


def sessions_to_month_end(sessions: Sequence[object] | pd.DatetimeIndex) -> pd.Series:
    """Return zero-based number of exchange sessions remaining in the month."""

    index = _sessions_index(sessions)
    frame = pd.DataFrame(index=index)
    position = frame.groupby([index.year, index.month]).cumcount()
    # pandas represents the empty-frame transform inconsistently by version;
    # group sizes mapped by period are simpler and stable.
    periods = index.to_period("M")
    counts = pd.Series(periods, index=index).groupby(periods).transform("size")
    values = counts.to_numpy(dtype=np.int16) - position.to_numpy(dtype=np.int16) - 1
    return pd.Series(values, index=index, name="sessions_to_month_end")


def sessions_to_quarter_end(
    sessions: Sequence[object] | pd.DatetimeIndex,
) -> pd.Series:
    """Return zero-based number of exchange sessions remaining in the quarter."""

    index = _sessions_index(sessions)
    frame = pd.DataFrame(index=index)
    position = frame.groupby([index.year, index.quarter]).cumcount()
    periods = index.to_period("Q")
    counts = pd.Series(periods, index=index).groupby(periods).transform("size")
    values = counts.to_numpy(dtype=np.int16) - position.to_numpy(dtype=np.int16) - 1
    return pd.Series(values, index=index, name="sessions_to_quarter_end")


def month_end_window(
    sessions: Sequence[object] | pd.DatetimeIndex, *, window: int = 5
) -> pd.Series:
    """Flag the final ``window`` exchange sessions of each month.

    Distinct from ``turn_of_month`` (last one + first three): this is the
    institutional rebalancing-flow window BEFORE the month boundary.
    """

    if window < 1:
        raise ValueError("window must be positive")
    index = _sessions_index(sessions)
    values = sessions_to_month_end(index).to_numpy() < window
    return pd.Series(values, index=index, name="month_end_window")


def quarter_end_window(
    sessions: Sequence[object] | pd.DatetimeIndex, *, window: int = 5
) -> pd.Series:
    """Flag the final ``window`` exchange sessions of each calendar quarter."""

    if window < 1:
        raise ValueError("window must be positive")
    index = _sessions_index(sessions)
    values = sessions_to_quarter_end(index).to_numpy() < window
    return pd.Series(values, index=index, name="quarter_end_window")


def turn_of_month(
    sessions: Sequence[object] | pd.DatetimeIndex,
    *,
    last_sessions: int = 1,
    first_sessions: int = 3,
) -> pd.Series:
    """Flag the McConnell-Xu turn-of-month exchange-session window.

    With defaults this is the last trading day of a month and the first three
    trading days of the following month.
    """

    if last_sessions < 0 or first_sessions < 0:
        raise ValueError("window lengths must be non-negative")
    index = _sessions_index(sessions)
    start = session_of_month(index).to_numpy()
    end = sessions_to_month_end(index).to_numpy()
    flag = (start < first_sessions) | (end < last_sessions)
    return pd.Series(flag, index=index, name="turn_of_month")


def holiday_proximity(
    sessions: Sequence[object] | pd.DatetimeIndex,
    holidays: Iterable[object],
) -> pd.DataFrame:
    """Flag sessions immediately before and after an exchange holiday.

    Holiday dates need not be sessions. A holiday is associated with the last
    session strictly before it and the first session strictly after it.
    """

    index = _sessions_index(sessions)
    holiday_index = pd.DatetimeIndex(list(holidays)).normalize().unique().sort_values()
    pre = np.zeros(len(index), dtype=bool)
    post = np.zeros(len(index), dtype=bool)
    normalized = index.normalize()
    for holiday in holiday_index:
        insertion = int(normalized.searchsorted(holiday, side="left"))
        if insertion > 0 and normalized[insertion - 1] < holiday:
            pre[insertion - 1] = True
        after = int(normalized.searchsorted(holiday, side="right"))
        if after < len(index):
            post[after] = True
    return pd.DataFrame({"pre_holiday": pre, "post_holiday": post}, index=index)


def event_proximity(
    sessions: Sequence[object] | pd.DatetimeIndex,
    events: Iterable[object],
) -> pd.DataFrame:
    """Flag the session before, session of, and week of scheduled events.

    Event dates that are not sessions receive no day-of flag. ``event_week`` is
    the ISO calendar week containing the scheduled date, as preregistered.
    """

    index = _sessions_index(sessions)
    normalized = index.normalize()
    event_index = pd.DatetimeIndex(list(events)).normalize().unique().sort_values()
    day_of = normalized.isin(event_index)
    day_before = np.zeros(len(index), dtype=bool)
    for event in event_index:
        position = int(normalized.searchsorted(event, side="left"))
        if position > 0:
            day_before[position - 1] = True
    session_iso = normalized.isocalendar()
    event_iso = event_index.isocalendar()
    event_keys = set(
        zip(event_iso.year.to_numpy(), event_iso.week.to_numpy(), strict=True)
    )
    week = np.fromiter(
        (
            (year, week_) in event_keys
            for year, week_ in zip(session_iso.year, session_iso.week, strict=True)
        ),
        dtype=bool,
        count=len(index),
    )
    return pd.DataFrame(
        {"event_day_before": day_before, "event_day_of": day_of, "event_week": week},
        index=index,
    )


def opex_week(sessions: Sequence[object] | pd.DatetimeIndex) -> pd.Series:
    """Flag the week containing each month's third Friday.

    This definition intentionally remains the scheduled third-Friday week even
    when the Friday is an exchange holiday.
    """

    index = _sessions_index(sessions)
    normalized = index.normalize()
    periods = normalized.to_period("M").unique()
    keys: set[tuple[int, int]] = set()
    for period in periods:
        first = period.start_time
        first_friday = first + pd.offsets.Week(weekday=4)
        if first.weekday() == 4:
            first_friday = first
        third_friday = first_friday + pd.Timedelta(days=14)
        iso = third_friday.isocalendar()
        keys.add((int(iso.year), int(iso.week)))
    session_iso = normalized.isocalendar()
    values = np.fromiter(
        (
            (year, week_) in keys
            for year, week_ in zip(session_iso.year, session_iso.week, strict=True)
        ),
        dtype=bool,
        count=len(index),
    )
    return pd.Series(values, index=index, name="opex_week")


def calendar_features(
    sessions: Sequence[object] | pd.DatetimeIndex,
    *,
    holidays: Iterable[object] = (),
    fomc_dates: Iterable[object] = (),
) -> pd.DataFrame:
    """Build the complete deterministic daily calendar feature table."""

    index = _sessions_index(sessions)
    features = pd.DataFrame(index=index)
    features["weekday"] = day_of_week(index)
    features["month"] = month(index)
    features["session_of_month"] = session_of_month(index)
    features["sessions_to_month_end"] = sessions_to_month_end(index)
    features["turn_of_month"] = turn_of_month(index)
    features["sessions_to_quarter_end"] = sessions_to_quarter_end(index)
    features["month_end_window"] = month_end_window(index)
    features["quarter_end_window"] = quarter_end_window(index)
    features = features.join(holiday_proximity(index, holidays))
    features = features.join(event_proximity(index, fomc_dates).add_prefix("fomc_"))
    features["opex_week"] = opex_week(index)
    return features


# Friendly aliases used in configuration-driven feature registries.
tom_window = turn_of_month
fomc_proximity = event_proximity
