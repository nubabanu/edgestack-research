"""NYSE-aware scheduling for the long-lived paper assistant."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from importlib import import_module
from typing import Any, cast
from zoneinfo import ZoneInfo

from edgestack.data.calendars import NYSECalendar

_NYSE = NYSECalendar()
_NEW_YORK = ZoneInfo("America/New_York")


def is_trading_day(day: date) -> bool:
    """Return whether NYSE has a regular or half-day session."""

    return _NYSE.is_session(day)


def is_market_open(at: datetime | None = None) -> bool:
    """Whether ``at`` falls inside the actual NYSE session, including half-days."""

    moment = at or datetime.now(UTC)
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("market-open checks require a timezone-aware timestamp")
    local_day = moment.astimezone(_NEW_YORK).date()
    schedule = _NYSE.schedule(local_day, local_day)
    if schedule.empty:
        return False
    opened = schedule.iloc[0]["market_open"].to_pydatetime()
    closed = schedule.iloc[0]["market_close"].to_pydatetime()
    return cast(bool, opened <= moment.astimezone(UTC) <= closed)


def build_scheduler(
    scan: Callable[[], Any],
    monitor: Callable[[], Any],
    scorecard: Callable[[], Any],
    *,
    scan_time: str = "08:30",
    poll_minutes: int = 15,
) -> Any:
    """Build an APScheduler instance without starting it."""

    try:
        aps_background = import_module("apscheduler.schedulers.background")
        aps_cron = import_module("apscheduler.triggers.cron")
        aps_interval = import_module("apscheduler.triggers.interval")
    except ImportError as exc:
        raise RuntimeError("install EdgeStack with the 'live' extra") from exc
    if poll_minutes <= 0:
        raise ValueError("poll_minutes must be positive")
    try:
        parsed_scan_time = datetime.strptime(scan_time, "%H:%M")
    except ValueError as exc:
        raise ValueError("scan_time must be a valid HH:MM value") from exc
    timezone = _NEW_YORK
    hour, minute = parsed_scan_time.hour, parsed_scan_time.minute
    scheduler = aps_background.BackgroundScheduler(timezone=timezone)
    scheduler.add_job(
        lambda: scan() if is_trading_day(datetime.now(timezone).date()) else None,
        aps_cron.CronTrigger(
            day_of_week="mon-fri", hour=hour, minute=minute, timezone=timezone
        ),
        id="pre_market_scan",
        replace_existing=True,
    )
    scheduler.add_job(
        lambda: monitor() if is_market_open(datetime.now(timezone)) else None,
        aps_interval.IntervalTrigger(minutes=poll_minutes, timezone=timezone),
        id="market_monitor",
        replace_existing=True,
    )
    scheduler.add_job(
        lambda: scorecard() if is_trading_day(datetime.now(timezone).date()) else None,
        aps_cron.CronTrigger(
            day_of_week="mon-fri", hour=16, minute=30, timezone=timezone
        ),
        id="post_close_scorecard",
        replace_existing=True,
    )
    return scheduler
