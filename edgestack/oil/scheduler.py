"""New-York-time automation for the fail-closed oil paper snapshots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from importlib import import_module
from typing import Any
from zoneinfo import ZoneInfo

from edgestack.live.scheduler import is_trading_day

NEW_YORK = ZoneInfo("America/New_York")


def build_oil_scheduler(
    run_snapshot: Callable[[str], Any],
    *,
    schedule_config: Mapping[str, Any] | None = None,
) -> Any:
    """Build, but do not start, the four fixed oil-paper jobs."""

    try:
        background = import_module("apscheduler.schedulers.background")
        cron = import_module("apscheduler.triggers.cron")
    except ImportError as error:
        raise RuntimeError("install EdgeStack with the 'live' extra") from error
    settings = dict(schedule_config or {})
    declared = (
        ("oil_pre_open_eligibility", "pre_open_et", "08:30", "mon-fri", "PRE_OPEN"),
        ("oil_intraday_refresh", "intraday_refresh_et", "09:50", "mon-fri", "INTRADAY_REFRESH"),
        ("oil_post_eia_refresh", "post_eia_refresh_et", "11:15", "wed", "POST_EIA"),
        ("oil_swing_snapshot", "swing_snapshot_et", "16:30", "mon-fri", "SWING_CLOSE"),
    )
    scheduler = background.BackgroundScheduler(timezone=NEW_YORK)
    for job_id, key, default, weekdays, reason in declared:
        raw = str(settings.get(key, default))
        try:
            parsed = datetime.strptime(raw, "%H:%M")
        except ValueError as error:
            raise ValueError(f"{key} must be HH:MM") from error

        def execute(label: str = reason) -> Any:
            today = datetime.now(NEW_YORK).date()
            return run_snapshot(label) if is_trading_day(today) else None

        scheduler.add_job(
            execute,
            cron.CronTrigger(
                day_of_week=weekdays,
                hour=parsed.hour,
                minute=parsed.minute,
                timezone=NEW_YORK,
            ),
            id=job_id,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
    return scheduler


__all__ = ["build_oil_scheduler"]
