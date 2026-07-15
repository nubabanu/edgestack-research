"""Pinned NYSE schedules, option expiries, holidays, and official FOMC dates."""

from __future__ import annotations

import calendar as month_calendar
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from html.parser import HTMLParser
from typing import Final, cast
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import pandas_market_calendars as market_calendars  # type: ignore[import-untyped]

from edgestack.data.sources import RawPayload, RawPayloadSink

NEW_YORK: Final = ZoneInfo("America/New_York")
FED_CALENDAR_URL: Final = (
    "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
)
FED_HISTORICAL_URL: Final = (
    "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
)


class NYSECalendar:
    """Thin deterministic wrapper around pinned ``pandas_market_calendars`` NYSE."""

    def __init__(self) -> None:
        self._calendar = market_calendars.get_calendar("NYSE")

    def schedule(self, start: date | str, end: date | str) -> pd.DataFrame:
        """Return UTC market-open/close timestamps indexed by session date."""

        if pd.Timestamp(end) < pd.Timestamp(start):
            raise ValueError("end must be on or after start")
        result = cast(
            pd.DataFrame,
            self._calendar.schedule(start_date=start, end_date=end),
        )
        result.index = pd.DatetimeIndex(result.index).tz_localize(None).normalize()
        return result.loc[:, ["market_open", "market_close"]].copy()

    def sessions(self, start: date | str, end: date | str) -> pd.DatetimeIndex:
        """Return valid NYSE session labels as timezone-naive normalized dates."""

        return pd.DatetimeIndex(self.schedule(start, end).index)

    def is_session(self, value: date | str | pd.Timestamp) -> bool:
        """Whether ``value`` is an NYSE trading session."""

        session = pd.Timestamp(value).normalize()
        return session in self.sessions(session, session)

    def next_session(
        self, value: date | str | pd.Timestamp, *, inclusive: bool = False
    ) -> pd.Timestamp:
        """Return the next valid session, optionally including ``value``."""

        start = pd.Timestamp(value).normalize()
        if not inclusive:
            start += pd.Timedelta(days=1)
        candidates = self.sessions(start, start + pd.Timedelta(days=14))
        if candidates.empty:
            raise RuntimeError(f"no NYSE session found after {value}")
        return candidates[0]

    def previous_session(
        self, value: date | str | pd.Timestamp, *, inclusive: bool = False
    ) -> pd.Timestamp:
        """Return the previous valid session, optionally including ``value``."""

        end = pd.Timestamp(value).normalize()
        if not inclusive:
            end -= pd.Timedelta(days=1)
        candidates = self.sessions(end - pd.Timedelta(days=14), end)
        if candidates.empty:
            raise RuntimeError(f"no NYSE session found before {value}")
        return candidates[-1]

    def close_time(self, value: date | str | pd.Timestamp) -> datetime:
        """Return the actual UTC close, including scheduled early closes."""

        session = pd.Timestamp(value).normalize()
        schedule = self.schedule(session, session)
        if schedule.empty:
            raise ValueError(f"not an NYSE session: {session.date()}")
        return pd.Timestamp(schedule.iloc[0]["market_close"]).to_pydatetime()

    def assert_reference_match(self, start: date, end: date) -> None:
        """Raise unless wrapper output exactly equals the pinned reference object."""

        wrapped = self.schedule(start, end)
        reference = market_calendars.get_calendar("NYSE").schedule(
            start_date=start, end_date=end
        )
        reference.index = (
            pd.DatetimeIndex(reference.index).tz_localize(None).normalize()
        )
        reference = reference.loc[:, ["market_open", "market_close"]]
        pd.testing.assert_frame_equal(wrapped, reference, check_freq=False)


def holiday_adjacency(
    start: date, end: date, *, nyse: NYSECalendar | None = None
) -> pd.DataFrame:
    """Identify sessions immediately before/after weekday exchange closures.

    Normal Saturdays and Sundays are not classified as holidays.  Multi-day
    closures map to one pre-holiday and one post-holiday session.
    """

    exchange = nyse or NYSECalendar()
    sessions = exchange.sessions(start, end)
    output = pd.DataFrame(
        {"session": sessions, "pre_holiday": False, "post_holiday": False}
    ).set_index("session")
    weekdays = pd.date_range(start, end, freq="B").normalize()
    closures = weekdays.difference(sessions)
    for closure in closures:
        before = sessions[sessions < closure]
        after = sessions[sessions > closure]
        if len(before):
            output.loc[before[-1], "pre_holiday"] = True
        if len(after):
            output.loc[after[0], "post_holiday"] = True
    return output.reset_index()


def monthly_opex_sessions(
    start: date, end: date, *, nyse: NYSECalendar | None = None
) -> pd.DatetimeIndex:
    """Return monthly US equity-option expiry sessions.

    The nominal expiry is the third Friday.  When it is an exchange holiday the
    preceding NYSE session (normally Thursday) is used.
    """

    exchange = nyse or NYSECalendar()
    start_month = pd.Timestamp(start).to_period("M")
    end_month = pd.Timestamp(end).to_period("M")
    expiries: list[pd.Timestamp] = []
    for period in pd.period_range(start_month, end_month, freq="M"):
        year, month = period.year, period.month
        friday_column = month_calendar.monthcalendar(year, month)
        fridays = [
            week[month_calendar.FRIDAY]
            for week in friday_column
            if week[month_calendar.FRIDAY]
        ]
        nominal = pd.Timestamp(date(year, month, fridays[2]))
        expiry = (
            nominal
            if exchange.is_session(nominal)
            else exchange.previous_session(nominal)
        )
        if pd.Timestamp(start) <= expiry <= pd.Timestamp(end):
            expiries.append(expiry)
    return pd.DatetimeIndex(expiries)


def opex_week_labels(
    start: date, end: date, *, nyse: NYSECalendar | None = None
) -> pd.DataFrame:
    """Label every NYSE session in the Monday-through-expiry OPEX week."""

    exchange = nyse or NYSECalendar()
    sessions = exchange.sessions(start, end)
    expiries = monthly_opex_sessions(start, end, nyse=exchange)
    weeks = {expiry.to_period("W-FRI") for expiry in expiries}
    return pd.DataFrame(
        {
            "session": sessions,
            "opex_week": [session.to_period("W-FRI") in weeks for session in sessions],
            "opex_day": [session in expiries for session in sessions],
        }
    )


@dataclass(frozen=True, slots=True)
class FOMCMeeting:
    """Official FOMC meeting interval and decision timestamp."""

    start: date
    end: date
    announcement_time: datetime
    scheduled: bool
    projections: bool
    source_url: str
    label: str

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("FOMC meeting end precedes start")
        if self.announcement_time.tzinfo is None:
            raise ValueError("announcement_time must be timezone-aware")


class _FedHTMLParser(HTMLParser):
    def __init__(self, *, historical_year: int | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self.historical_year = historical_year
        self.year: int | None = historical_year
        self.month: str | None = None
        self.records: list[tuple[int, str, str]] = []
        self.historical_labels: list[str] = []
        self._capture: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag in {"h4", "h5"}:
            self._capture = "heading"
            self._parts = []
        elif "fomc-meeting__month" in classes:
            self._capture = "month"
            self._parts = []
        elif "fomc-meeting__date" in classes:
            self._capture = "date"
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._capture == "heading" and tag in {"h4", "h5"}:
            text = " ".join("".join(self._parts).split())
            year_match = re.search(r"\b(19|20)\d{2}\b", text)
            if year_match:
                self.year = int(year_match.group())
            if self.historical_year is not None and (
                "Meeting" in text or "Conference Call" in text
            ):
                self.historical_labels.append(text)
            self._capture = None
        elif self._capture == "month" and tag == "div":
            self.month = " ".join("".join(self._parts).split())
            self._capture = None
        elif self._capture == "date" and tag == "div":
            text = " ".join("".join(self._parts).split())
            if self.year is not None and self.month is not None:
                self.records.append((self.year, self.month, text))
            self._capture = None


def parse_fomc_calendar_html(
    html: str, source_url: str = FED_CALENDAR_URL
) -> tuple[FOMCMeeting, ...]:
    """Parse current/future official Federal Reserve calendar markup."""

    parser = _FedHTMLParser()
    parser.feed(html)
    meetings = [
        _meeting_from_parts(year, month, days, source_url, scheduled=True)
        for year, month, days in parser.records
    ]
    return _dedupe_meetings(meeting for meeting in meetings if meeting is not None)


def parse_fomc_historical_html(
    html: str, year: int, source_url: str
) -> tuple[FOMCMeeting, ...]:
    """Parse official historical-page meeting and conference-call headings."""

    parser = _FedHTMLParser(historical_year=year)
    parser.feed(html)
    meetings: list[FOMCMeeting] = []
    for label in parser.historical_labels:
        # Examples: "January 29-30 Meeting - 2019" and
        # "October 7 Conference Call - 2008".
        match = re.search(
            r"([A-Za-z]+(?:/[A-Za-z]+)?)\s+(\d{1,2}(?:-\d{1,2})?)\s+"
            r"(Meeting|Conference Call)",
            label,
            re.IGNORECASE,
        )
        if not match:
            continue
        meeting = _meeting_from_parts(
            year,
            match.group(1),
            match.group(2),
            source_url,
            scheduled=match.group(3).lower() == "meeting",
            label=label,
        )
        if meeting is not None:
            meetings.append(meeting)
    return _dedupe_meetings(meetings)


class FOMCCalendarSource:
    """Download official current and historical FOMC meeting pages."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        raw_sink: RawPayloadSink | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._client = client
        self._raw_sink = raw_sink
        self._timeout = timeout

    async def fetch_meetings(
        self,
        start: date,
        end: date,
        *,
        include_unscheduled: bool = False,
    ) -> tuple[FOMCMeeting, ...]:
        """Fetch meetings intersecting ``[start, end]`` from official pages."""

        if end < start:
            raise ValueError("end must be on or after start")
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={"User-Agent": "EdgeStack research data client"},
        )
        try:
            meetings: list[FOMCMeeting] = []
            if end.year >= 2021:
                response = await client.get(FED_CALENDAR_URL)
                response.raise_for_status()
                self._capture(response, FED_CALENDAR_URL)
                meetings.extend(parse_fomc_calendar_html(response.text))
            for year in range(start.year, min(end.year, 2020) + 1):
                url = FED_HISTORICAL_URL.format(year=year)
                response = await client.get(url)
                response.raise_for_status()
                self._capture(response, url)
                meetings.extend(parse_fomc_historical_html(response.text, year, url))
        finally:
            if owns_client:
                await client.aclose()
        return tuple(
            meeting
            for meeting in _dedupe_meetings(meetings)
            if meeting.end >= start
            and meeting.start <= end
            and (include_unscheduled or meeting.scheduled)
        )

    def _capture(self, response: httpx.Response, url: str) -> None:
        if self._raw_sink is None:
            return
        fetched_at = datetime.now(UTC)
        self._raw_sink.store(
            RawPayload(
                source="federal_reserve",
                asset=None,
                fetched_at=fetched_at,
                media_type=response.headers.get("content-type", "text/html").split(";")[
                    0
                ],
                body=response.content,
                request_url=url,
                status_code=response.status_code,
                response_headers={
                    key.lower(): value
                    for key, value in response.headers.items()
                    if key.lower() in {"content-type", "etag", "last-modified", "date"}
                },
            )
        )


def fomc_event_labels(
    start: date,
    end: date,
    meetings: Sequence[FOMCMeeting],
    *,
    nyse: NYSECalendar | None = None,
) -> pd.DataFrame:
    """Label FOMC day-before, day-of, and announcement-week NYSE sessions."""

    exchange = nyse or NYSECalendar()
    sessions = exchange.sessions(start, end)
    announcements = {
        pd.Timestamp(meeting.end) for meeting in meetings if start <= meeting.end <= end
    }
    day_before = {
        exchange.previous_session(announcement)
        for announcement in announcements
        if announcement > pd.Timestamp(start)
    }
    week_keys = {announcement.to_period("W-SUN") for announcement in announcements}
    return pd.DataFrame(
        {
            "session": sessions,
            "fomc_day_before": [session in day_before for session in sessions],
            "fomc_day_of": [session in announcements for session in sessions],
            "fomc_event_week": [
                session.to_period("W-SUN") in week_keys for session in sessions
            ],
        }
    )


def _meeting_from_parts(
    year: int,
    month_text: str,
    day_text: str,
    source_url: str,
    *,
    scheduled: bool,
    label: str | None = None,
) -> FOMCMeeting | None:
    month_names = [part.strip() for part in month_text.split("/")]
    months: list[int] = []
    for part in month_names:
        try:
            months.append(datetime.strptime(part[:3], "%b").month)
        except ValueError:
            return None
    days = [int(value) for value in re.findall(r"\d{1,2}", day_text)[:2]]
    if not days:
        return None
    start_month = months[0]
    end_month = months[-1] if len(months) > 1 else start_month
    start_day = days[0]
    end_day = days[-1]
    end_year = year + (1 if end_month < start_month else 0)
    try:
        start_date = date(year, start_month, start_day)
        end_date = date(end_year, end_month, end_day)
    except ValueError:
        return None
    announcement = datetime.combine(end_date, time(14), tzinfo=NEW_YORK)
    text = label or f"{month_text} {day_text} FOMC meeting"
    return FOMCMeeting(
        start_date,
        end_date,
        announcement,
        scheduled,
        "*" in day_text,
        source_url,
        text,
    )


def _dedupe_meetings(meetings: Iterable[FOMCMeeting]) -> tuple[FOMCMeeting, ...]:
    by_key: dict[tuple[date, date, bool], FOMCMeeting] = {}
    for meeting in meetings:
        by_key[(meeting.start, meeting.end, meeting.scheduled)] = meeting
    return tuple(sorted(by_key.values(), key=lambda item: (item.end, item.start)))


__all__ = [
    "FED_CALENDAR_URL",
    "FED_HISTORICAL_URL",
    "FOMCCalendarSource",
    "FOMCMeeting",
    "NYSECalendar",
    "fomc_event_labels",
    "holiday_adjacency",
    "monthly_opex_sessions",
    "opex_week_labels",
    "parse_fomc_calendar_html",
    "parse_fomc_historical_html",
]
