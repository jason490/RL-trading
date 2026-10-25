"""Disabled, date-bounded Treasury context under an explicit lag assumption."""

from bisect import bisect_right
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation

from rl_stocks.data.providers.treasury.xml import FIELDS

NAMES = ("treasury_2y_pct", "treasury_10y_minus_2y_pp")


def validate_rows(rows):
    dates = []
    for row in rows:
        if set(row) != {"observation_date", *FIELDS}:
            raise ValueError("exact normalized Treasury source schema required")
        day = date.fromisoformat(row["observation_date"])
        if not date(2009, 1, 1) <= day < date(2024, 1, 1) or day.weekday() >= 5:
            raise ValueError("pre2024 Treasury business dates required")
        dates.append(day.isoformat())
        for name in FIELDS:
            value = row[name]
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError("decimal source strings or explicit null required")
            try:
                number = Decimal(value)
            except InvalidOperation as error:
                raise ValueError("invalid Treasury source number") from error
            if not number.is_finite() or not -100 <= number <= 100:
                raise ValueError("finite percent Treasury source required")
    if not rows or dates != sorted(set(dates)):
        raise ValueError("nonempty unique ordered Treasury dates required")


def context(rows, sessions, *, decision_times):
    import exchange_calendars as xc

    validate_rows(rows)
    if (
        not sessions
        or sessions != sorted(set(sessions))
        or any(not "2010-01-01" <= s < "2024-01-01" for s in sessions)
        or set(decision_times) != set(sessions)
    ):
        raise ValueError("exact pre2024 train/development decision sessions required")
    calendar = xc.get_calendar("XNYS", start="2009-01-01", end="2024-01-15")
    days = [str(s.date()) for s in calendar.sessions]
    indices = {day: i for i, day in enumerate(days)}
    available, records = [], []
    for row in rows:
        # First following XNYS session is bisect_right, even when Treasury's
        # observation date is an NYSE holiday. One more session gives lag2.
        index = bisect_right(days, row["observation_date"]) + 1
        stamp = (calendar.session_close(days[index]) + timedelta(minutes=15)).to_pydatetime()
        available.append(stamp)
        records.append(row)
    result = {}
    for session in sessions:
        if session not in indices:
            raise ValueError("decision must be an XNYS session")
        decision = datetime.fromisoformat(decision_times[session])
        if decision.tzinfo is None or decision.utcoffset().total_seconds() != 0:
            raise ValueError("UTC decision time required")
        if decision.date().isoformat() != session:
            raise ValueError("decision UTC date must match session")
        at = bisect_right(available, decision) - 1
        if at < 0:
            result[session] = dict(ready=False, reason="not_yet_available")
            continue
        row = records[at]
        source_day = row["observation_date"]
        age_xnys = indices[session] - bisect_right(days, source_day) + 1
        age_calendar = (date.fromisoformat(session) - date.fromisoformat(source_day)).days
        value = dict(
            observation_date=source_day,
            available_time=available[at].isoformat(),
            age_xnys_sessions=age_xnys,
            age_calendar_days=age_calendar,
            historical_publication_vintages_verified=False,
        )
        if not 2 <= age_xnys <= 3 or not 0 <= age_calendar <= 7:
            result[session] = dict(value, ready=False, reason="observation_too_old")
        elif any(row[name] is None for name in FIELDS):
            result[session] = dict(value, ready=False, reason="missing_maturity")
        else:
            two, ten = (Decimal(row[name]) for name in FIELDS)
            result[session] = dict(value, ready=True, values=[float(two), float(ten - two)])
    return result
