"""Optional lagged implied-volatility context, independent of rewards/returns."""

from datetime import date, datetime, timedelta
import math

from rl_stocks.data.providers.yahoo.volatility import validate_closes

NAMES = ("vix_log_level", "vix_log_change20")


def context(rows, sessions, *, decision_times):
    """Exact previous-XNYS-session and21st-prior-session closes; no stale fill."""
    import exchange_calendars as xc

    if not sessions or sessions != sorted(set(sessions)) or sessions[-1] >= "2024-01-01":
        raise ValueError("unique ordered pre2024 decision sessions required")
    start = date.fromisoformat(rows[0]["session"]) if rows else date(2009, 1, 1)
    validate_closes(rows, start, date(2024, 1, 1))
    calendar = xc.get_calendar("XNYS", start=str(start), end=max(sessions[-1], rows[-1]["session"]))
    days = [str(s.date()) for s in calendar.sessions]
    indices = {d: i for i, d in enumerate(days)}
    values = {r["session"]: r["close"] for r in rows}
    if any(d not in indices for d in values):
        raise ValueError("source observation outside requested XNYS calendar")
    result = {}
    for session in sessions:
        if session not in indices:
            raise ValueError("decision must be an XNYS session")
        i = indices[session]
        if i < 21:
            result[session] = dict(ready=False, reason="warmup")
            continue
        latest, older = days[i - 1], days[i - 21]
        if latest not in values or older not in values:
            result[session] = dict(
                ready=False,
                reason="missing_exact_source",
                source_session=latest,
                lookback_session=older,
            )
            continue
        available = calendar.session_close(session) + timedelta(minutes=15)
        decision = datetime.fromisoformat(decision_times[session])
        if decision.tzinfo is None or decision.utcoffset().total_seconds() != 0:
            raise ValueError("UTC decision timestamp required")
        if decision < available:
            result[session] = dict(ready=False, reason="not_yet_available")
            continue
        result[session] = dict(
            ready=True,
            source_session=latest,
            lookback_session=older,
            available_time=available.isoformat(),
            historical_availability_verified=False,
            values=[math.log(values[latest]), math.log(values[latest] / values[older])],
        )
    return result
