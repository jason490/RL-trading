from datetime import date, datetime
import math
from rl_stocks.core.contracts import Bar, Request

def timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError("timestamps must be UTC")
    return result

def validate(bars: list[Bar]) -> None:
    if not bars:
        raise ValueError("empty dataset")
    previous = {}
    schemas=set()
    for b in bars:
        if not b.symbol or str(date.fromisoformat(b.session))!=b.session:
            raise ValueError("invalid symbol/session identity")
        schemas.add((b.provider,b.feed,b.adjustments))
        if b.session <= previous.get(b.symbol, ""):
            raise ValueError("duplicate or unsorted sessions")
        previous[b.symbol] = b.session
        o, e, a, r = map(timestamp, (b.open_time, b.event_time, b.available_time, b.retrieved_time))
        if not o < e <= a or r < e:
            raise ValueError("invalid event/availability/retrieval chronology")
        values = (b.open, b.high, b.low, b.close, b.volume, b.dividend, b.split)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite bar")
        if min(b.open, b.high, b.low, b.close, b.split) <= 0 or b.volume < 0:
            raise ValueError("invalid prices/volume/split")
        if not b.low <= min(b.open, b.close) <= max(b.open, b.close) <= b.high:
            raise ValueError("invalid OHLC geometry")
        if b.adjustments not in ("raw_with_actions", "split_adjusted_with_dividends"):
            raise ValueError("unsupported executable price adjustment")
    if len(schemas)!=1:
        raise ValueError("mixed providers/feeds/adjustments require an audited harmonization policy")

def coverage(bars: list[Bar], request: Request, expected_sessions=None):
    validate(bars)
    report = {}
    for symbol in request.symbols:
        sessions = [b.session for b in bars if b.symbol == symbol]
        expected = set(expected_sessions or [])
        report[symbol] = dict(count=len(sessions), first=min(sessions, default=None),
            last=max(sessions, default=None), missing=sorted(expected-set(sessions)),
            calendar_verified=expected_sessions is not None)
    return report

def exchange_sessions(request):
    import exchange_calendars as xc
    calendar = xc.get_calendar("XNYS", start=str(request.start), end=str(request.end))
    # Requested endpoints can be holidays/weekends outside the first/last
    # actual session. sessions_in_range rejects e.g. January 1 against bounds.
    return [str(s.date()) for s in calendar.sessions
            if request.start <= s.date() < request.end]
