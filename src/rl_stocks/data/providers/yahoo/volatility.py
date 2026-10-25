"""Optional date-bounded index closes; never executable equity bars."""

from datetime import date
import math

from rl_stocks.core.contracts import utcnow


def validate_closes(rows, start, end):
    if not rows or start >= end or end > date(2024, 1, 1):
        raise ValueError("nonempty pre2024 bounded close history required")
    previous = ""
    for row in rows:
        session = row["session"]
        parsed = date.fromisoformat(session)
        if str(parsed) != session or not start <= parsed < end or session <= previous:
            raise ValueError("ordered unique requested-range observations required")
        if (
            type(row["close"]) not in (int, float)
            or not math.isfinite(row["close"])
            or row["close"] <= 0
        ):
            raise ValueError("positive finite index close required")
        previous = session
    return rows


def fetch_closes(start, end, *, ticker_factory=None):
    if start >= end or end > date(2024, 1, 1):
        raise ValueError("pre2024 request bounds required before network access")
    if ticker_factory is None:
        import yfinance as yf

        ticker_factory = yf.Ticker
    frame = ticker_factory("^VIX").history(
        start=str(start),
        end=str(end),
        interval="1d",
        auto_adjust=False,
        back_adjust=False,
        actions=False,
        repair=False,
        raise_errors=True,
        timeout=30,
    )
    rows = [
        {"session": str(index.date()), "close": float(row.Close)} for index, row in frame.iterrows()
    ]
    validate_closes(rows, start, end)
    return dict(
        symbol="^VIX",
        provider="yahoo",
        feed="yahoo_index_close",
        rows=rows,
        request=dict(start=str(start), end_exclusive=str(end)),
        retrieved_time=utcnow(),
        historical_publication_vintages=False,
        executable=False,
        limitations="Unofficial personal-use current-vintage history; observation dates only, no historical publication timestamps; no feed fallback",
    )
