from datetime import date
from types import SimpleNamespace

import exchange_calendars as xc
import numpy as np
import pandas as pd
import pytest

from rl_stocks.data.providers.yahoo.volatility import fetch_closes, validate_closes
from rl_stocks.features.volatility_context import context


def sample():
    cal = xc.get_calendar("XNYS", start="2020-01-01", end="2020-03-31")
    return [{"session": str(d.date()), "close": 20.0 + i} for i, d in enumerate(cal.sessions)]


def test_lagged_context_exact_prior_sessions_and_no_stale_fill():
    rows = sample()
    day = rows[30]["session"]
    values = context(rows, [day], decision_times={day: day + "T23:00:00+00:00"})[day]
    assert values["source_session"] == rows[29]["session"]
    assert values["lookback_session"] == rows[9]["session"]
    assert values["values"] == pytest.approx([np.log(49), np.log(49 / 29)])
    assert values["available_time"].startswith(day)
    assert not values["historical_availability_verified"]
    changed = [dict(r) for r in rows]
    changed[30]["close"] = 999
    assert context(changed, [day], decision_times={day: day + "T23:00:00+00:00"})[day] == values
    missing = [r for r in rows if r["session"] != rows[29]["session"]]
    assert not context(missing, [day], decision_times={day: day + "T23:00:00+00:00"})[day]["ready"]
    assert not context(
        rows,
        [rows[10]["session"]],
        decision_times={rows[10]["session"]: rows[10]["session"] + "T23:00:00+00:00"},
    )[rows[10]["session"]]["ready"]


def test_future_nonpositive_and_duplicates_reject():
    rows = sample()
    for bad in (
        [dict(session="2024-01-02", close=20)],
        [dict(session="2020-01-02", close=0)],
        rows + rows,
    ):
        with pytest.raises(ValueError):
            validate_closes(bad, date(2009, 1, 1), date(2024, 1, 1))
    with pytest.raises(ValueError):
        context(rows, ["2024-01-02"], decision_times={})


def test_date_bounded_adapter_and_network_preflight():
    calls = []

    def factory(symbol):
        assert symbol == "^VIX"

        def history(**kwargs):
            calls.append(kwargs)
            return pd.DataFrame(
                {"Close": [20.0, 21.0]}, index=pd.to_datetime(["2020-01-02", "2020-01-03"])
            )

        return SimpleNamespace(history=history)

    result = fetch_closes(date(2020, 1, 1), date(2020, 2, 1), ticker_factory=factory)
    assert calls[0]["start"] == "2020-01-01" and calls[0]["end"] == "2020-02-01"
    assert calls[0]["repair"] is False and calls[0]["actions"] is False
    assert result["executable"] is False and not result["historical_publication_vintages"]
    with pytest.raises(ValueError):
        fetch_closes(date(2020, 1, 1), date(2025, 1, 1), ticker_factory=factory)
    assert len(calls) == 1


def test_context_not_exposed_before_assumed_availability():
    rows = sample()
    day = rows[30]["session"]
    result = context(rows, [day], decision_times={day: day + "T20:00:00+00:00"})
    assert result[day] == dict(ready=False, reason="not_yet_available")
    with pytest.raises(ValueError):
        context(rows, [day], decision_times={day: day + "T23:00:00"})
