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


def consumer_fixture():
    from rl_stocks.features.registry.assemble import NAMES

    rows = sample()
    episodes = {}
    for split, day in [("train", rows[30]["session"]), ("validation", "2021-01-04")]:
        episodes[split] = dict(
            symbol="A",
            split=split,
            bars=[dict(symbol="A", session=day, available_time=day + "T23:00:00+00:00")],
            features=[[0.0] * 7],
            atr=[1.0],
        )
    return dict(
        schema="daily-indicators-v1",
        names=list(NAMES),
        normalizer=dict(names=list(NAMES)),
        manifest=dict(included_splits=["train", "validation"]),
        episodes=episodes,
    )


def test_consumer_alignment_preserves_source_and_reports_missing_context():
    from copy import deepcopy
    from rl_stocks.data.generation.volatility_context import align_context

    data = consumer_fixture()
    before = deepcopy(data)
    joined, counts, symbols = align_context(data, sample())
    assert counts == dict(train=1, validation=1)
    assert symbols == dict(train=["A"], validation=["A"])
    assert joined[data["episodes"]["train"]["bars"][0]["session"]]["ready"]
    assert not joined["2021-01-04"]["ready"]
    assert data == before


def test_consumer_rejects_split_or_row_misalignment():
    from rl_stocks.data.generation.volatility_context import align_context

    data = consumer_fixture()
    data["episodes"]["train"]["bars"][0]["session"] = "2021-01-04"
    with pytest.raises(ValueError):
        align_context(data, sample())
    data = consumer_fixture()
    data["episodes"]["train"]["features"] = []
    with pytest.raises(ValueError):
        align_context(data, sample())
