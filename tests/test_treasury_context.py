"""Known exchange holidays, early close, immutable causality and missing facts."""

from copy import deepcopy

import exchange_calendars as xc
import pytest

from rl_stocks.features.treasury_context import context


def row(day, two="1.58", ten="1.88"):
    return dict(observation_date=day, BC_2YEAR=two, BC_10YEAR=ten)


def decisions(days):
    from datetime import timedelta

    c = xc.get_calendar("XNYS", start="2009-01-01", end="2024-01-15")
    return {d: (c.session_close(d) + timedelta(minutes=15)).isoformat() for d in days}


def test_second_following_session_and_explicit_age_cap():
    days = ["2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]
    r = context([row("2020-01-02")], days, decision_times=decisions(days))
    assert not r[days[0]]["ready"]
    assert r[days[1]]["ready"] and r[days[1]]["age_xnys_sessions"] == 2
    assert r[days[1]]["values"] == [1.58, 0.3]
    assert r[days[2]]["ready"] and r[days[2]]["age_xnys_sessions"] == 3
    assert not r[days[3]]["ready"] and r[days[3]]["reason"] == "observation_too_old"


def test_treasury_observation_can_be_nyse_holiday_and_latest_null_does_not_fallback():
    days = ["2010-04-06"]
    r = context(
        [row("2010-04-01"), row("2010-04-02", two=None)], days, decision_times=decisions(days)
    )
    assert r[days[0]]["observation_date"] == "2010-04-02"
    assert r[days[0]]["reason"] == "missing_maturity" and not r[days[0]]["ready"]


def test_hurricane_closures_and_calendar_day_bound():
    days = ["2012-11-01", "2012-11-02", "2012-11-05"]
    r = context([row("2012-10-26")], days, decision_times=decisions(days))
    assert r[days[0]]["ready"] and r[days[0]]["age_calendar_days"] == 6
    assert r[days[1]]["ready"] and r[days[1]]["age_calendar_days"] == 7
    assert not r[days[2]]["ready"]


def test_early_close_and_future_data_do_not_advance_availability_or_mutate():
    rows = [row("2020-11-25")]
    before = deepcopy(rows)
    days = ["2020-11-27", "2020-11-30"]
    stamps = decisions(days)
    base = context(rows, days, decision_times=stamps)
    assert not base[days[0]]["ready"] and base[days[1]]["ready"]
    assert context(rows + [row("2020-12-01")], days, decision_times=stamps) == base
    assert rows == before
    stamps[days[1]] = "2020-11-30T21:00:00+00:00"
    assert not context(rows, days, decision_times=stamps)[days[1]]["ready"]


@pytest.mark.parametrize("kind", ["naive", "offset", "retired", "duplicate", "nan"])
def test_bad_source_and_decision_contracts_reject(kind):
    rows = [row("2020-01-02")]
    days = ["2020-01-06"]
    stamps = decisions(days)
    if kind == "naive":
        stamps[days[0]] = "2020-01-06T21:15:00"
    if kind == "offset":
        stamps[days[0]] = "2020-01-06T16:15:00-05:00"
    if kind == "retired":
        days = ["2024-01-02"]
        stamps = {"2024-01-02": "2024-01-02T21:15:00+00:00"}
    if kind == "duplicate":
        rows = rows + rows
    if kind == "nan":
        rows[0]["BC_2YEAR"] = "NaN"
    with pytest.raises(ValueError):
        context(rows, days, decision_times=stamps)
