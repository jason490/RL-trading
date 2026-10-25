"""Fixed request ceilings, pacing, caching and first-failure stop without network."""

import pytest

from rl_stocks.data.providers.treasury.collection import collect


def test_paced_annual_requests_with_cache_no_extra_request():
    now = [0.0]
    starts = []
    events = []

    def fetch(cache, year, deadline):
        starts.append(now[0])
        return dict(receipt={"year": year}, network_requested=True)

    def cached(cache, year):
        return dict(receipt={"year": year}, network_requested=False) if year == 2010 else None

    spec = dict(years=[2009, 2010, 2011], maximum_requests=3, minimum_request_spacing_seconds=1)
    sources, count = collect(
        None,
        spec,
        10,
        events.append,
        fetch=fetch,
        cached=cached,
        clock=lambda: now[0],
        sleep=lambda delay: now.__setitem__(0, now[0] + delay),
    )
    assert count == 2 and starts == [0, 1] and len(sources) == 3
    assert events[-1]["requests_started"] == 2


def test_first_failure_stops_no_retry():
    calls = []

    def fetch(cache, year, deadline):
        calls.append(year)
        raise ValueError("provider failure")

    spec = dict(years=[2009, 2010], maximum_requests=15, minimum_request_spacing_seconds=1)
    with pytest.raises(ValueError):
        collect(
            None, spec, 10, lambda event: None, fetch=fetch, cached=lambda *a: None, clock=lambda: 0
        )
    assert calls == [2009]


@pytest.mark.parametrize("case", ["cap", "deadline"])
def test_projected_request_cap_and_deadline_stop_before_network(case):
    spec = dict(
        years=[2009], maximum_requests=0 if case == "cap" else 15, minimum_request_spacing_seconds=1
    )
    with pytest.raises((ValueError, TimeoutError)):
        collect(
            None,
            spec,
            0 if case == "deadline" else 10,
            lambda event: None,
            fetch=lambda *a, **k: pytest.fail("network"),
            cached=lambda *a: None,
            clock=lambda: 0,
        )
