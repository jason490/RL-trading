"""Single-collector bounded annual pacing; caller holds the operation lock."""

import time

from rl_stocks.data.providers.treasury.acquire import cached_year, fetch_year


def collect(
    cache,
    spec,
    deadline,
    progress,
    *,
    fetch=fetch_year,
    cached=cached_year,
    clock=time.monotonic,
    sleep=time.sleep,
):
    requests_started = 0
    last_request = None
    sources = []
    for year in spec["years"]:
        if clock() >= deadline:
            raise TimeoutError("Treasury collection deadline")
        existing = cached(cache, year)
        if existing is None:
            if requests_started >= spec["maximum_requests"]:
                raise ValueError("fixed Treasury request cap")
            delay = (
                0
                if last_request is None
                else max(0, spec["minimum_request_spacing_seconds"] - (clock() - last_request))
            )
            if clock() + delay >= deadline:
                raise TimeoutError("Treasury pacing deadline")
            if delay:
                sleep(delay)
            last_request = clock()
            requests_started += 1
            progress(
                dict(
                    status="source_request",
                    year=year,
                    requests_started=requests_started,
                    completed_years=[s["receipt"]["year"] for s in sources],
                )
            )
            existing = fetch(cache, year, deadline=deadline)
        sources.append(existing)
        progress(
            dict(
                status="source_retained",
                year=year,
                requests_started=requests_started,
                completed_years=[s["receipt"]["year"] for s in sources],
            )
        )
    return sources, requests_started
