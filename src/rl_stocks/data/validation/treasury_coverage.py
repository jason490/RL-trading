"""Outcome-free Treasury coverage on frozen train/development consumers."""

from collections import Counter
from datetime import datetime


def decisions(data):
    if data["manifest"]["included_splits"] != ["train", "validation"]:
        raise ValueError("train/development only; retired data forbidden")
    times, groups = {}, {}
    split_counts = Counter()
    for episode in data["episodes"].values():
        split, symbol = episode["split"], episode["symbol"]
        if split not in ("train", "validation") or not episode["bars"]:
            raise ValueError("nonempty train/development consumers required")
        split_counts[split] += 1
        days = []
        for bar in episode["bars"]:
            day = bar["session"]
            first, last = (
                ("2010-01-01", "2020-12-31") if split == "train" else ("2021-01-01", "2023-12-31")
            )
            if bar["symbol"] != symbol or not first <= day <= last:
                raise ValueError("causal date/symbol consumer contract required")
            stamp = datetime.fromisoformat(bar["available_time"])
            if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
                raise ValueError("UTC consumer decision timestamp required")
            times[day] = min(times.get(day, stamp), stamp)
            days.append(day)
        if days != sorted(set(days)):
            raise ValueError("ordered unique episode days required")
        groups.setdefault(symbol + "/" + split, []).extend(days)
    if not times or set(split_counts) != {"train", "validation"}:
        raise ValueError("both consumer splits required")
    return (
        {day: stamp.isoformat() for day, stamp in sorted(times.items())},
        groups,
        dict(split_counts),
    )


def coverage(days, joined):
    ready = sum(joined[d]["ready"] for d in days)
    return dict(
        decisions=len(days),
        ready=ready,
        fraction=None if not days else ready / len(days),
        missing=sorted(set(d for d in days if not joined[d]["ready"])),
    )


def report(times, groups, joined, spec):
    if set(times) != set(joined):
        raise ValueError("exact Treasury decision coverage required")
    unique = sorted(times)
    consumers = [day for days in groups.values() for day in days]
    result = {}
    for kind, days in [("unique_dates", unique), ("consumer_rows", consumers)]:
        eras = {
            name: coverage([d for d in days if first <= d <= last], joined)
            for name, (first, last) in spec["eras"].items()
        }
        years = {
            str(y): coverage([d for d in days if d.startswith(str(y))], joined)
            for y in range(2010, 2024)
        }
        result[kind] = dict(overall=coverage(days, joined), eras=eras, years=years)
    gate = spec["coverage_gate"]
    passed = True
    for entry in result.values():
        values = [(entry["overall"], gate["minimum_overall_fraction"])]
        values += [(v, gate["minimum_era_fraction"]) for v in entry["eras"].values()]
        values += [(v, gate["minimum_year_fraction"]) for v in entry["years"].values()]
        passed = passed and all(
            v["fraction"] is not None and v["fraction"] >= threshold for v, threshold in values
        )
    result.update(
        coverage_passed=passed,
        per_symbol_split={name: coverage(days, joined) for name, days in groups.items()},
        methodology_eras={
            name: coverage([d for d in unique if (d >= "2021-12-06") == after], joined)
            for name, after in [("before_change", False), ("after_change", True)]
        },
        ready_age_xnys_counts=dict(
            Counter(str(v["age_xnys_sessions"]) for v in joined.values() if v["ready"])
        ),
        ready_age_calendar_counts=dict(
            Counter(str(v["age_calendar_days"]) for v in joined.values() if v["ready"])
        ),
        unavailable_reasons=dict(Counter(v["reason"] for v in joined.values() if not v["ready"])),
    )
    return result
