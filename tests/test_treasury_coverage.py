"""Both date/row denominators and causal split/UTC requirements."""

import pytest

from rl_stocks.data.validation.treasury_coverage import decisions, report


def data():
    def episode(symbol, split, day):
        return dict(
            symbol=symbol,
            split=split,
            bars=[dict(symbol=symbol, session=day, available_time=day + "T21:15:00+00:00")],
        )

    return dict(
        manifest={"included_splits": ["train", "validation"]},
        episodes={
            "a": episode("A", "train", "2010-01-04"),
            "b": episode("B", "validation", "2021-01-04"),
        },
    )


def test_outcome_free_consumer_collection_and_reject_test():
    value = data()
    times, groups, splits = decisions(value)
    assert splits == dict(train=1, validation=1) and len(times) == 2
    value["episodes"]["a"]["bars"][0]["session"] = "2024-01-02"
    with pytest.raises(ValueError):
        decisions(value)


def test_every_year_is_required_and_both_denominators_reported():
    value = data()
    times, groups, _ = decisions(value)
    joined = {d: dict(ready=True, age_xnys_sessions=2, age_calendar_days=2) for d in times}
    spec = dict(
        eras={"train": ["2010-01-01", "2020-12-31"], "validation": ["2021-01-01", "2023-12-31"]},
        coverage_gate=dict(
            minimum_overall_fraction=0.995, minimum_era_fraction=0.995, minimum_year_fraction=0.99
        ),
    )
    result = report(times, groups, joined, spec)
    assert not result["coverage_passed"]  # Missing years cannot silently pass.
    assert result["unique_dates"]["overall"]["fraction"] == 1
    assert result["consumer_rows"]["overall"]["fraction"] == 1
    assert len(result["per_symbol_split"]) == 2
    joined.pop(next(iter(joined)))
    with pytest.raises(ValueError):
        report(times, groups, joined, spec)
