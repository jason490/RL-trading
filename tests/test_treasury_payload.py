"""Readiness, purge and training-only preprocessing without any optimization."""

from copy import deepcopy
from datetime import timedelta

import exchange_calendars as xc
import pytest

from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.treasury_payload import (
    SOURCE_PROTOCOL_SHA256,
    build_payload,
    eligible_bounds,
    validate_payload_pair,
)
from rl_stocks.features.registry.assemble import NAMES
from rl_stocks.environment.sampling.windows import StockEtfCycles


@pytest.fixture(scope="module")
def sources():
    calendar = xc.get_calendar("XNYS", start="2009-01-01", end="2023-12-31")
    dates = [str(d.date()) for d in calendar.sessions]
    train = [d for d in dates if d >= "2010-01-04"][:1000]
    validation = [d for d in dates if d >= "2021-01-04"][:20]
    episodes = {}
    for symbol in ["A", "SPY", "QQQ", "IWM", "DIA"]:
        for split, days in [("train", train), ("validation", validation)]:
            episodes[symbol + "-" + split + "-0"] = dict(
                symbol=symbol,
                split=split,
                bars=[
                    dict(
                        symbol=symbol,
                        session=d,
                        available_time=(
                            calendar.session_close(d) + timedelta(minutes=15)
                        ).isoformat(),
                    )
                    for d in days
                ],
                features=[[0.1] * 7 for d in days],
                atr=[1.0] * len(days),
            )
    baseline = dict(
        schema="daily-indicators-v1",
        names=list(NAMES),
        normalizer=dict(names=list(NAMES), means=[0.0] * 7, scales=[1.0] * 7),
        episodes=episodes,
        manifest=dict(included_splits=["train", "validation"], snapshot="1" * 64),
    )
    rows = [
        dict(
            observation_date=d,
            BC_2YEAR=str(1 + (i % 100) / 100),
            BC_10YEAR=str(1.4 + (i % 100) / 100),
        )
        for i, d in enumerate(dates)
    ]
    missing_source = dates[dates.index(train[100]) - 2]
    next(r for r in rows if r["observation_date"] == missing_source).update(
        BC_2YEAR=None, BC_10YEAR=None
    )
    treasury = dict(
        schema="treasury-annual-source-v1",
        protocol_sha256=SOURCE_PROTOCOL_SHA256,
        receipts=[f"{i:064x}" for i in range(15)],
        rows=rows,
        historical_publication_vintages_verified=False,
    )
    return baseline, treasury


def test_fixed_purge_short_segments_and_no_outcome_selection():
    ready = [True] * 1000
    ready[100] = False
    first, last, counts = eligible_bounds(ready)
    assert (first, last) == (101, 1000)
    assert counts == dict(
        source_rows=1000, unavailable=1, purged=20, short_segment_rows=80, retained=899
    )
    for ready in [[True] * 512, [False] * 1000, [True] * 650 + [False] + [True] * 650]:
        with pytest.raises(ValueError, match="exactly one"):
            eligible_bounds(ready)


def test_mask_and_full_identical_eligibility_original_rows_and_sampling(sources):
    baseline, treasury = sources
    before = digest(baseline)
    source_before = digest(treasury)
    masked = build_payload(baseline, treasury, "masked")
    full = build_payload(baseline, treasury, "full")
    assert (
        masked["manifest"]["treasury_context"]["eligibility"]
        == full["manifest"]["treasury_context"]["eligibility"]
    )
    assert masked["normalizer"]["means"][7:] == [0, 0] and masked["normalizer"]["scales"][7:] == [
        1,
        1,
    ]
    for name in ["names", "means", "scales"]:
        assert (
            masked["normalizer"][name][:7]
            == full["normalizer"][name][:7]
            == baseline["normalizer"][name]
        )
    assert full["manifest"]["treasury_context"]["extra_normalizer_fit_rows"] == 5 * 899
    for key, e in full["episodes"].items():
        a, b = e, masked["episodes"][key]
        original = baseline["episodes"][key]
        start = 101 if e["split"] == "train" else 0
        assert a["bars"] == b["bars"] == original["bars"][start:]
        assert a["atr"] == b["atr"] == original["atr"][start:]
        assert [r[:7] for r in a["features"]] == original["features"][start:]
        assert all(r[7:] == [0, 0] for r in b["features"])
    a = StockEtfCycles([e for e in masked["episodes"].values() if e["split"] == "train"], 512)
    b = StockEtfCycles([e for e in full["episodes"].values() if e["split"] == "train"], 512)
    assert a.bounds == b.bounds == [387] * 5 and a.groups == b.groups
    assert digest(baseline) == before and digest(treasury) == source_before


def test_future_validation_rate_changes_never_fit_preprocessing(sources):
    baseline, treasury = sources
    changed = deepcopy(treasury)
    for row in changed["rows"]:
        if row["observation_date"] >= "2021-01-01":
            row.update(BC_2YEAR="10", BC_10YEAR="15")
    first = build_payload(baseline, treasury, "full")
    second = build_payload(baseline, changed, "full")
    assert first["normalizer"] == second["normalizer"]
    assert first["episodes"]["A-train-0"] == second["episodes"]["A-train-0"]
    assert (
        first["episodes"]["A-validation-0"]["features"]
        != second["episodes"]["A-validation-0"]["features"]
    )


def test_validation_gap_rejected_even_for_mask(sources):
    baseline, treasury = sources
    changed = deepcopy(treasury)
    calendar = xc.get_calendar("XNYS", start="2009-01-01", end="2023-12-31")
    days = [str(d.date()) for d in calendar.sessions]
    day = baseline["episodes"]["A-validation-0"]["bars"][10]["session"]
    source_day = days[days.index(day) - 2]
    next(r for r in changed["rows"] if r["observation_date"] == source_day).update(BC_2YEAR=None)
    for mode in ["masked", "full"]:
        with pytest.raises(ValueError, match="validation"):
            build_payload(baseline, changed, mode)


def test_reconstruction_tampering_wrong_mode_and_auxiliary_fields_reject(sources):
    baseline, treasury = sources
    payload = build_payload(baseline, treasury, "full")
    validate_payload_pair(baseline, payload, treasury, "full")
    payload["normalizer"]["means"][-1] += 1
    with pytest.raises(ValueError, match="exact"):
        validate_payload_pair(baseline, payload, treasury, "full")
    with pytest.raises(ValueError):
        build_payload(baseline, treasury, "unknown")
    changed = deepcopy(baseline)
    changed["episodes"]["A-train-0"]["unsliced"] = [1]
    with pytest.raises(ValueError, match="auxiliary"):
        build_payload(changed, treasury, "full")
