from copy import deepcopy

import exchange_calendars as xc
import pytest

from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.data.generation.volatility_payload import (
    build_payload,
    validate_frozen_contract,
    validate_payload_pair,
)
from rl_stocks.features.registry.assemble import NAMES


def fixture():
    calendar = xc.get_calendar("XNYS", start="2020-01-01", end="2021-04-30")
    rows = [dict(session=str(d.date()), close=20 + i / 10) for i, d in enumerate(calendar.sessions)]
    source = dict(
        schema="daily-indicators-v1",
        names=list(NAMES),
        normalizer=dict(names=list(NAMES), means=[1.0] * 7, scales=[2.0] * 7),
        manifest=dict(
            included_splits=["train", "validation"],
            snapshot="fixture",
            provenance=dict(capabilities=dict(feed="fixture")),
        ),
        episodes={},
    )
    for split, indices in [("train", [30, 31, 32]), ("validation", [290, 291])]:
        source["episodes"][split] = dict(
            symbol="A",
            split=split,
            bars=[
                dict(
                    symbol="A",
                    session=rows[i]["session"],
                    available_time=rows[i]["session"] + "T23:00:00+00:00",
                    close=100.0,
                )
                for i in indices
            ],
            atr=[1.0] * len(indices),
            features=[[float(i)] * 7 for i in indices],
        )
    vix = dict(
        symbol="^VIX",
        provider="yahoo",
        feed="yahoo_index_close",
        rows=rows,
        request=dict(start="2020-01-01", end_exclusive="2024-01-01"),
        retrieved_time="2026-10-25T00:00:00+00:00",
        historical_publication_vintages=False,
        executable=False,
    )
    return source, vix


def manifest(payload):
    return dict(
        feature_schema=payload["schema"],
        names=payload["names"],
        normalizer=payload["normalizer"],
        data_manifest=payload["manifest"],
    )


@pytest.mark.parametrize("mode", ["masked", "full"])
def test_immutable_originals_and_training_only_extra_statistics(mode):
    source, vix = fixture()
    original = deepcopy(source)
    result = build_payload(source, vix, mode)
    assert source == original
    assert validate_payload_pair(source, result, vix, mode) == result
    assert validate_frozen_contract(manifest(result)) == mode
    for key, e in result["episodes"].items():
        assert (
            e["bars"] == source["episodes"][key]["bars"]
            and e["atr"] == source["episodes"][key]["atr"]
        )
        assert [r[:7] for r in e["features"]] == source["episodes"][key]["features"]
    for key in ("names", "means", "scales"):
        assert result["normalizer"][key][:7] == source["normalizer"][key]
    assert result["manifest"]["vix_context"]["extra_normalizer_fit_rows"] == (
        3 if mode == "full" else 0
    )
    if mode == "masked":
        normalizer = normalizer_from(result["normalizer"])
        assert all(
            normalizer.transform(r, result["names"])[7:] == [0.0, 0.0]
            for e in result["episodes"].values()
            for r in e["features"]
        )
    result["episodes"]["train"]["bars"][0]["close"] = 123
    assert source == original


def test_development_values_cannot_change_extra_preprocessing():
    source, vix = fixture()
    original = build_payload(source, vix, "full")
    changed = deepcopy(vix)
    for row in changed["rows"]:
        if row["session"] >= "2021-01-01":
            row["close"] *= 10
    altered = build_payload(source, changed, "full")
    assert altered["normalizer"] == original["normalizer"]
    assert (
        altered["episodes"]["validation"]["features"]
        != original["episodes"]["validation"]["features"]
    )


@pytest.mark.parametrize(
    "kind",
    [
        "version",
        "lineage",
        "preprocessing",
        "vintage",
        "executable",
        "mask",
        "fit_rows",
        "source_feed",
        "source_bounds",
    ],
)
def test_frozen_false_declarations_reject(kind):
    source, vix = fixture()
    result = build_payload(source, vix, "masked")
    frozen = manifest(result)
    c = frozen["data_manifest"]["vix_context"]
    if kind == "version":
        c["version"] = True
    elif kind == "lineage":
        del frozen["data_manifest"]["vix_context"]
    elif kind == "preprocessing":
        frozen["normalizer"]["means"][0] += 1
    elif kind == "vintage":
        c["historical_publication_vintages_verified"] = True
    elif kind == "executable":
        c["executable"] = True
    elif kind == "mask":
        frozen["normalizer"]["means"][7] = 1.0
    elif kind == "source_feed":
        c["source_feed"] = "other"
    elif kind == "source_bounds":
        c["source_request"]["end_exclusive"] = "2025-01-01"
    else:
        c["extra_normalizer_fit_rows"] = True
    with pytest.raises(ValueError):
        validate_frozen_contract(frozen)


def test_missing_context_future_rows_or_modified_payload_reject():
    source, vix = fixture()
    latest = source["episodes"]["train"]["bars"][0]["session"]
    index = next(i for i, r in enumerate(vix["rows"]) if r["session"] == latest)
    missing = deepcopy(vix)
    del missing["rows"][index - 1]
    for mode in ("masked", "full"):
        with pytest.raises(ValueError):
            build_payload(source, missing, mode)
    result = build_payload(source, vix, "full")
    result["episodes"]["validation"]["features"][0][7] += 1
    with pytest.raises(ValueError):
        validate_payload_pair(source, result, vix, "full")
    source["episodes"]["train"]["bars"][0]["session"] = "2024-01-02"
    with pytest.raises(ValueError):
        build_payload(source, vix, "full")


@pytest.mark.parametrize("mode", ["masked", "full"])
def test_untrained_recurrent_frozen_parity_mask_and_state_immutability(tmp_path, mode):
    from dataclasses import asdict
    from gymnasium import Env, spaces
    import numpy as np
    import torch
    from rl_stocks.core.contracts import digest
    from rl_stocks.environment.observations.state import observation_spec
    from rl_stocks.inference.bundle import freeze, FrozenPolicy
    from rl_stocks.risk.exposure import filter_exposure
    from rl_stocks.rl.registry.factory import create
    from rl_stocks.training.vix_spec import parameters, preregistration

    class ModelSpaces(Env):
        observation_space = spaces.Box(-np.inf, np.inf, (23,), dtype=np.float32)
        action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)

    source, vix = fixture()
    data = build_payload(source, vix, mode)
    torch.set_num_threads(1)
    model = create(
        "recurrent_ppo", ModelSpaces(), seed=11, device="cpu", overrides=parameters(True)
    )
    assert sum(p.numel() for p in model.policy.parameters()) == 62537
    model.save(tmp_path / "untrained")
    bundle = freeze(
        tmp_path / "untrained.zip",
        data,
        dict(
            algorithm="recurrent_ppo",
            observation=observation_spec("execution_masked"),
            action_execution=preregistration()["action_execution"],
        ),
        tmp_path / "frozen",
    )
    frozen = FrozenPolicy(bundle, data["schema"], "fixture")
    normal_before = digest(data["normalizer"])
    first = data["episodes"]["train"]["features"][0]
    observation = np.array(
        normalizer_from(data["normalizer"]).transform(first, data["names"]) + [0.0] * 14,
        dtype=np.float32,
    )
    initial = observation.copy()
    state = fstate = None
    for i in range(3):
        expected, state = model.predict(
            observation, state=state, episode_start=np.array([i == 0]), deterministic=True
        )
        caller = None if fstate is None else tuple(s.copy() for s in fstate)
        previous = fstate
        actual, fstate = frozen.predict_recurrent(observation, state=fstate, episode_start=i == 0)
        np.testing.assert_array_equal(actual, filter_exposure(expected, "unit_magnitude"))
        for a, b in zip(state, fstate, strict=True):
            np.testing.assert_array_equal(a, b)
        if caller is not None:
            for a, b in zip(caller, previous, strict=True):
                np.testing.assert_array_equal(a, b)
        if i == 0:
            first_action = actual.copy()
        observation = observation.copy()
        observation[0] += 0.1
    reset, _ = frozen.predict_recurrent(initial, state=fstate, episode_start=True)
    np.testing.assert_array_equal(reset, first_action)
    assert digest(asdict(frozen.normalizer)) == normal_before
    assert model.num_timesteps == 0 and not model.policy.optimizer.state
    if mode == "masked":
        bad = initial.copy()
        bad[7] = 1.0
        with pytest.raises(ValueError, match="masked VIX"):
            frozen.predict_recurrent(bad, state=None, episode_start=True)
