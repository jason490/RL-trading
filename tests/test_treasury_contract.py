"""Portable schema lineage and exact frozen recurrent masks, without learning."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from test_treasury_payload import synthetic_sources
from test_treasury_initialization import model
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.data.generation.treasury_payload import build_payload
from rl_stocks.data.generation.treasury_contract import validate_frozen_contract
from rl_stocks.environment.observations.state import observation_spec
from rl_stocks.inference.bundle import freeze, FrozenPolicy
from rl_stocks.risk.exposure import filter_exposure
from rl_stocks.training.treasury_initialization import apply_initialization
from rl_stocks.training.treasury_spec import protocol


@pytest.fixture(scope="module")
def sources():
    return synthetic_sources()


def manifest(data):
    return dict(
        feature_schema=data["schema"],
        names=data["names"],
        normalizer=data["normalizer"],
        data_manifest=data["manifest"],
    )


@pytest.mark.parametrize("mode", ["masked", "full"])
def test_portable_complete_contract_and_original_prefix(sources, mode):
    data = build_payload(*sources, mode)
    value = manifest(data)
    assert validate_frozen_contract(value) == mode
    value["normalizer"]["means"][0] += 1
    with pytest.raises(ValueError, match="seven"):
        validate_frozen_contract(value)


@pytest.mark.parametrize(
    "change",
    [
        "lag",
        "age",
        "availability",
        "vintage",
        "fit_rows",
        "eligibility",
        "receipt",
        "schema",
        "missing",
    ],
)
def test_rehashed_semantically_wrong_context_rejected(sources, change):
    data = build_payload(*sources, "full")
    value = manifest(data)
    lineage = value["data_manifest"]["treasury_context"]
    if change == "lag":
        lineage["source_lag_sessions"] = 1
    if change == "age":
        lineage["maximum_age_calendar_days"] = 8
    if change == "availability":
        lineage["availability"] = "same_day"
    if change == "vintage":
        lineage["historical_publication_vintages_verified"] = True
    if change == "fit_rows":
        lineage["extra_normalizer_fit_rows"] += 1
    if change == "eligibility":
        lineage["eligibility"]["A-validation-0"]["source_bounds"][0] = 1
        lineage["eligibility_sha256"] = digest(lineage["eligibility"])
    if change == "receipt":
        lineage["source_receipts"][0] = "not a hash"
    if change == "schema":
        value["feature_schema"] = "daily-indicators-v1"
    if change == "missing":
        value["data_manifest"].pop("treasury_context")
    with pytest.raises(ValueError):
        validate_frozen_contract(value)


def test_other_feature_families_unchanged():
    assert (
        validate_frozen_contract(
            dict(feature_schema="daily-indicators-v1", names=["return_1"], data_manifest={})
        )
        is None
    )


@pytest.mark.parametrize("mode", ["masked", "full"])
def test_untrained_frozen_state_and_normalizer_parity(tmp_path, sources, mode, monkeypatch):
    from sb3_contrib import RecurrentPPO

    def forbidden(*a, **k):
        pytest.fail("local optimization attempted")

    monkeypatch.setattr(RecurrentPPO, "learn", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    baseline = deepcopy(sources[0])
    baseline["manifest"]["provenance"] = dict(capabilities={"feed": "fixture"})
    data = build_payload(baseline, sources[1], mode)
    m = model(mode)
    apply_initialization(m, mode, smoke=True)
    m.save(tmp_path / "untrained")
    bundle = freeze(
        tmp_path / "untrained.zip",
        data,
        dict(
            algorithm="recurrent_ppo",
            observation=observation_spec("execution_masked"),
            action_execution=protocol()["action_execution"],
        ),
        tmp_path / "frozen",
    )
    frozen = FrozenPolicy(bundle, data["schema"], "fixture")
    before = digest(asdict(frozen.normalizer))
    first = data["episodes"]["A-train-0"]["features"][0]
    observation = np.array(
        normalizer_from(data["normalizer"]).transform(first, data["names"]) + [0.0] * 14,
        dtype=np.float32,
    )
    initial = observation.copy()
    state = fstate = None
    for i in range(4):
        expected, state = m.predict(
            observation, state=state, episode_start=np.array([i == 0]), deterministic=True
        )
        caller = None if fstate is None else tuple(a.copy() for a in fstate)
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
    assert (
        digest(asdict(frozen.normalizer)) == before
        and m.num_timesteps == 0
        and not m.policy.optimizer.state
    )
    if mode == "masked":
        bad = initial.copy()
        bad[7] = 1
        with pytest.raises(ValueError, match="masked Treasury"):
            frozen.predict_recurrent(bad, state=None, episode_start=True)
    bad = initial.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        frozen.predict_recurrent(bad, state=None, episode_start=True)
