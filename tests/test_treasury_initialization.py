"""Actual untrained tensors and manual serialization fixtures; no learning."""

from copy import deepcopy

import pytest
import torch
from sb3_contrib import RecurrentPPO

from test_vix_context import ModelSpaces
from rl_stocks.rl.registry.factory import create
from rl_stocks.training.treasury_spec import parameters
from rl_stocks.training import treasury_initialization as study


@pytest.fixture(autouse=True)
def forbid_optimization(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("local optimization attempted")

    monkeypatch.setattr(RecurrentPPO, "learn", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)


def model(arm, seed=11, smoke=True):
    torch.set_num_threads(1)
    return create(
        "recurrent_ppo",
        ModelSpaces(21 if arm == "legacy" else 23),
        seed=seed,
        device="cpu",
        overrides=parameters(arm, smoke),
    )


@pytest.mark.parametrize("seed", [11, 23, 37])
def test_shared_empty_adam_mapped_tensors_and_rng(seed):
    records = {}
    for arm in ["legacy", "masked", "full"]:
        m = model(arm, seed, False)
        owners = [id(p) for p in m.policy.parameters()]
        study.apply_initialization(m, arm, seed, False)
        records[arm] = study.model_record(m)
        assert owners == [id(p) for p in m.policy.parameters()] and not m.policy.optimizer.state
        assert records[arm]["optimizer"]["parameter_count"] == (61513 if arm == "legacy" else 62537)
        with pytest.raises(ValueError):
            study.apply_initialization(m, arm, seed, False)
    assert study.validate_initial_records(records, False, seed)["full_masked_records_exact"]


def test_actual_all_assets_and_smoke_reuses_same_full_geometry():
    value = study.validate_assets()
    assert value["smoke_reuses_full64"] and value["original_cpu_rng_exact"]
    assert not value["models_constructed"]
    for arm in ["legacy", "masked", "full"]:
        assert study.initialization_entry(arm, 11, True) == study.initialization_entry(
            arm, 11, False
        )
    for name in ["/tmp/outside.npz", "../../.env"]:
        with pytest.raises(ValueError):
            study.portable_path(name)


@pytest.mark.parametrize("change", ["value_coef", "normalize", "amsgrad", "gamma_alias"])
def test_bad_settings_reject_before_any_tensor_rng_or_owner_mutation(change):
    m = model("full")
    if change == "value_coef":
        m.vf_coef = 0.7
    if change == "normalize":
        m.normalize_advantage = False
    if change == "amsgrad":
        m.policy.optimizer.param_groups[0]["amsgrad"] = True
    if change == "gamma_alias":
        m.gamma = True
    before = study.model_record(m)
    with pytest.raises(ValueError):
        study.apply_initialization(m, "full", smoke=True)
    assert study.model_record(m) == before


def test_manually_populated_adam_round_trip_and_rng_is_not_promised(tmp_path):
    m = model("full")
    study.apply_initialization(m, "full", smoke=True)
    m.num_timesteps = 512
    m._n_updates = 40
    for p in m.policy.parameters():
        m.policy.optimizer.state[p] = dict(
            step=torch.tensor(80.0),
            exp_avg=torch.full_like(p, 0.01),
            exp_avg_sq=torch.full_like(p, 0.02),
        )
    with torch.no_grad():
        next(m.policy.parameters()).add_(0.001)
    before = study.model_record(m)
    study.validate_record(before, "full", True)
    path = tmp_path / "manual-fixture.zip"
    m.save(path)
    loaded = RecurrentPPO.load(path, device="cpu")
    after = study.model_record(loaded)
    study.validate_record(after, "full", True)
    assert {k: v for k, v in before.items() if k != "rng"} == {
        k: v for k, v in after.items() if k != "rng"
    }
    bad = deepcopy(after)
    bad["optimizer"]["adam_step_min"] = 79
    with pytest.raises(ValueError):
        study.validate_record(bad, "full", True)
