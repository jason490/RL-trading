"""Real untrained models and manual Adam serialization; never optimizer steps."""

from copy import deepcopy

import pytest
import torch
from sb3_contrib import RecurrentPPO

from test_vix_context import ModelSpaces
from rl_stocks.rl.registry.factory import create
from rl_stocks.training.gae_spec import parameters
from rl_stocks.training import gae_initialization as study


def model(arm, seed=11, smoke=True):
    torch.set_num_threads(1)
    return create(
        "recurrent_ppo", ModelSpaces(21), seed=seed, device="cpu", overrides=parameters(arm, smoke)
    )


@pytest.mark.parametrize("seed", [11, 23, 37])
def test_both_arms_share_full_size_weights_rng_and_empty_optimizer(seed):
    records = {}
    for arm in ["control", "mc_trace"]:
        m = model(arm, seed, False)
        owners = [id(p) for p in m.policy.parameters()]
        study.apply_initialization(m, arm, seed, False)
        records[arm] = study.model_record(m)
        assert owners == [id(p) for p in m.policy.parameters()]
        assert not m.policy.optimizer.state and m.num_timesteps == 0
        with pytest.raises(ValueError):
            study.apply_initialization(m, arm, seed, False)
    assert study.validate_initial_records(records, False, seed)["only_gae_lambda_differs"]


def test_smoke_reuses_full64_archive_and_settings_reject_before_mutation():
    assert study.validate_assets()["smoke_reuses_full64"]
    assert study.initialization_entry(11, True) == study.initialization_entry(11, False)
    m = model("control")
    before = study.model_record(m)
    with pytest.raises(ValueError):
        study.apply_initialization(m, "mc_trace", smoke=True)
    assert study.model_record(m) == before
    m.vf_coef = 0.7
    with pytest.raises(ValueError):
        study.apply_initialization(m, "control", smoke=True)
    assert study.model_record(m)["policy_tensor_sha256"] == before["policy_tensor_sha256"]


def test_actual_checkpoint_preserves_gae_and_manually_populated_adam(tmp_path):
    m = model("mc_trace")
    study.apply_initialization(m, "mc_trace", smoke=True)
    assert study.model_record(m)["optimizer"]["parameter_count"] == 61513
    m.num_timesteps = 512
    m._n_updates = 40
    for p in m.policy.parameters():
        m.policy.optimizer.state[p] = dict(
            step=torch.tensor(80.0),
            exp_avg=torch.full_like(p, 0.01),
            exp_avg_sq=torch.full_like(p, 0.02),
        )
    before = study.model_record(m)
    study.validate_record(before, "mc_trace", True)
    path = tmp_path / "fixture.zip"
    m.save(path)
    loaded = RecurrentPPO.load(path, device="cpu")
    after = study.model_record(loaded)
    study.validate_record(after, "mc_trace", True)
    assert {k: v for k, v in before.items() if k != "rng"} == {
        k: v for k, v in after.items() if k != "rng"
    }
    bad = deepcopy(after)
    bad["gae_lambda"] = 0.95
    with pytest.raises(ValueError):
        study.validate_record(bad, "mc_trace", True)
    bad = deepcopy(after)
    bad["optimizer"]["adam_step_max"] = 79
    with pytest.raises(ValueError):
        study.validate_record(bad, "mc_trace", True)
    bad = deepcopy(after)
    bad["coupled_adam"]["group_decoupled_weight_decay"] = [True]
    with pytest.raises(ValueError):
        study.validate_record(bad, "mc_trace", True)
