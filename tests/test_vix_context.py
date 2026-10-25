"""VIX dispatch and real recurrent serialization without local optimization."""

from copy import deepcopy

from gymnasium import Env, spaces
import numpy as np
import pytest
import torch
from sb3_contrib import RecurrentPPO

from rl_stocks.training import vix_context as study
from rl_stocks.training.vix_spec import parameters


class ModelSpaces(Env):
    action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)

    def __init__(self, size):
        self.observation_space = spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32)


def construct(arm, seed=11):
    from rl_stocks.rl.registry.factory import create

    torch.set_num_threads(1)
    return create(
        "recurrent_ppo",
        ModelSpaces(21 if arm == "legacy" else 23),
        seed=seed,
        device="cpu",
        overrides=parameters(True),
    )


def request():
    return dict(
        dataset_id=study.binding()["datasets"]["full"],
        algorithm="recurrent_ppo",
        overrides=parameters(True),
        seed=11,
        smoke=True,
        steps=512,
        every=256,
        initialization="standard",
        policy_variant="native",
        observation_profile="execution_masked",
        start_sampling="stock_etf_cycle_windows",
        reward_kind="net_log_equity",
        reward_scale=1.0,
        training_exposure_mode="unrestricted",
        evaluation_exposure_mode="unit_magnitude",
        final_test=False,
        compact_checkpoints=True,
        milestones=None,
    )


def test_explicit_dispatch_protects_augmented_payloads():
    legacy = dict(schema="daily-indicators-v1", names=["returns"], manifest={})
    assert not study.validate_dispatch(legacy, None)
    assert study.validate_dispatch(legacy, study.PROTOCOL_SHA256)
    for data in [
        dict(legacy, schema="daily-indicators-v1-vix-full-v1"),
        dict(legacy, names=["vix_log_level"]),
        dict(legacy, manifest={"vix_context": {}}),
    ]:
        with pytest.raises(ValueError, match="explicit"):
            study.validate_dispatch(data, None)
        assert study.validate_dispatch(data, study.PROTOCOL_SHA256)


@pytest.mark.parametrize(
    "key,value",
    [
        ("steps", 1024),
        ("every", 512),
        ("seed", True),
        ("smoke", 1),
        ("final_test", True),
        ("reward_scale", True),
        ("compact_checkpoints", False),
        ("milestones", [256, 512]),
        ("evaluation_exposure_mode", "unrestricted"),
        ("training_exposure_mode", "unit_magnitude"),
        ("algorithm", "ppo"),
        ("start_sampling", "legacy_uniform"),
        ("dataset_id", "unknown"),
        ("overrides", {}),
        ("observation_profile", "execution_full"),
    ],
)
def test_fixed_request_rejects_drift(key, value):
    args = request()
    assert study.validate_request(**args) == "full"
    args[key] = value
    with pytest.raises(ValueError):
        study.validate_request(**args)


def test_actual_initial_records_share_all_augmented_tensors_and_rng():
    records = {}
    for arm in ["legacy", "masked", "full"]:
        model = construct(arm)
        study.apply_initialization(model, arm, smoke=True)
        records[arm] = study.model_record(model)
        assert not model.policy.optimizer.state and model.num_timesteps == 0
        with pytest.raises(ValueError):
            study.apply_initialization(model, arm, smoke=True)
    assert study.validate_initial_records(records, smoke=True)["sampling_rng_identical"]
    changed = deepcopy(records)
    changed["full"]["optimizer_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        study.validate_initial_records(changed, smoke=True)
    changed = deepcopy(records)
    changed["legacy"]["rng"]["numpy"] = "0" * 64
    with pytest.raises(ValueError):
        study.validate_initial_records(changed, smoke=True)


def test_actual_save_load_preserves_manually_populated_adam_and_predictions(tmp_path):
    model = construct("full")
    study.apply_initialization(model, "full", smoke=True)
    # Serialization fixture only: no gradient, optimizer.step or learn call.
    model.num_timesteps = 512
    model._n_updates = 40
    for parameter in model.policy.parameters():
        model.policy.optimizer.state[parameter] = dict(
            step=torch.tensor(80.0),
            exp_avg=torch.full_like(parameter, 0.01),
            exp_avg_sq=torch.full_like(parameter, 0.02),
        )
    before = study.model_record(model)
    study.validate_record(before, "full", True)
    path = tmp_path / "fixture.zip"
    model.save(path)
    restored = RecurrentPPO.load(path, device="cpu")
    after = study.model_record(restored)
    study.validate_record(after, "full", True)
    # SB3 loading does not promise restoration of global sampling RNG.
    assert {k: v for k, v in before.items() if k != "rng"} == {
        k: v for k, v in after.items() if k != "rng"
    }
    obs = np.linspace(-1, 1, 23, dtype=np.float32)
    left, ls = model.predict(obs, deterministic=True)
    right, rs = restored.predict(obs, deterministic=True)
    np.testing.assert_array_equal(left, right)
    for a, b in zip(ls, rs, strict=True):
        np.testing.assert_array_equal(a, b)
    restored.policy.optimizer.param_groups[0]["decoupled_weight_decay"] = True
    with pytest.raises(ValueError, match="coupled Adam"):
        study.validate_record(study.model_record(restored), "full", True)


@pytest.mark.parametrize("registered", [False, True])
def test_runner_rejects_invalid_vix_before_environment_or_model(monkeypatch, registered):
    from rl_stocks.training import runner
    from rl_stocks.training import dependencies

    def forbidden(*args, **kwargs):
        pytest.fail("invalid request reached environment/model construction")

    data = dict(schema="daily-indicators-v1-vix-full-v1", names=["vix_log_level"], manifest={})
    monkeypatch.setattr(runner, "read_dataset", lambda path: data)
    monkeypatch.setattr(dependencies, "verify_environment", lambda: {})
    monkeypatch.setattr(runner, "TradingEnv", forbidden)
    monkeypatch.setattr(runner, "create", forbidden)
    args = request()
    dataset_id = args.pop("dataset_id")
    args["evaluation_every"] = args.pop("every")
    args["steps"] = 1024
    args["study_protocol_sha256"] = study.PROTOCOL_SHA256 if registered else None
    with pytest.raises(ValueError, match="transitions|explicit"):
        runner.train(dataset_id, **args)
