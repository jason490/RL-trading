from copy import deepcopy

import pytest

from rl_stocks.training.gae_spec import protocol, parameters, PROTOCOL_SHA256
from rl_stocks.training.gae_context import validate_request


def request():
    return dict(
        dataset_id=protocol()["dataset"],
        algorithm="recurrent_ppo",
        overrides=parameters("mc_trace", True),
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
        device="cpu",
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("dataset_id", "unknown"),
        ("algorithm", "ppo"),
        ("seed", True),
        ("smoke", 1),
        ("steps", 1000),
        ("every", 512),
        ("overrides", {}),
        ("reward_scale", True),
        ("final_test", True),
        ("device", "cuda"),
        ("observation_profile", "execution_full"),
        ("training_exposure_mode", "unit_magnitude"),
        ("evaluation_exposure_mode", "unrestricted"),
        ("milestones", [256, 512]),
        ("compact_checkpoints", False),
    ],
)
def test_fixed_request_and_wrong_settings_reject(key, value):
    args = request()
    assert validate_request(**args) == "mc_trace"
    changed = deepcopy(args)
    changed[key] = value
    with pytest.raises(ValueError):
        validate_request(**changed)


def test_runner_rejects_bad_gae_request_before_constructing_model_or_environment(monkeypatch):
    from rl_stocks.training import runner, dependencies

    def forbidden(*args, **kwargs):
        pytest.fail("invalid request reached model/environment")

    data = dict(schema="daily-indicators-v1", names=["return_1"], manifest={})
    monkeypatch.setattr(runner, "read_dataset", lambda path: data)
    monkeypatch.setattr(dependencies, "verify_environment", lambda: {})
    monkeypatch.setattr(runner, "create", forbidden)
    monkeypatch.setattr(runner, "TradingEnv", forbidden)
    args = request()
    dataset = args.pop("dataset_id")
    args["evaluation_every"] = args.pop("every")
    args["steps"] = 1024
    args["study_protocol_sha256"] = PROTOCOL_SHA256
    with pytest.raises(ValueError, match="steps/cadence"):
        runner.train(dataset, **args)
