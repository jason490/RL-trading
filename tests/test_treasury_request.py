"""Strict Treasury request contracts reject changes before construction."""

import pytest

from rl_stocks.training import treasury_context as study
from rl_stocks.training.treasury_spec import binding, parameters


def request(arm="full", smoke=True):
    return dict(
        dataset_id=binding()["datasets"][arm],
        algorithm="recurrent_ppo",
        overrides=parameters(arm, smoke),
        seed=11,
        smoke=smoke,
        steps=512 if smoke else 100000,
        every=256 if smoke else 10000,
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


@pytest.mark.parametrize("arm", ["legacy", "masked", "full"])
def test_registered_requests_and_full_input_shapes(arm):
    for smoke in [True, False]:
        assert study.validate_request(**request(arm, smoke)) == arm
    assert study.market_features(arm) == (7 if arm == "legacy" else 9)


@pytest.mark.parametrize(
    "field,value",
    [
        ("algorithm", "sac"),
        ("seed", True),
        ("smoke", 1),
        ("steps", 1024),
        ("every", 128),
        ("initialization", "wide"),
        ("observation_profile", "execution_full"),
        ("start_sampling", "legacy_uniform"),
        ("reward_kind", "initial_capital_equity"),
        ("reward_scale", 2.0),
        ("training_exposure_mode", "unit_magnitude"),
        ("evaluation_exposure_mode", "unrestricted"),
        ("final_test", True),
        ("compact_checkpoints", False),
        ("milestones", [256, 512]),
        ("device", "cuda"),
    ],
)
def test_no_parameter_or_execution_grid(field, value):
    args = request()
    args[field] = value
    with pytest.raises(ValueError):
        study.validate_request(**args)


def test_typed_parameter_alias_and_dispatch_bypass_reject():
    args = request()
    args["overrides"]["ent_coef"] = False
    with pytest.raises(ValueError):
        study.validate_request(**args)
    data = dict(schema="daily-indicators-v1-treasury-full-v1", manifest={}, names=[])
    for proof in [None, "other"]:
        with pytest.raises(ValueError, match="explicit"):
            study.validate_dispatch(data, proof)
    assert study.validate_dispatch(data, study.PROTOCOL_SHA256)
    assert not study.validate_dispatch(
        dict(schema="daily-indicators-v1", names=[], manifest={}), None
    )
    with pytest.raises(ValueError):
        study.validate_payload(dict(schema="bad"), "full")


@pytest.mark.parametrize("registered", [True, False])
def test_runner_rejects_bad_request_before_environment_or_policy(monkeypatch, registered):
    from rl_stocks.training import runner, dependencies

    args = request()
    dataset = args.pop("dataset_id")
    args["evaluation_every"] = args.pop("every")
    args["steps"] = 1024
    data = dict(
        schema="daily-indicators-v1-treasury-full-v1", names=["treasury_2y_pct"], manifest={}
    )
    monkeypatch.setattr(runner, "read_dataset", lambda p: data)
    monkeypatch.setattr(dependencies, "verify_environment", lambda: {})

    def forbidden(*a, **k):
        pytest.fail("model/environment created before request acceptance")

    monkeypatch.setattr(runner, "create", forbidden)
    monkeypatch.setattr(runner, "TradingEnv", forbidden)
    args["study_protocol_sha256"] = study.PROTOCOL_SHA256 if registered else None
    with pytest.raises(ValueError, match="transitions|explicit"):
        runner.train(dataset, **args)
