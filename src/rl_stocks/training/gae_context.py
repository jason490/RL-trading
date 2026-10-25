"""Explicit GAE study dispatch and fixed preconstruction request/data contracts."""

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.training.gae_spec import protocol, parameters, PROTOCOL_SHA256 as PROTOCOL_SHA256
from rl_stocks.training.gae_initialization import validate_assets


def validate_payload(data):
    spec = protocol()
    if (
        digest(data) != spec["dataset"]
        or data["schema"] != spec["feature_schema"]
        or len(data["names"]) != spec["market_features"]
        or digest(data["normalizer"]) != spec["normalizer_sha256"]
        or data["manifest"]["included_splits"] != ["train", "validation"]
    ):
        raise ValueError("fixed original GAE data/preprocessing required")
    from rl_stocks.training.diversity import validate_payload as original

    original(data, "expanded")
    return data


def validate_request(
    dataset_id,
    algorithm,
    overrides,
    *,
    seed,
    smoke,
    steps,
    every,
    initialization,
    policy_variant,
    observation_profile,
    start_sampling,
    reward_kind,
    reward_scale,
    training_exposure_mode,
    evaluation_exposure_mode,
    final_test,
    compact_checkpoints,
    milestones,
    device,
):
    spec = protocol()
    if (
        type(seed) is not int
        or seed not in spec["confirmation_seeds"]
        or type(smoke) is not bool
        or smoke
        and seed != 11
        or algorithm != spec["algorithm"]
        or dataset_id != spec["dataset"]
    ):
        raise ValueError("fixed GAE algorithm/data/seed/phase required")
    matches = [
        arm for arm in spec["arms"] if canonical(overrides) == canonical(parameters(arm, smoke))
    ]
    if len(matches) != 1:
        raise ValueError("one registered GAE parameter arm required")
    actual = dict(
        initialization=initialization,
        policy_variant=policy_variant,
        observation_profile=observation_profile,
        start_sampling=start_sampling,
        reward_kind=reward_kind,
        reward_scale=reward_scale,
        training_exposure_mode=training_exposure_mode,
        evaluation_exposure_mode=evaluation_exposure_mode,
        final_test=final_test,
        compact_checkpoints=compact_checkpoints,
        device=device,
    )
    expected = dict(
        initialization="standard",
        policy_variant=spec["policy_variant"],
        observation_profile=spec["observation_profile"],
        start_sampling=spec["start_sampling"],
        reward_kind=spec["reward"]["kind"],
        reward_scale=spec["reward"]["scale"],
        training_exposure_mode=spec["action_execution"]["training_exposure_mode"],
        evaluation_exposure_mode=spec["action_execution"]["evaluation_exposure_mode"],
        final_test=False,
        compact_checkpoints=True,
        device="cpu",
    )
    target = spec["smoke_steps" if smoke else "steps_per_cell"]
    interval = spec["smoke_evaluation_every" if smoke else "evaluation_every"]
    if canonical(actual) != canonical(expected):
        raise ValueError("fixed GAE input/reward/execution/checkpoint contract required")
    if (
        type(steps) is not int
        or steps != target
        or type(every) is not int
        or every != interval
        or milestones is not None
        and canonical(milestones) != canonical([target])
    ):
        raise ValueError("fixed GAE steps/cadence/single milestone required")
    validate_assets()
    return matches[0]
