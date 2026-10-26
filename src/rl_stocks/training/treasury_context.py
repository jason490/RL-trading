"""Registered Treasury data/request validation; runner enablement is separate."""

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.data.generation.treasury_contract import validate_frozen_contract
from rl_stocks.data.generation.treasury_payload import SCHEMAS
from rl_stocks.features.treasury_context import NAMES
from rl_stocks.training.treasury_spec import (
    PROTOCOL_SHA256 as PROTOCOL_SHA256,
    binding,
    parameters,
    protocol,
)
from rl_stocks.training.treasury_initialization import validate_assets


def validate_dispatch(data, protocol_id):
    family = (
        data.get("schema") in SCHEMAS.values()
        or str(data.get("schema", "")).startswith("daily-indicators-v1-treasury-")
        or "treasury_context" in data.get("manifest", {})
        or bool(set(NAMES) & set(data.get("names", [])))
    )
    if family and protocol_id != PROTOCOL_SHA256:
        raise ValueError("Treasury payload requires explicit registered study dispatch")
    return protocol_id == PROTOCOL_SHA256


def market_features(arm):
    if arm not in protocol()["arms"]:
        raise ValueError("fixed Treasury arm required")
    return len(binding()["dataset_contracts"][arm]["names"])


def validate_payload(data, arm):
    spec = binding()
    if arm not in protocol()["arms"]:
        raise ValueError("fixed Treasury arm required")
    contract = spec["dataset_contracts"][arm]
    if (
        digest(data) != spec["datasets"][arm]
        or data["schema"] != contract["schema"]
        or data["names"] != contract["names"]
        or digest(data["normalizer"]) != contract["normalizer_sha256"]
        or data["manifest"]["included_splits"] != ["train", "validation"]
    ):
        raise ValueError("fixed Treasury dataset/schema/preprocessing required")
    if arm == "legacy":
        from rl_stocks.training.diversity import validate_payload as original

        original(data, "expanded")
    else:
        lineage = data["manifest"]["treasury_context"]
        mode = validate_frozen_contract(
            dict(
                feature_schema=data["schema"],
                names=data["names"],
                normalizer=data["normalizer"],
                data_manifest=data["manifest"],
            )
        )
        if (
            mode != arm
            or lineage["source_dataset"] != spec["datasets"]["legacy"]
            or lineage["source_context"] != spec["source_context"]
            or lineage["eligibility_sha256"] != contract["eligibility_sha256"]
            or lineage["extra_normalizer_fit_rows"] != contract["extra_normalizer_fit_rows"]
        ):
            raise ValueError("fixed Treasury source/readiness lineage required")
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
    arms = [a for a, d in binding()["datasets"].items() if d == dataset_id]
    if (
        len(arms) != 1
        or algorithm != spec["algorithm"]
        or type(seed) is not int
        or seed not in spec["confirmation_seeds"]
        or type(smoke) is not bool
        or smoke
        and seed != spec["screen_seed"]
    ):
        raise ValueError("fixed Treasury algorithm/data/seed/phase required")
    arm = arms[0]
    if canonical(overrides) != canonical(parameters(arm, smoke)):
        raise ValueError("fixed Treasury parameters required before model construction")
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
    if canonical(actual) != canonical(expected):
        raise ValueError("fixed Treasury input/reward/execution/checkpoint/CPU contract required")
    target = spec["smoke_steps" if smoke else "steps_per_cell"]
    interval = spec["smoke_evaluation_every" if smoke else "evaluation_every"]
    if (
        type(steps) is not int
        or steps != target
        or type(every) is not int
        or every != interval
        or milestones is not None
        and canonical(milestones) != canonical([target])
    ):
        raise ValueError("fixed Treasury transitions/cadence/single milestone required")
    validate_assets()
    return arm
