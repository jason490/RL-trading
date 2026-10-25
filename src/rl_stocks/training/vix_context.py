"""Explicit VIX study data/model contracts, shared starts and read-only records."""

import hashlib

import torch

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.data.generation.volatility_payload import validate_frozen_contract, SCHEMAS
from rl_stocks.features.volatility_context import NAMES
from rl_stocks.reporting.optimizer import optimizer_fingerprint
from rl_stocks.training.recurrent_record import (
    model_record as base_record,
    validate_record as base_validate,
)
from rl_stocks.training.shared_initialization import load_initialization
from rl_stocks.training.vix_assets import (
    binding,
    initialization_entry,
    initialization_manifest,
    BINDING_SHA256,
    ROOT,
)
from rl_stocks.training.vix_initialization import seeded_fingerprints
from rl_stocks.training.vix_spec import (
    preregistration,
    parameters,
    market_features,
    PREREGISTRATION_SHA256,
)

PROTOCOL_SHA256 = PREREGISTRATION_SHA256


def validate_dispatch(data, protocol_id):
    is_vix = (
        data["schema"] in SCHEMAS.values()
        or "vix_context" in data["manifest"]
        or bool(set(NAMES) & set(data["names"]))
    )
    if is_vix and protocol_id != PROTOCOL_SHA256:
        raise ValueError("VIX payload requires explicit registered VIX study dispatch")
    return protocol_id == PROTOCOL_SHA256


def validate_payload(data, arm):
    market_features(arm)
    spec = binding()
    contract = spec["dataset_contracts"][arm]
    if (
        digest(data) != spec["datasets"][arm]
        or data["schema"] != contract["schema"]
        or data["names"] != contract["names"]
        or digest(data["normalizer"]) != contract["normalizer_sha256"]
        or data["manifest"]["included_splits"] != ["train", "validation"]
    ):
        raise ValueError("fixed VIX arm dataset/preprocessing required")
    if arm == "legacy":
        from rl_stocks.training.diversity import validate_payload as original

        original(data, "expanded")
    else:
        lineage = data["manifest"]["vix_context"]
        if (
            validate_frozen_contract(
                dict(
                    feature_schema=data["schema"],
                    names=data["names"],
                    normalizer=data["normalizer"],
                    data_manifest=data["manifest"],
                )
            )
            != arm
            or lineage["source_dataset"] != spec["datasets"]["legacy"]
            or lineage["source_context"] != spec["source_context"]
            or lineage["extra_normalizer_fit_rows"] != contract["extra_normalizer_fit_rows"]
        ):
            raise ValueError("fixed VIX source/preprocessing lineage required")
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
):
    spec = preregistration()
    if (
        type(smoke) is not bool
        or type(seed) is not int
        or seed not in spec["confirmation_seeds"]
        or smoke
        and seed != 11
    ):
        raise ValueError("fixed VIX seed/phase required")
    arms = [a for a, d in binding()["datasets"].items() if d == dataset_id]
    if (
        len(arms) != 1
        or algorithm != "recurrent_ppo"
        or canonical(overrides) != canonical(parameters(smoke))
    ):
        raise ValueError("explicit fixed VIX algorithm/data/parameters required")
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
    )
    expected = dict(
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
    )
    if canonical(actual) != canonical(expected):
        raise ValueError("fixed VIX reward/input/execution/checkpoint contract required")
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
        raise ValueError("fixed VIX transitions/cadence/single milestone required")
    return arms[0]


def initialization_spec(arm, seed=11, smoke=False):
    entry = initialization_entry(arm, seed, smoke)
    return dict(
        version=1,
        artifact_sha256=entry["sha256"],
        policy_tensor_sha256=entry["policy_tensor_sha256"],
        rng_sha256=entry["rng_sha256"],
        torch_version=entry["torch_version"],
        generated_without_fitting=True,
        seed=seed,
        smoke=smoke,
        artifact_file=entry["file"],
        binding_sha256=BINDING_SHA256,
    )


def apply_initialization(model, arm, seed=11, smoke=False):
    if (
        type(model.seed) is not int
        or model.seed != seed
        or model.observation_space.shape != (market_features(arm) + 14,)
    ):
        raise ValueError("actual VIX model seed/input shape required")
    if seeded_fingerprints() != initialization_manifest()["seeded_numpy_python"][str(seed)]:
        raise ValueError("native VIX seeded NumPy/Python states required before learning")
    entry = initialization_entry(arm, seed, smoke)
    proof = initialization_spec(arm, seed, smoke)
    loaded = load_initialization(model, ROOT / entry["file"], entry)
    if any(loaded[k] != proof[k] for k in loaded):
        raise ValueError("actual VIX initialization did not restore")
    model._research_initialization = proof
    validate_record(model_record(model), arm, smoke, initial=True, seed=seed)
    return proof


def model_record(model):
    result = base_record(model)
    optimizer = model.policy.optimizer
    result.update(
        seed=model.seed,
        optimizer_sha256=optimizer_fingerprint(model),
        rng=dict(
            torch=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            **seeded_fingerprints(),
        ),
        coupled_adam=dict(
            version=1,
            optimizer_class=type(optimizer).__name__,
            defaults_decoupled_weight_decay=optimizer.defaults.get("decoupled_weight_decay"),
            group_decoupled_weight_decay=[
                g.get("decoupled_weight_decay") for g in optimizer.param_groups
            ],
        ),
    )
    return result


def validate_record(record, arm, smoke=False, *, initial=False, selected_steps=None, seed=11):
    proof = initialization_spec(arm, seed, smoke)
    spec = preregistration()
    base_validate(
        record,
        parameters(smoke),
        market_features(arm) + 14,
        spec["smoke_steps" if smoke else "steps_per_cell"],
        proof,
        initial=initial,
        selected_steps=selected_steps,
        weight_decay=0.0,
    )
    coupling = record.get("coupled_adam", {})
    if (
        canonical(coupling)
        != canonical(
            dict(
                version=1,
                optimizer_class="Adam",
                defaults_decoupled_weight_decay=False,
                group_decoupled_weight_decay=[False],
            )
        )
        or type(record.get("seed")) is not int
        or record["seed"] != seed
    ):
        raise ValueError("actual VIX coupled Adam/seed required")
    value = record.get("optimizer_sha256", "")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("actual VIX optimizer fingerprint required")
    if initial and record.get("rng") != dict(
        torch=proof["rng_sha256"], **initialization_manifest()["seeded_numpy_python"][str(seed)]
    ):
        raise ValueError("actual VIX initial Torch/NumPy/Python streams mismatch")
    return record


def validate_initial_records(records, smoke=False, seed=11):
    if set(records) != set(preregistration()["arms"]):
        raise ValueError("all3 actual VIX initial records required")
    for arm, record in records.items():
        validate_record(record, arm, smoke, initial=True, seed=seed)
    if (
        records["full"] != records["masked"]
        or len({canonical(r["rng"]) for r in records.values()}) != 1
    ):
        raise ValueError("actual shared full/masked tensors and all sampling RNG required")
    return dict(
        full_masked_records_exact=True,
        sampling_rng_identical=True,
        common_columns_and_zero_embedding_verified=True,
    )
