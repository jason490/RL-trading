"""Pinned common recurrent starts and read-only actual GAE model evidence."""

import hashlib
import json
import math
from pathlib import Path

import torch

from rl_stocks.core.contracts import canonical
from rl_stocks.reporting.optimizer import optimizer_fingerprint
from rl_stocks.training.gae_spec import ROOT, protocol, parameters
from rl_stocks.training.recurrent_record import (
    model_record as base_record,
    validate_record as base_validate,
)
from rl_stocks.training.shared_initialization import load_initialization, read_initialization
from rl_stocks.training.vix_initialization import seeded_fingerprints


def initialization_entry(seed=11, smoke=False):
    spec = protocol()
    shared = spec["shared_initialization"]
    if (
        type(seed) is not int
        or seed not in shared["seeds"]
        or type(smoke) is not bool
        or smoke
        and seed != 11
    ):
        raise ValueError("fixed GAE seed/phase required")
    manifest = ROOT / shared["source_manifest"]
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != shared["source_manifest_sha256"]:
        raise ValueError("original GAE full64 source manifest changed")
    entry = json.loads(manifest.read_text())["artifacts"][f"control-screen-{seed}"]
    if entry["torch_version"] != torch.__version__.split("+")[0]:
        raise ValueError("original GAE initializer Torch version required")
    path = Path(entry["file"])
    resolved = (ROOT / path).resolve()
    if (
        path.is_absolute()
        or ".." in path.parts
        or not resolved.is_relative_to((ROOT / "configs/initializations").resolve())
    ):
        raise ValueError("portable original GAE initializer required")
    if sum(math.prod(t["shape"]) for t in entry["tensors"].values()) != shared["parameter_count"]:
        raise ValueError("full64 GAE initializer geometry required, including smokes")
    return entry


def initialization_spec(seed=11, smoke=False):
    entry = initialization_entry(seed, smoke)
    return dict(
        version=1,
        artifact_sha256=entry["sha256"],
        policy_tensor_sha256=entry["policy_tensor_sha256"],
        rng_sha256=entry["rng_sha256"],
        torch_version=entry["torch_version"],
        generated_without_fitting=True,
        source_manifest_sha256=protocol()["shared_initialization"]["source_manifest_sha256"],
        artifact_file=entry["file"],
        seed=seed,
        smoke=smoke,
    )


def validate_assets():
    for seed in protocol()["shared_initialization"]["seeds"]:
        entry = initialization_entry(seed)
        _, rng = read_initialization((ROOT / entry["file"]).read_bytes(), entry)
        torch.Generator(device="cpu").set_state(torch.from_numpy(rng))
    return dict(passed=True, seeds=[11, 23, 37], parameter_count=61513, smoke_reuses_full64=True)


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
        effective_defaults={
            k: getattr(model, k) for k in protocol()["unchanged_effective_defaults"]
        },
        coupled_adam=dict(
            optimizer_class=type(optimizer).__name__,
            defaults_decoupled_weight_decay=optimizer.defaults.get("decoupled_weight_decay"),
            group_decoupled_weight_decay=[
                g.get("decoupled_weight_decay") for g in optimizer.param_groups
            ],
        ),
    )
    return result


def _base_validate(record, arm, smoke, proof, initial, selected_steps):
    spec = protocol()
    params = parameters(arm, smoke)
    for key in (
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "ent_coef",
        "learning_rate",
    ):
        if canonical(record.get(key)) != canonical(params[key]):
            raise ValueError("typed actual GAE algorithm parameters required")
    base_validate(
        record,
        parameters(arm, smoke),
        21,
        spec["smoke_steps" if smoke else "steps_per_cell"],
        proof,
        initial=initial,
        selected_steps=selected_steps,
        weight_decay=0.0,
    )
    if canonical(record.get("effective_defaults")) != canonical(
        spec["unchanged_effective_defaults"]
    ):
        raise ValueError("unchanged actual GAE effective defaults required")
    if canonical(record.get("coupled_adam")) != canonical(
        dict(
            optimizer_class="Adam",
            defaults_decoupled_weight_decay=False,
            group_decoupled_weight_decay=[False],
        )
    ):
        raise ValueError("actual native coupled Adam required")
    h = record.get("optimizer_sha256", "")
    if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise ValueError("actual optimizer fingerprint required")


def validate_record(record, arm, smoke=False, *, initial=False, selected_steps=None, seed=11):
    proof = initialization_spec(seed, smoke)
    _base_validate(record, arm, smoke, proof, initial, selected_steps)
    if type(record.get("seed")) is not int or record["seed"] != seed:
        raise ValueError("actual GAE model seed required")
    if initial and record.get("rng") != dict(
        torch=proof["rng_sha256"], **seeded_fingerprints(seed)
    ):
        raise ValueError("actual original Torch/NumPy/Python GAE starts required")
    return record


def apply_initialization(model, arm, seed=11, smoke=False):
    parameters(arm, smoke)
    entry = initialization_entry(seed, smoke)
    proof = initialization_spec(seed, smoke)
    if (
        type(model.seed) is not int
        or model.seed != seed
        or seeded_fingerprints() != seeded_fingerprints(seed)
    ):
        raise ValueError("actual GAE seed and native NumPy/Python starts required")
    # Check fresh geometry/settings/empty Adam against its current tensors before
    # loading pinned tensors. This local validation scaffold is never published
    # as shared-start evidence or attached to the model.
    live = model_record(model)
    fresh = dict(proof, policy_tensor_sha256=live["policy_tensor_sha256"])
    live["shared_initialization"] = fresh
    _base_validate(live, arm, smoke, fresh, True, None)
    loaded = load_initialization(model, ROOT / entry["file"], entry)
    if any(loaded[k] != proof[k] for k in loaded):
        raise ValueError("actual GAE source restoration mismatch")
    model._research_initialization = proof
    validate_record(model_record(model), arm, smoke, initial=True, seed=seed)
    return proof


def validate_initial_records(records, smoke=False, seed=11):
    if set(records) != set(protocol()["arms"]):
        raise ValueError("both actual GAE initial records required")
    for arm, record in records.items():
        validate_record(record, arm, smoke, initial=True, seed=seed)
    left, right = (
        {k: v for k, v in records[a].items() if k != "gae_lambda"} for a in protocol()["arms"]
    )
    if canonical(left) != canonical(right):
        raise ValueError("only GAE lambda may differ at shared start")
    return dict(
        common_initial_tensors_and_empty_optimizers_exact=True,
        sampling_rng_identical=True,
        only_gae_lambda_differs=True,
    )
