"""Pinned untrained21/23-input starts; actual Treasury model/Adam/RNG evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rl_stocks.core.contracts import canonical
from rl_stocks.reporting.optimizer import optimizer_fingerprint
from rl_stocks.training.recurrent_record import (
    model_record as base_record,
    validate_record as base_validate,
)
from rl_stocks.training.shared_initialization import (
    load_initialization,
    read_initialization,
)
from rl_stocks.training.treasury_spec import ROOT, BINDING_SHA256, binding, parameters, protocol
from rl_stocks.training.vix_initialization import seeded_fingerprints, validate_mapping


def portable_path(name):
    path = Path(name)
    resolved = (ROOT / path).resolve()
    if (
        path.is_absolute()
        or ".." in path.parts
        or not resolved.is_relative_to((ROOT / "configs/initializations").resolve())
    ):
        raise ValueError("portable tracked Treasury initializer required")
    return resolved


def _manifest(name, expected):
    path = portable_path(name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Treasury untrained source manifest changed")
    value = json.loads(path.read_text())
    if (
        value.get("generated_without_fitting") is not True
        or value.get("no_dataset_or_provider_access") is not True
    ):
        raise ValueError("unfitted source without dataset/provider access required")
    for name, expected_sha in value.get("implementation_sha256", {}).items():
        relative = Path(name)
        implementation = (ROOT / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not implementation.is_relative_to(ROOT.resolve())
        ):
            raise ValueError("portable initializer implementation source required")
        if hashlib.sha256(implementation.read_bytes()).hexdigest() != expected_sha:
            raise ValueError("original untrained initializer generation source changed")
    return value


def initialization_entry(arm, seed=11, smoke=False):
    spec = protocol()
    shared = spec["shared_initialization"]
    if (
        arm not in spec["arms"]
        or type(seed) is not int
        or seed not in spec["confirmation_seeds"]
        or type(smoke) is not bool
        or smoke
        and seed != 11
    ):
        raise ValueError("fixed Treasury arm/seed/phase required")
    if arm == "legacy":
        manifest = _manifest(shared["source_manifest"], shared["source_manifest_sha256"])
        entry = manifest["artifacts"][f"control-screen-{seed}"]
    else:
        manifest = _manifest(shared["augmented_manifest"], shared["augmented_manifest_sha256"])
        entry = manifest["artifacts"][f"full-screen-{seed}"]
    if (
        type(entry["seed"]) is not int
        or entry["seed"] != seed
        or entry["smoke"] is not False
        or entry["torch_version"] != torch.__version__.split("+")[0]
    ):
        raise ValueError("full64 source phase/seed/Torch identity required")
    portable_path(entry["file"])
    return entry


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


def validate_assets():
    proofs = {}
    for seed in protocol()["confirmation_seeds"]:
        old = initialization_entry("legacy", seed)
        new = initialization_entry("full", seed)
        original, rng = read_initialization(portable_path(old["file"]).read_bytes(), old)
        augmented, new_rng = read_initialization(portable_path(new["file"]).read_bytes(), new)
        if not np.array_equal(rng, new_rng) or initialization_entry(
            "full", seed
        ) != initialization_entry("masked", seed):
            raise ValueError("original CPU RNG and shared augmented source required")
        torch.Generator(device="cpu").set_state(torch.from_numpy(rng))
        mapped = validate_mapping(original, augmented)
        proofs[str(seed)] = dict(
            common_tensors_exact=mapped["common_tensors_exact"],
            original_columns_exact=mapped["original_market_and_account_columns_exact"],
            new_columns_zero=mapped["new_vix_columns_zero"],
            parameter_counts=mapped["parameter_counts"],
            zero_columns_sha256=mapped["zero_columns_sha256"],
        )
        if mapped["parameter_counts"] != protocol()["shared_initialization"]["parameter_counts"]:
            raise ValueError("Treasury source geometry mismatch")
    return dict(
        passed=True,
        mapping_proofs=proofs,
        seeds=[11, 23, 37],
        full_masked_tensors_exact=True,
        original_cpu_rng_exact=True,
        smoke_reuses_full64=True,
        models_constructed=False,
        optimizer_steps=0,
    )


def model_record(model):
    result = base_record(model)
    optimizer = model.policy.optimizer
    coupling = dict(
        optimizer_class=type(optimizer).__name__,
        defaults_decoupled_weight_decay=optimizer.defaults.get("decoupled_weight_decay"),
        group_decoupled_weight_decay=[
            g.get("decoupled_weight_decay") for g in optimizer.param_groups
        ],
        group_amsgrad=[g.get("amsgrad") for g in optimizer.param_groups],
    )
    defaults = {
        k: getattr(model, k) for k in ["normalize_advantage", "vf_coef", "max_grad_norm", "use_sde"]
    }
    defaults["clip_range_vf"] = (
        None if model.clip_range_vf is None else float(model.clip_range_vf(1.0))
    )
    defaults.update(
        optimizer="coupled Adam"
        if coupling["optimizer_class"] == "Adam"
        and coupling["defaults_decoupled_weight_decay"] is False
        else coupling["optimizer_class"],
        optimizer_betas=list(optimizer.param_groups[0]["betas"]),
        optimizer_amsgrad=optimizer.defaults.get("amsgrad"),
    )
    result.update(
        seed=model.seed,
        optimizer_sha256=optimizer_fingerprint(model),
        rng=dict(
            torch=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            **seeded_fingerprints(),
        ),
        effective_defaults=defaults,
        coupled_adam=coupling,
    )
    return result


def _validate(record, arm, smoke, proof, initial, selected_steps):
    spec = protocol()
    params = parameters(arm, smoke)
    for key in [
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "ent_coef",
        "learning_rate",
    ]:
        if canonical(record.get(key)) != canonical(params[key]):
            raise ValueError("typed actual Treasury algorithm parameters required")
    base_validate(
        record,
        params,
        len(binding()["dataset_contracts"][arm]["names"]) + 14,
        spec["smoke_steps" if smoke else "steps_per_cell"],
        proof,
        initial=initial,
        selected_steps=selected_steps,
        weight_decay=0.0,
    )
    if canonical(record.get("effective_defaults")) != canonical(spec["effective_defaults"]):
        raise ValueError("fixed actual Treasury effective defaults required")
    expected = dict(
        optimizer_class="Adam",
        defaults_decoupled_weight_decay=False,
        group_decoupled_weight_decay=[False],
        group_amsgrad=[False],
    )
    if canonical(record.get("coupled_adam")) != canonical(expected):
        raise ValueError("actual Treasury coupled Adam semantics required")
    h = record.get("optimizer_sha256", "")
    if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise ValueError("actual optimizer fingerprint required")


def validate_record(record, arm, smoke=False, *, initial=False, selected_steps=None, seed=11):
    proof = initialization_spec(arm, seed, smoke)
    _validate(record, arm, smoke, proof, initial, selected_steps)
    if type(record.get("seed")) is not int or record["seed"] != seed:
        raise ValueError("actual Treasury model seed required")
    if initial and record.get("rng") != dict(
        torch=proof["rng_sha256"], **seeded_fingerprints(seed)
    ):
        raise ValueError("actual original Torch/NumPy/Python starts required")
    if not initial and record["policy_tensor_sha256"] == proof["policy_tensor_sha256"]:
        raise ValueError("learned Treasury tensors must differ from untrained source")
    return record


def apply_initialization(model, arm, seed=11, smoke=False):
    entry = initialization_entry(arm, seed, smoke)
    proof = initialization_spec(arm, seed, smoke)
    if (
        type(model.seed) is not int
        or model.seed != seed
        or seeded_fingerprints() != seeded_fingerprints(seed)
    ):
        raise ValueError("native Treasury seeded streams required")
    live = model_record(model)
    fresh = dict(proof, policy_tensor_sha256=live["policy_tensor_sha256"])
    live["shared_initialization"] = fresh
    _validate(live, arm, smoke, fresh, True, None)
    loaded = load_initialization(model, portable_path(entry["file"]), entry)
    if any(loaded[k] != proof[k] for k in loaded):
        raise ValueError("Treasury untrained source restoration mismatch")
    model._research_initialization = proof
    validate_record(model_record(model), arm, smoke, initial=True, seed=seed)
    return proof


def validate_initial_records(records, smoke=False, seed=11):
    if set(records) != set(protocol()["arms"]):
        raise ValueError("all three Treasury initial records required")
    validate_assets()
    for arm, record in records.items():
        validate_record(record, arm, smoke, initial=True, seed=seed)
    if (
        canonical(records["full"]) != canonical(records["masked"])
        or len({canonical(r["rng"]) for r in records.values()}) != 1
    ):
        raise ValueError("exact augmented tensors/empty Adam and original shared RNG required")
    return dict(
        full_masked_records_exact=True,
        sampling_rng_identical=True,
        common_columns_and_zero_embedding_verified=True,
    )
