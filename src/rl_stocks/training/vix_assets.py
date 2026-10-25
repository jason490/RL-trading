"""Checksummed portable VIX data/initializer bindings; no model creation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rl_stocks.core.contracts import digest
from rl_stocks.training.shared_initialization import read_initialization, tensor_fingerprint
from rl_stocks.training.vix_initialization import (
    source_initialization,
    validate_mapping,
    seeded_fingerprints,
)
from rl_stocks.training.vix_spec import PREREGISTRATION_SHA256, preregistration

ROOT = Path(__file__).resolve().parents[3]
BINDING_SHA256 = "5ee939d3c96a100365ad5bf9f19bfa656be763bf432e9e0ba511f09bc0e37f85"


def binding():
    value = json.loads((ROOT / "configs/experiments/vix_context_binding.json").read_text())
    if digest(value) != BINDING_SHA256 or value["preregistration_sha256"] != PREREGISTRATION_SHA256:
        raise ValueError("fixed VIX data/initializer binding changed")
    return value


def _portable_path(name):
    value = Path(name)
    path = (ROOT / value).resolve()
    if (
        value.is_absolute()
        or ".." in value.parts
        or not path.is_relative_to((ROOT / "configs/initializations").resolve())
    ):
        raise ValueError("portable tracked VIX initializer path required")
    return path


def initialization_manifest():
    spec = binding()
    path = _portable_path(spec["initialization_manifest"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["initialization_manifest_sha256"]:
        raise ValueError("VIX initializer manifest checksum mismatch")
    result = json.loads(path.read_text())
    if (
        type(result.get("version")) is not int
        or result["version"] != 1
        or result.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or result.get("generated_without_fitting") is not True
        or result.get("no_dataset_or_provider_access") is not True
        or result.get("original_assets_unchanged") is not True
    ):
        raise ValueError("fixed untrained VIX initializer provenance required")
    for name, sha in result["implementation_sha256"].items():
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha
        ):
            raise ValueError("VIX initializer generation implementation changed")
    return result


def initialization_entry(arm, seed=11, smoke=False):
    spec = preregistration()
    if (
        arm not in spec["arms"]
        or type(seed) is not int
        or seed not in spec["confirmation_seeds"]
        or type(smoke) is not bool
        or smoke
        and seed != 11
    ):
        raise ValueError("fixed VIX arm/seed/phase required")
    key = arm + "-" + ("smoke" if smoke else "screen") + "-" + str(seed)
    entry = initialization_manifest()["artifacts"][key]
    if entry["seed"] != seed or type(entry["seed"]) is not int or entry["smoke"] is not smoke:
        raise ValueError("actual VIX initializer seed/phase mismatch")
    if entry["torch_version"] != torch.__version__.split("+")[0]:
        raise ValueError("original VIX Torch initialization version required")
    _portable_path(entry["file"])
    return entry


def validate_assets():
    spec = preregistration()
    manifest = initialization_manifest()
    proofs = {}
    expected = {a + "-screen-" + str(s) for a in spec["arms"] for s in spec["confirmation_seeds"]}
    expected |= {a + "-smoke-11" for a in spec["arms"]}
    if set(manifest["artifacts"]) != expected:
        raise ValueError("exact12 fixed VIX initializer entries required")
    for seed in spec["confirmation_seeds"]:
        _, original, original_rng = source_initialization(seed)
        fingerprints = {}
        for arm in spec["arms"]:
            entry = initialization_entry(arm, seed)
            arrays, rng = read_initialization(_portable_path(entry["file"]).read_bytes(), entry)
            if not np.array_equal(original_rng, rng):
                raise ValueError("original CPU VIX sampling RNG changed")
            if arm == "legacy":
                if tensor_fingerprint(arrays) != tensor_fingerprint(original):
                    raise ValueError("original legacy tensors changed")
            else:
                proofs[str(seed)] = validate_mapping(original, arrays)
            fingerprints[arm] = entry["policy_tensor_sha256"]
            if seed == 11:
                smoke = initialization_entry(arm, seed, True)
                if {k: v for k, v in smoke.items() if k != "smoke"} != {
                    k: v for k, v in entry.items() if k != "smoke"
                }:
                    raise ValueError("smoke must reuse full64 VIX initializer")
        if fingerprints["full"] != fingerprints["masked"] or manifest["seeded_numpy_python"][
            str(seed)
        ] != seeded_fingerprints(seed):
            raise ValueError("full/masked tensors or seeded VIX streams mismatch")
        if proofs[str(seed)] != manifest["mapping_proofs"][str(seed)]:
            raise ValueError("actual VIX embedding proof mismatch")
    return dict(
        passed=True,
        seeds=[11, 23, 37],
        artifact_entries=12,
        full_masked_tensors_exact=True,
        original_common_tensors_and_cpu_rng_exact=True,
        new_columns_zero=True,
        seeded_numpy_python_verified=True,
        mapping_proofs=proofs,
        models_constructed=False,
        optimizer_steps=0,
    )
