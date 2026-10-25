"""Untrained23-input embedding with exact original recurrent tensors and zero VIX columns."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from rl_stocks.core.contracts import digest
from rl_stocks.training.shared_initialization import read_initialization, tensor_fingerprint
from rl_stocks.training.vix_spec import preregistration

ROOT = Path(__file__).resolve().parents[3]
INPUT_KEYS = ("lstm_actor.weight_ih_l0", "lstm_critic.weight_ih_l0")


def source_initialization(seed):
    spec = preregistration()["shared_initialization"]
    if type(seed) is not int or seed not in spec["seeds"]:
        raise ValueError("fixed VIX initializer seed required")
    path = ROOT / spec["source_manifest"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["source_manifest_sha256"]:
        raise ValueError("pinned original recurrent manifest required")
    original = json.loads(path.read_text())["artifacts"][f"control-screen-{seed}"]
    arrays, rng = read_initialization((ROOT / original["file"]).read_bytes(), original)
    return original, arrays, rng


def validate_mapping(original, augmented):
    if not original or set(original) != set(augmented) or set(INPUT_KEYS) - set(original):
        raise ValueError("exact common recurrent tensor keys required")
    for key, old in original.items():
        new = augmented[key]
        if old.dtype != new.dtype or not np.isfinite(old).all() or not np.isfinite(new).all():
            raise ValueError("finite matching tensor dtypes required")
        if key in INPUT_KEYS:
            if (
                old.shape != (256, 21)
                or new.shape != (256, 23)
                or not np.array_equal(old[:, :7], new[:, :7])
                or not np.array_equal(old[:, 7:], new[:, 9:])
                or np.any(new[:, 7:9] != 0)
            ):
                raise ValueError("exact original input columns and zero new VIX columns required")
        elif not np.array_equal(old, new):
            raise ValueError("every common policy tensor must stay exact")
    counts = dict(
        legacy=sum(a.size for a in original.values()),
        augmented=sum(a.size for a in augmented.values()),
    )
    if counts != preregistration()["shared_initialization"]["parameter_counts"]:
        raise ValueError("fixed61,513/62,537 recurrent parameter counts required")
    return dict(
        common_tensors_exact=True,
        original_market_and_account_columns_exact=True,
        new_vix_columns_zero=True,
        parameter_counts=counts,
        zero_columns_sha256=tensor_fingerprint({k: augmented[k][:, 7:9] for k in INPUT_KEYS}),
    )


def map_initial_tensors(model, original):
    optimizer = getattr(model.policy, "optimizer", None)
    if (
        model.num_timesteps != 0
        or model._n_updates != 0
        or optimizer is None
        or optimizer.state
        or getattr(model, "_research_initialization", None) is not None
        or getattr(model, "_vix_mapping_applied", False)
        or any(p.device.type != "cpu" for p in model.policy.parameters())
    ):
        raise ValueError("fresh CPU recurrent model, empty optimizer and one mapping required")
    current = {k: v.detach().cpu().numpy().copy() for k, v in model.policy.state_dict().items()}
    if set(current) != set(original):
        raise ValueError("same recurrent policy tensor keys required")
    mapped = {}
    for key, new in current.items():
        old = original[key]
        if old.dtype != new.dtype:
            raise ValueError("same recurrent tensor dtype required")
        if key in INPUT_KEYS:
            if old.shape != (256, 21) or new.shape != (256, 23):
                raise ValueError("fixed21-to23 input embedding required")
            new[:, :7] = old[:, :7]
            new[:, 7:9] = 0.0
            new[:, 9:] = old[:, 7:]
            mapped[key] = new
        else:
            if old.shape != new.shape:
                raise ValueError("common recurrent tensor shapes required")
            mapped[key] = old.copy()
    proof = validate_mapping(original, mapped)
    model.policy.load_state_dict({k: torch.from_numpy(v) for k, v in mapped.items()}, strict=True)
    validate_mapping(
        original, {k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()}
    )
    model._vix_mapping_applied = True
    return proof


def seeded_fingerprints(seed=None):
    """Private seeded probes never mutate global NumPy/Python streams."""
    n = np.random.get_state() if seed is None else np.random.RandomState(seed).get_state()
    p = random.getstate() if seed is None else random.Random(seed).getstate()
    return dict(
        numpy=digest(
            dict(
                kind=n[0],
                keys=n[1].tolist(),
                position=int(n[2]),
                has_gauss=int(n[3]),
                cached_gaussian=float(n[4]),
            )
        ),
        python=digest(p),
    )
