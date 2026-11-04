"""Independent offline acceptance of trusted, retrieved TRPO smoke artifacts.

Native ZIP loading is restricted to project-owned artifacts; this is not an
untrusted-checkpoint sandbox. No environment collection or optimization occurs.
"""

import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch
from stable_baselines3 import PPO
from sb3_contrib import TRPO

from rl_stocks.evaluation.offpolicy_fit import fingerprint
from rl_stocks.evaluation.trpo_updates import verify_updates
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.canonical_actor import encode32
from rl_stocks.inference.canonical_tanh_export import export_policy
from rl_stocks.inference.canonical_tanh_policy import CanonicalTanhPolicy
from rl_stocks.training.sac_update_worker import preserved_rng
from rl_stocks.training.trpo_audit import actor_parameters, clone_actor
from rl_stocks.training.trpo_feasibility import protocol as feasibility
from rl_stocks.training.trpo_smoke import protocol

FILES = {
    "actor-updates.pt": "actor_updates_sha256",
    "initial.zip": "initial_model_sha256",
    "final.zip": "final_model_sha256",
    "reference-initial.json": "reference_initial_sha256",
    "reference-final.json": "reference_final_sha256",
}


def verify_optimizer(model, *, algorithm, steps):
    named = dict(model.policy.named_parameters())
    actor = actor_parameters(model.policy)
    required = named if algorithm == "ppo" else {k: v for k, v in named.items() if k not in actor}
    state = model.policy.optimizer.state
    if set(state) != set(required.values()):
        raise ValueError("restored Adam ownership mismatch")
    for parameter, moments in state.items():
        if (
            set(moments) != {"step", "exp_avg", "exp_avg_sq"}
            or moments["step"].numel() != 1
            or float(moments["step"].item()) != steps
            or any(
                moments[k].shape != parameter.shape
                or moments[k].dtype != torch.float32
                or not torch.isfinite(moments[k]).all()
                for k in ("exp_avg", "exp_avg_sq")
            )
            or (moments["exp_avg_sq"] < 0).any()
        ):
            raise ValueError("restored finite exact Adam moments required")
    return len(required)


def verify_native(model, case, *, initial):
    algorithm = case["algorithm"]
    spec = feasibility()
    parameters = dict(spec["shared_parameters"])
    parameters.update(spec["ppo_parameters" if algorithm == "ppo" else "trpo_parameters"])
    parameters.update(n_steps=case["rollout"], batch_size=case["batch"])
    if (
        type(model) is not (PPO if algorithm == "ppo" else TRPO)
        or model.seed != 11
        or model.n_envs != 1
        or str(model.device) != "cpu"
        or any((getattr(model, k)(1.0) if k == "clip_range" else getattr(model, k)) != v
               for k, v in parameters.items())
        or model.num_timesteps != (0 if initial else case["steps"])
        or model._n_updates != (0 if initial else case["nominal_updates"] * (
            10 if algorithm == "ppo" else 1
        ))
        or any(t.dtype != torch.float32 or not torch.isfinite(t).all()
               for t in model.policy.state_dict().values())
    ):
        raise ValueError("restored native configuration/counters/state mismatch")
    if initial:
        if model.policy.optimizer.state:
            raise ValueError("initial optimizer must be empty")
    else:
        verify_optimizer(model, algorithm=algorithm, steps=case["adam_steps"])


@torch.no_grad()
def verify_case(folder, case):
    folder = Path(folder)
    record = json.loads((folder / "result.json").read_text())
    if record["case"] != case or record["performance_evaluated"] is not False:
        raise ValueError("fixed nonperformance case required")
    for name, key in FILES.items():
        path = folder / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("bounded regular owned artifact required")
        if checksum(path) != record[key]:
            raise ValueError("retrieved artifact checksum mismatch")
    before_hashes = {name: checksum(folder / name) for name in FILES}
    algorithm = case["algorithm"]
    cls = PPO if algorithm == "ppo" else TRPO
    with preserved_rng():
        initial = cls.load(folder / "initial.zip", device="cpu")
        final = cls.load(folder / "final.zip", device="cpu")
        verify_native(initial, case, initial=True)
        verify_native(final, case, initial=False)
        states = torch.load(folder / "actor-updates.pt", map_location="cpu", weights_only=True)
        proof = verify_updates(states, record["audit"]["records"], algorithm=algorithm,
                               rollout=case["rollout"])
        if (
            len(states) != case["nominal_updates"]
            or fingerprint(clone_actor(initial.policy)) != fingerprint(states[0]["before"])
            or fingerprint(clone_actor(final.policy)) != fingerprint(states[-1]["after"])
        ):
            raise ValueError("saved update endpoints mismatch native checkpoints")
        for phase in ("initial", "final"):
            with tempfile.TemporaryDirectory() as tmp:
                restored = export_policy(folder / f"{phase}.zip", Path(tmp) / "actor.json",
                                         algorithm=algorithm)
                if checksum(restored) != checksum(folder / f"reference-{phase}.json"):
                    raise ValueError("reference actor does not restore native checkpoint")
        obs = states[-1]["observations"]
        x = obs[np.linspace(0, len(obs) - 1, 8, dtype=int)].numpy().astype(np.float32)
        if [[encode32(v) for v in row] for row in x] != record["reference_inputs_hex"]:
            raise ValueError("reference probe must use saved rollout observations")
        reference = CanonicalTanhPolicy.load(folder / "reference-final.json",
                                            expected_source=checksum(folder / "final.zip"),
                                            expected_algorithm=algorithm)
        outputs = [[encode32(v) for v in reference.predict(row.tolist())] for row in x]
        if outputs != record["reference_output_hex"]:
            raise ValueError("exact frozen reference output mismatch")
        original = fingerprint((final.policy.state_dict(), final.policy.optimizer.state_dict()))
        a = final.predict(x, deterministic=True)[0]
        np.testing.assert_array_equal(a, final.predict(x, deterministic=True)[0])
        if original != fingerprint((final.policy.state_dict(), final.policy.optimizer.state_dict())):
            raise ValueError("inference changed restored native state")
    audit = record["audit"]
    counts = [row["adam_steps"] for row in audit["records"]]
    per_update = case["adam_steps"] // case["nominal_updates"]
    if (
        counts != [per_update * i for i in range(1, case["nominal_updates"] + 1)]
        or any(r["adam_steps_in_update"] != per_update for r in audit["records"])
        or audit["actual_adam_steps"] != case["adam_steps"]
        or audit["nominal_policy_updates"] != case["nominal_updates"]
        or audit["native_n_updates"] != final._n_updates
        or record["sampling"]["mode"] != "stock_etf_cycle_windows"
        or record["sampling"]["transitions"] != case["steps"]
        or record["backend"]["selected_device"] != "cpu"
        or record["backend"]["torch_intraop_threads"] != 1
        or record["normalization_unchanged"] is not True
        or record["native_save_load_exact"] is not True
        or not math.isfinite(record["learn_seconds"])
        or record["learn_seconds"] <= 0
    ):
        raise ValueError("actual update/sampling/backend receipt mismatch")
    hvps = [row["hvp_calls_in_update"] for row in audit["records"]]
    accepted = sum(row.get("line_search_accepted", False) for row in audit["records"])
    if (
        audit["actual_hvp_calls"] != sum(hvps)
        or audit["accepted_actor_line_searches"] != accepted
        or algorithm == "trpo" and (not accepted or any(not 2 <= x <= 17 for x in hvps))
        or algorithm == "ppo" and (any(hvps) or accepted)
    ):
        raise ValueError("actual native search counters mismatch")
    if before_hashes != {name: checksum(folder / name) for name in FILES}:
        raise ValueError("offline acceptance mutated retrieved artifacts")
    return {"case": case, "independent_updates": proof, "artifacts_unchanged": True,
            "initial_policy_sha256": fingerprint(initial.policy.state_dict()),
            "normalization_verified_separately": False, "performance_evaluated": False}


def verify_matrix(folder):
    """Four cases only; root provenance/dataset admission remains controller-owned."""
    spec = protocol()
    rows = [verify_case(Path(folder) / f'{c["mode"]}-{c["algorithm"]}', c)
            for c in spec["cases"]]
    for offset in (0, 2):
        if rows[offset]["initial_policy_sha256"] != rows[offset + 1]["initial_policy_sha256"]:
            raise ValueError("full PPO/TRPO initial pair differs")
    return {"schema": "trpo-smoke-matrix-acceptance-v1", "cases": rows,
            "training_performed": False, "performance_promotion": False}
