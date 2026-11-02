"""Native routed learner with a read-only, RNG-preserving common-start preflight."""

import json
from pathlib import Path

import torch
from stable_baselines3 import SAC

from rl_stocks.evaluation.offpolicy_fit import fingerprint
from rl_stocks.training.recurrent_sac_learner import require_colab_cpu
from rl_stocks.training.sac_update_worker import preserved_rng
from rl_stocks.training.task_reference_assets import check_model
from rl_stocks.training.task_reference_runtime import checksum, execution_spec


def check_initial_parity(routed, shared):
    check_model(routed, role="tagged_routed", steps=0)
    check_model(shared, role="masked_shared", steps=0)
    expected = shared.policy.state_dict()
    seen = set()
    actual = routed.policy.state_dict()
    for name, value in actual.items():
        mapped = name.replace(".stock.", ".").replace(".etf.", ".")
        if mapped not in expected or not torch.equal(value, expected[mapped]):
            raise ValueError("routed initial array differs from retained shared start")
        seen.add(mapped)
    if seen != set(expected) or not torch.equal(routed.log_ent_coef, shared.log_ent_coef):
        raise ValueError("complete common initialization required")
    return {
        "schema": "task-routed-initial-parity-v1",
        "routed_arrays": len(actual),
        "shared_arrays": len(expected),
        "shared_policy_sha256": fingerprint(expected),
        "routed_policy_sha256": fingerprint(actual),
        "all_mapped_arrays_exact": True,
    }


def execute_reference_candidate_colab(dataset, evidence, output):
    require_colab_cpu()  # Before path/model/RNG mutation or any optimization.
    from rl_stocks.training import task_screen_worker as worker

    spec = execution_spec(Path.cwd())
    initial = Path.cwd() / "initial-control.zip"
    if checksum(initial) != spec["initial_control_zip_sha256"]:
        raise ValueError("trusted common-initialization archive changed")
    original = worker.create_task_sac
    proof = None

    def guarded_factory(*args, **kwargs):
        nonlocal proof
        model = original(*args, **kwargs)
        with preserved_rng(), torch.no_grad():
            shared = SAC.load(initial, device="cpu")
            if fingerprint(shared.policy.state_dict()) != spec["initial_shared_policy_sha256"]:
                raise ValueError("retained shared initialization changed")
            proof = check_initial_parity(model, shared)
        return model

    worker.create_task_sac = guarded_factory
    try:
        worker.execute_screen_colab(dataset, evidence, output, task_role="tagged_routed")
    finally:
        worker.create_task_sac = original
    if proof is None:
        raise ValueError("common-start guard never executed")
    # Also persist in completion metadata at the remote entry point, not an
    # unregistered extra file inside the native training artifact directory.
    return json.loads(json.dumps(proof))
