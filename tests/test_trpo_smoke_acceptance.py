import json

import pytest
import torch

from test_trpo_feasibility import ContractEnv
from rl_stocks.core.contracts import canonical
from rl_stocks.evaluation.trpo_smoke import verify_native, verify_optimizer, verify_case, FILES
from rl_stocks.inference.bundle import checksum
from rl_stocks.training.trpo_audit import actor_parameters
from rl_stocks.training.trpo_feasibility import create
from rl_stocks.training.trpo_smoke import protocol


@pytest.mark.parametrize("algorithm", ["ppo", "trpo"])
def test_native_initial_restore_and_counter_rejection(tmp_path, algorithm):
    from stable_baselines3 import PPO
    from sb3_contrib import TRPO
    case = next(c for c in protocol()["cases"] if c["algorithm"] == algorithm)
    model = create(algorithm, ContractEnv(), smoke=True)
    model.save(tmp_path / "initial.zip")
    restored = (PPO if algorithm == "ppo" else TRPO).load(tmp_path / "initial.zip")
    verify_native(restored, case, initial=True)
    with pytest.raises(ValueError, match="counters"):
        verify_native(restored, case, initial=False)
    restored.num_timesteps = 128
    with pytest.raises(ValueError, match="counters"):
        verify_native(restored, case, initial=True)


def populate(model, algorithm, steps):
    actor = actor_parameters(model.policy)
    required = {k: v for k, v in model.policy.named_parameters()
                if algorithm == "ppo" or k not in actor}
    # Explicit fixture moments; no backward, optimizer.step, or learning.
    for p in required.values():
        model.policy.optimizer.state[p] = {
            "step": torch.tensor(float(steps)),
            "exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.ones_like(p),
        }
    return required


@pytest.mark.parametrize("algorithm", ["ppo", "trpo"])
def test_restored_optimizer_ownership_and_moments(algorithm):
    model = create(algorithm, ContractEnv(), smoke=True)
    required = populate(model, algorithm, 320)
    assert verify_optimizer(model, algorithm=algorithm, steps=320) == len(required)
    state = next(iter(model.policy.optimizer.state.values()))
    state["step"] = torch.tensor(320.5)
    with pytest.raises(ValueError, match="moments"):
        verify_optimizer(model, algorithm=algorithm, steps=320)
    state["step"] = torch.tensor(320.0)
    state["exp_avg_sq"].fill_(-1)
    with pytest.raises(ValueError, match="moments"):
        verify_optimizer(model, algorithm=algorithm, steps=320)


def test_trpo_rejects_actor_adam_state():
    model = create("trpo", ContractEnv(), smoke=True)
    populate(model, "ppo", 320)
    with pytest.raises(ValueError, match="ownership"):
        verify_optimizer(model, algorithm="trpo", steps=320)


def test_artifact_checksum_rejected_before_native_load(tmp_path, monkeypatch):
    from stable_baselines3 import PPO
    case = protocol()["cases"][0]
    record = {"case": case, "performance_evaluated": False}
    for filename, key in FILES.items():
        (tmp_path / filename).write_text("fixture")
        record[key] = checksum(tmp_path / filename)
    (tmp_path / "final.zip").write_text("tampered")
    (tmp_path / "result.json").write_bytes(canonical(record))
    def forbidden(*a, **kw):
        raise AssertionError("checksum must precede native ZIP loading")
    monkeypatch.setattr(PPO, "load", forbidden)
    with pytest.raises(ValueError, match="checksum"):
        verify_case(tmp_path, case)
    record["performance_evaluated"] = True
    (tmp_path / "result.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="nonperformance"):
        verify_case(tmp_path, case)


def test_complete_synthetic_ppo_artifact_restoration(tmp_path):
    import numpy as np
    from rl_stocks.evaluation.offpolicy_fit import fingerprint
    from rl_stocks.inference.canonical_actor import encode32
    from rl_stocks.inference.canonical_tanh_export import export_policy
    from rl_stocks.inference.canonical_tanh_policy import CanonicalTanhPolicy
    from rl_stocks.training.trpo_audit import clone_actor

    case = protocol()["cases"][0]
    model = create("ppo", ContractEnv(), smoke=True)
    model.save(tmp_path / "initial.zip")
    actor = clone_actor(model.policy)
    obs = torch.zeros((128, 21))
    states = [{"before": actor, "after": actor, "observations": obs} for _ in range(8)]
    rows = [{"transition": 128 * i, "adam_steps": 40 * i, "adam_steps_in_update": 40,
             "hvp_calls_in_update": 0, "actor_changed": False,
             "actor_before_sha256": fingerprint(actor),
             "actor_after_sha256": fingerprint(actor)} for i in range(1, 9)]
    # A synthetic serialized fixture only; this does not prove actual learning.
    populate(model, "ppo", 320)
    model.num_timesteps, model._n_updates = 1024, 80
    model.save(tmp_path / "final.zip")
    torch.save(states, tmp_path / "actor-updates.pt")
    for phase in ("initial", "final"):
        export_policy(tmp_path / f"{phase}.zip", tmp_path / f"reference-{phase}.json",
                      algorithm="ppo")
    reference = CanonicalTanhPolicy.load(tmp_path / "reference-final.json")
    outputs = [[encode32(v) for v in reference.predict(np.zeros(21).tolist())]] * 8
    record = {
        "case": case, "performance_evaluated": False,
        "audit": {"records": rows, "actual_adam_steps": 320, "nominal_policy_updates": 8,
                  "native_n_updates": 80, "actual_hvp_calls": 0,
                  "accepted_actor_line_searches": 0},
        "sampling": {"mode": "stock_etf_cycle_windows", "transitions": 1024},
        "backend": {"selected_device": "cpu", "torch_intraop_threads": 1},
        "normalization_unchanged": True, "native_save_load_exact": True,
        "learn_seconds": 1.0, "reference_inputs_hex": [[encode32(0)] * 21] * 8,
        "reference_output_hex": outputs,
    }
    record.update({key: checksum(tmp_path / filename) for filename, key in FILES.items()})
    (tmp_path / "result.json").write_bytes(canonical(record))
    proof = verify_case(tmp_path, case)
    assert proof["artifacts_unchanged"] and not proof["performance_evaluated"]
    record["reference_output_hex"][0][0] = encode32(0.5)
    (tmp_path / "result.json").write_bytes(canonical(record))
    with pytest.raises(ValueError, match="reference output"):
        verify_case(tmp_path, case)
