from copy import deepcopy
from types import SimpleNamespace
import gzip
import io
import json
import tarfile

import pytest
import torch

from test_task_reference_assets import initial_model
from rl_stocks.evaluation.canonical_diagnostic import inference_only
from rl_stocks.runtime.colab.jobs import task_reference_candidate_plan, validate_plan
from rl_stocks.runtime.colab.task_reference_snapshots import retain
from rl_stocks.training.task_reference_candidate_acceptance import reference_fitted_gate
from rl_stocks.training.task_reference_runtime import CLAIM, REFERENCE_SHA256, claim_candidate
from rl_stocks.training.task_reference_worker import (
    check_initial_parity,
    execute_reference_candidate_colab,
)


def test_fixed_job_cannot_change_seed_role_steps_hardware_or_deadlines():
    plan = task_reference_candidate_plan()
    validate_plan(plan)
    for key, value in [
        ("accelerator", "A100"),
        ("allocation_deadline_seconds", 9000),
        ("teardown_reserve_seconds", 0),
        ("reference_execution_sha256", "a" * 64),
    ]:
        bad = deepcopy(plan)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_plan(bad)
    for role, steps, seed in [
        ("tagged_shared", 100000, 11),
        ("tagged_routed", 200000, 11),
        ("tagged_routed", 100000, 23),
    ]:
        bad = deepcopy(plan)
        bad["task_screen"]["task_role"] = role
        bad["jobs"][0].update(steps=steps, seed=seed)
        with pytest.raises(ValueError):
            validate_plan(bad)


def test_actual_routed_common_start_is_exact_and_readonly():
    with inference_only():
        shared = initial_model("masked_shared")
        routed = initial_model("tagged_routed")
        rng = torch.get_rng_state().clone()
        proof = check_initial_parity(routed, shared)
        assert (
            proof["all_mapped_arrays_exact"]
            and proof["routed_arrays"] == 44
            and proof["shared_arrays"] == 32
        )
        assert torch.equal(torch.get_rng_state(), rng)
        with torch.no_grad():
            routed.actor.mu.etf.bias[0] += 0.01
        with pytest.raises(ValueError, match="initial array"):
            check_initial_parity(routed, shared)


def test_candidate_training_guard_precedes_any_local_path_or_model(tmp_path, monkeypatch):
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    target = tmp_path / "artifacts/task-screen"
    with pytest.raises(RuntimeError, match="Colab"):
        execute_reference_candidate_colab(None, None, target)
    assert not target.exists()


def test_one_time_candidate_claim_preserves_original_charges(tmp_path, monkeypatch):
    from rl_stocks.training import task_runtime

    monkeypatch.setattr(task_runtime, "shared_budget", lambda ledger: (2.0, 39.92585919127071))
    ledger = SimpleNamespace(path=tmp_path / "ledger.json", value={"consumed": 40.1267})
    claim_candidate(ledger, "rl-stocks-cpu-012345abcdef")
    with pytest.raises(RuntimeError, match="already claimed"):
        claim_candidate(ledger, "rl-stocks-cpu-abcdef012345")
    record = json.loads((tmp_path / CLAIM).read_text())
    assert record["study_baseline"] == 39.92585919127071 and ledger.value["consumed"] == 40.1267


def tar_snapshot(path, content):
    with tarfile.open(path, "w:gz") as stream:
        member = tarfile.TarInfo("artifacts/initial.zip")
        member.size = len(content)
        stream.addfile(member, io.BytesIO(content))


def test_snapshot_corruption_or_collision_cannot_replace_valid_prior_bytes(tmp_path):
    first = tmp_path / "first.partial"
    tar_snapshot(first, b"checkpoint-a")
    target = tmp_path / "checkpoint-000001.tar.gz"
    record = retain(first, target, [])
    before = target.read_bytes()
    bad = tmp_path / "second.partial"
    bad.write_bytes(gzip.compress(b"not tar")[:-4])
    with pytest.raises((EOFError, OSError, tarfile.TarError)):
        retain(bad, tmp_path / "checkpoint-000002.tar.gz", [record])
    assert target.read_bytes() == before
    third = tmp_path / "third.partial"
    third.write_bytes(before)
    assert retain(third, tmp_path / "checkpoint-000003.tar.gz", [record]) is None
    assert not third.exists() and target.read_bytes() == before
    fourth = tmp_path / "fourth.partial"
    tar_snapshot(fourth, b"new")
    with pytest.raises(ValueError, match="overwritten"):
        retain(fourth, target, [record])
    assert target.read_bytes() == before


def fitted_point(role, score=0.60):
    groups = {
        kind: {
            "count": 22 if kind == "stock" else 4,
            "excess_return": 0.03,
            "median_excess_return": 0.03,
        }
        for kind in ("stock", "etf")
    }
    return {
        "role": role,
        "protocol_sha256": REFERENCE_SHA256,
        "phase": "final",
        "steps": 100000,
        "period": "training-window",
        "performance_execution": "reference-actor-v1",
        "seed": 11,
        "holding_path_sha256": "a" * 64,
        "metrics": {"score": score, "groups": groups},
    }


def test_reference_gate_retains_original_thresholds_and_rejects_mixed_execution():
    controls = {
        role: {"final": fitted_point(role, 0.55)} for role in ("masked_shared", "tagged_shared")
    }
    candidate = fitted_point("tagged_routed")
    assert reference_fitted_gate(controls, candidate)["passed"]
    bad = deepcopy(candidate)
    bad["metrics"]["score"] = 0.56
    assert not reference_fitted_gate(controls, bad)["passed"]
    bad = deepcopy(candidate)
    bad["metrics"]["groups"]["etf"]["excess_return"] = -0.021
    assert not reference_fitted_gate(controls, bad)["passed"]
    bad = deepcopy(candidate)
    bad["performance_execution"] = "native"
    with pytest.raises(ValueError):
        reference_fitted_gate(controls, bad)
    assert candidate["protocol_sha256"] == REFERENCE_SHA256
