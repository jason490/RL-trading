from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from rl_stocks.runtime.colab.budget import BudgetStop
from rl_stocks.training.crossq_spec import PROTOCOL_SHA256, protocol
from rl_stocks.training.crossq_assets import BINDING_SHA256
from rl_stocks.training.crossq_runtime import (
    make_runtime_profile, projection, shared_study_budget, validate_runtime_profile,
)


def accepted_profile(tmp_path):
    records = {
        arm: dict(duration_seconds=30.0, actual_saved_models_restored=True, frozen_predictions=78,
                  steps=4096, final_replay_restored=True, cleanup_verified=True,
                  observed_actor_and_critic_renorm_verified=True)
        for arm in protocol()["arms"]
    }
    path = tmp_path / ".runtime/crossq-smoke-acceptance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(
                protocol_sha256=PROTOCOL_SHA256,
                binding_sha256=BINDING_SHA256,
                passed=True,
                cleanup_verified=True,
                initialization_checks={
                    "sampling_rng_identical": True,
                    "shared_linear_tensors_verified": True,
                },
                records=records,
            )
        )
    )
    return make_runtime_profile(tmp_path)


def test_runtime_projection_and_acceptance_mutations_do_not_authorize_screen(tmp_path):
    profile = accepted_profile(tmp_path)
    validate_runtime_profile(profile, tmp_path)
    bad = deepcopy(profile)
    bad["projection"]["projected_seconds"] = 1
    with pytest.raises(ValueError):
        validate_runtime_profile(bad)
    assert not projection(dict(sac=1, crossq=40))["screen_allocation_allowed"]
    path = tmp_path / ".runtime/crossq-smoke-acceptance.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        validate_runtime_profile(profile, tmp_path)


def test_budget_baseline_survives_controllers_and_cannot_reset(tmp_path):
    ledger = SimpleNamespace(path=tmp_path / "usage-ledger.json", value={"consumed": 39.0})
    assert shared_study_budget(ledger) == (2.0, 39.0)
    ledger.value["consumed"] = 39.5
    assert shared_study_budget(ledger) == (2.0, 39.0)
    with pytest.raises(ValueError):
        shared_study_budget(ledger, requested_start=39.5)
    with pytest.raises(ValueError):
        shared_study_budget(ledger, requested_limit=3)
    ledger.value["consumed"] = 41.0
    with pytest.raises(BudgetStop):
        shared_study_budget(ledger)


@pytest.mark.parametrize("field", ["cleanup_verified", "mapping", "restoration", "predictions", "protocol"])
def test_unverified_smokes_never_authorize_training(tmp_path, field):
    accepted_profile(tmp_path)
    path = tmp_path / ".runtime/crossq-smoke-acceptance.json"
    value = json.loads(path.read_text())
    if field == "cleanup_verified":
        value[field] = False
    elif field == "mapping":
        value["initialization_checks"]["shared_linear_tensors_verified"] = False
    elif field == "restoration":
        value["records"]["crossq"]["actual_saved_models_restored"] = False
    elif field == "predictions":
        value["records"]["sac"]["frozen_predictions"] = 77
    else:
        value["protocol_sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        make_runtime_profile(tmp_path)


@pytest.mark.parametrize("durations", [{}, dict(sac=True, crossq=1), dict(sac=1, crossq=float("nan"))])
def test_invalid_actual_durations_reject(durations):
    with pytest.raises(ValueError):
        projection(durations)


@pytest.mark.parametrize("field", ["final_replay_restored", "cleanup_verified", "observed_actor_and_critic_renorm_verified", "steps"])
def test_crossq_actual_learning_evidence_required(tmp_path, field):
    accepted_profile(tmp_path)
    path = tmp_path / ".runtime/crossq-smoke-acceptance.json"
    value = json.loads(path.read_text())
    value["records"]["crossq"][field] = 512 if field == "steps" else False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        make_runtime_profile(tmp_path)


def test_persisted_integrity_stop_blocks_profiles_and_new_controllers(tmp_path):
    profile = accepted_profile(tmp_path)
    ledger = SimpleNamespace(path=tmp_path / '.runtime/usage-ledger.json', value={'consumed': 39.0})
    assert shared_study_budget(ledger) == (2.0, 39.0)
    baseline = (tmp_path / '.runtime/crossq-study-budget.json').read_bytes()
    # Even a malformed/empty stop record must fail closed, never erase history.
    stop = tmp_path / '.runtime/crossq-integrity-stop.json'
    stop.write_text('{}')
    with pytest.raises(RuntimeError, match='no automatic restart'):
        validate_runtime_profile(profile, tmp_path)
    with pytest.raises(RuntimeError, match='no automatic restart'):
        make_runtime_profile(tmp_path)
    with pytest.raises(RuntimeError, match='no automatic restart'):
        shared_study_budget(ledger)
    assert stop.read_text() == '{}'
    assert (tmp_path / '.runtime/crossq-study-budget.json').read_bytes() == baseline
