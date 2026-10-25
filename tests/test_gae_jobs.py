from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_stocks.runtime.colab.budget import BudgetStop
from rl_stocks.runtime.colab.jobs import gae_plan, validate_plan
from rl_stocks.runtime.colab.payloads import prepare_payload
from rl_stocks.training.gae_spec import (
    PROTOCOL_SHA256,
    protocol,
)
from rl_stocks.training.gae_runtime import (
    make_runtime_profile,
    projection,
    shared_study_budget,
    validate_runtime_profile,
)


def accepted_profile(tmp_path):
    records = {
        arm: dict(
            duration_seconds=30.0,
            actual_saved_models_restored=True,
            actual_rollout_arrays_verified=True,
            frozen_predictions=78,
        )
        for arm in protocol()["arms"]
    }
    path = tmp_path / ".runtime/gae-smoke-acceptance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(
                protocol_sha256=PROTOCOL_SHA256,
                passed=True,
                cleanup_verified=True,
                initialization_checks={
                    "sampling_rng_identical": True,
                    "common_initial_tensors_and_empty_optimizers_exact": True,
                    "only_gae_lambda_differs": True,
                },
                records=records,
            )
        )
    )
    return make_runtime_profile(tmp_path)


@pytest.mark.parametrize("arm", list(protocol()["arms"]))
def test_actual_bound_smoke_and_gated_full_plans(tmp_path, arm):
    smoke = gae_plan(arm, smoke=True)
    validate_plan(smoke)
    assert smoke["version"] == 35 and smoke["jobs"][0]["steps"] == 512
    assert smoke["jobs"][0]["algorithm"] == "recurrent_ppo"
    with pytest.raises(ValueError):
        gae_plan(arm)
    full = gae_plan(arm, profile=accepted_profile(tmp_path))
    validate_plan(full)
    assert full["jobs"][0]["steps"] == 100000


@pytest.mark.parametrize(
    "field", ["algorithm", "steps", "overrides", "initialization", "boolean_alias"]
)
def test_plan_semantic_tampering_rejected(field):
    plan = deepcopy(gae_plan("mc_trace", smoke=True))
    if field == "algorithm":
        plan["jobs"][0][field] = "sac"
    if field == "steps":
        plan["jobs"][0][field] = 2048
    if field == "overrides":
        plan["jobs"][0][field]["n_epochs"] = 1
    if field == "boolean_alias":
        plan["jobs"][0]["overrides"]["n_steps"] = True
    if field == "initialization":
        plan["initialization_contract"]["rng_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_runtime_projection_and_acceptance_mutations_do_not_authorize_screen(tmp_path):
    profile = accepted_profile(tmp_path)
    validate_runtime_profile(profile, tmp_path)
    bad = deepcopy(profile)
    bad["projection"]["projected_seconds"] = 1
    with pytest.raises(ValueError):
        validate_runtime_profile(bad)
    assert not projection(dict(control=1, mc_trace=40))["screen_allocation_allowed"]
    path = tmp_path / ".runtime/gae-smoke-acceptance.json"
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


def test_source_payload_carries_all_bound_initializer_files(tmp_path):
    import tarfile

    root = Path(__file__).parents[1]
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("{}")
    # Use repository source plus isolated job staging; no account or data requests.
    plan = gae_plan("mc_trace", smoke=True)
    packed = prepare_payload(root, dataset, plan, "test-gae-payload-" + tmp_path.name)
    try:
        with tarfile.open(packed) as archive:
            names = {m.name for m in archive.getmembers() if m.isfile()}
            assert "configs/initializations/recurrent_capacity_v1/manifest.json" in names
            assert (
                sum(
                    n.startswith("configs/initializations/recurrent_guard_v2/")
                    and n.endswith(".npz")
                    for n in names
                )
                >= 3
            )
    finally:
        import shutil

        shutil.rmtree(packed.parent)


def test_fleet_automatically_preserves_study_baseline_and_stops_projected_overrun(
    tmp_path, monkeypatch
):
    from rl_stocks.runtime.colab import fleet

    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    ledger_path = runtime / "usage-ledger.json"
    value = dict(
        ceiling=150.0,
        working_limit=130.0,
        consumed=39.0,
        attempts=[],
        budget_stopped=False,
        minimum_account_balance=150,
    )
    ledger_path.write_text(json.dumps(value))
    tasks = [dict(id="gae-full-smoke", dataset="unused", plan=gae_plan("mc_trace", smoke=True))]
    monkeypatch.setattr(fleet, "validate_tasks", lambda tasks, root: tasks)
    first = fleet.Fleet(tmp_path, object(), tasks, 1, runtime / "state.json")
    assert first.study_budget == dict(start_units=39.0, limit_units=2.0)
    value["consumed"] = 40.9
    value["attempts"] = [dict(last_time=0.0, rate=1.0, session="owned-fleet")]
    ledger_path.write_text(json.dumps(value))
    second = fleet.Fleet(tmp_path, object(), tasks, 1, runtime / "state2.json")
    assert second.study_budget == first.study_budget
    second.budget.ledger.clock = lambda: 0.0
    with pytest.raises(BudgetStop):
        second.guard_study(360)
    saved = json.loads(ledger_path.read_text())
    assert saved["budget_stopped"] and saved["minimum_account_balance"] == 150
    assert saved["consumed"] == 40.9
    assert json.loads((runtime / "state2.json").read_text())["study_budget_stopped"] is True


def test_fleet_rejects_wrong_bound_dataset_before_allocation(tmp_path):
    from rl_stocks.runtime.colab.fleet import validate_tasks

    dataset = tmp_path / "wrong-identity"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("{}")
    tasks = [dict(id="gae-full", dataset=str(dataset), plan=gae_plan("mc_trace", smoke=True))]
    with pytest.raises(ValueError, match="identity"):
        validate_tasks(tasks, tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(cleanup_verified=False),
        lambda v: v["initialization_checks"].update(only_gae_lambda_differs=False),
        lambda v: v["initialization_checks"].update(
            common_initial_tensors_and_empty_optimizers_exact=False
        ),
        lambda v: v["records"]["mc_trace"].update(actual_rollout_arrays_verified=False),
        lambda v: v["records"]["mc_trace"].update(frozen_predictions=77),
    ],
)
def test_self_consistent_smoke_file_without_required_acceptance_cannot_gate_full_run(
    tmp_path, change
):
    accepted_profile(tmp_path)
    path = tmp_path / ".runtime/gae-smoke-acceptance.json"
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="smokes required"):
        make_runtime_profile(tmp_path)
