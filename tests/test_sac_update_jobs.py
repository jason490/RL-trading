from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from rl_stocks.runtime.colab.jobs import sac_update_plan, validate_plan
from rl_stocks.runtime.colab.budget import Usage, BudgetStop
from rl_stocks.runtime.colab.fleet_budget import FleetBudget
from rl_stocks.training.sac_update_runtime import shared_budget, claim_allocation


def test_fixed_plan_and_mutation_rejection():
    value = sac_update_plan()
    validate_plan(json.loads(json.dumps(value)))
    assert value["allocation_deadline_seconds"] == 900 and value["jobs"][0]["steps"] == 2
    for field, bad in [("steps", 3), ("device", "cuda"), ("algorithm", "crossq"), ("smoke", True)]:
        changed = deepcopy(value)
        changed["jobs"][0][field] = bad
        with pytest.raises(ValueError):
            validate_plan(changed)


def test_budget_and_single_claim_never_reset_or_erase_prior_crossq_stop(tmp_path):
    ledger = SimpleNamespace(path=tmp_path / "usage-ledger.json", value={"consumed": 39.5})
    stop = tmp_path / "crossq-integrity-stop.json"
    stop.write_text("retained")
    assert shared_budget(ledger) == (0.1, 39.5)
    ledger.value["consumed"] = 39.55
    assert shared_budget(ledger) == (0.1, 39.5)
    with pytest.raises(ValueError):
        shared_budget(ledger, requested_start=39.55)
    with pytest.raises(ValueError):
        shared_budget(ledger, requested_limit=1)
    claim_allocation(ledger, "rl-stocks-cpu-fixture")
    with pytest.raises(RuntimeError, match="no automatic retry"):
        shared_budget(ledger)
    with pytest.raises(RuntimeError):
        claim_allocation(ledger, "rl-stocks-cpu-another")
    assert stop.read_text() == "retained"


def test_diagnostic_deadline_reserves_time_for_cleanup(tmp_path):
    now = [0.0]
    budget = FleetBudget(tmp_path / "usage-ledger.json", clock=lambda: now[0])
    budget.start("fixture", Usage(350, 0, 0), 1)
    child = budget.admit("owned", "CPU", allocation_seconds=900, teardown_reserve_seconds=420)
    assert child["deadline"] == 900
    now[0] = 479
    budget.guard("owned", 1)
    now[0] = 480
    with pytest.raises(BudgetStop):
        budget.guard("owned", 1)
    assert budget.ledger.value["budget_stopped"]


def test_fleet_inherits_diagnostic_allowance_and_rejects_concurrency(tmp_path, monkeypatch):
    from rl_stocks.runtime.colab import fleet

    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    (runtime / "usage-ledger.json").write_text(
        json.dumps(
            dict(
                ceiling=150.0,
                working_limit=130.0,
                consumed=39.5,
                attempts=[],
                budget_stopped=False,
                minimum_account_balance=150,
            )
        )
    )
    tasks = [dict(id="sac-update", dataset="unused", plan=sac_update_plan())]
    monkeypatch.setattr(fleet, "validate_tasks", lambda tasks, root: tasks)
    with pytest.raises(ValueError, match="exactly one"):
        fleet.Fleet(tmp_path, object(), tasks, 2, runtime / "state.json")
    controller = fleet.Fleet(tmp_path, object(), tasks, 1, runtime / "state.json")
    assert controller.study_budget == dict(start_units=39.5, limit_units=0.1)
    claim_allocation(controller.budget.ledger, "rl-stocks-cpu-fixture")
    with pytest.raises(RuntimeError, match="no automatic retry"):
        fleet.Fleet(tmp_path, object(), tasks, 1, runtime / "new-state.json")
