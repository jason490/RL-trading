"""Pre-allocation enforcement using a temporary ledger, never a Colab session."""
import json

import pytest

from rl_stocks.runtime.colab import fleet
from rl_stocks.runtime.colab.budget import BudgetStop
from rl_stocks.runtime.colab.jobs import crossq_plan, treasury_plan


def test_original_study_budget_and_project_stop_survive_fleet_retry(tmp_path, monkeypatch):
    runtime = tmp_path / '.runtime'
    runtime.mkdir()
    path = runtime / 'usage-ledger.json'
    value = dict(ceiling=150., working_limit=130., consumed=39., attempts=[],
                 budget_stopped=False, minimum_account_balance=150)
    path.write_text(json.dumps(value))
    tasks = [dict(id='crossq-smoke', dataset='unused', plan=crossq_plan('crossq', smoke=True))]
    monkeypatch.setattr(fleet, 'validate_tasks', lambda tasks, root: tasks)
    first = fleet.Fleet(tmp_path, object(), tasks, 1, runtime / 'state.json')
    assert first.study_budget == dict(start_units=39., limit_units=2.)
    value.update(consumed=40.9, attempts=[dict(last_time=0., rate=1., session='owned')])
    path.write_text(json.dumps(value))
    second = fleet.Fleet(tmp_path, object(), tasks, 1, runtime / 'retry.json')
    assert second.study_budget == first.study_budget
    second.budget.ledger.clock = lambda: 0.
    with pytest.raises(BudgetStop):
        second.guard_study(360)
    assert json.loads(path.read_text())['budget_stopped'] is True
    assert json.loads(path.read_text())['minimum_account_balance'] == 150


def test_wrong_dataset_stops_before_allocation(tmp_path):
    directory = tmp_path / 'wrong'
    directory.mkdir()
    (directory / 'dataset.json').write_text('{}')
    tasks = [dict(id='crossq-smoke', dataset=str(directory), plan=crossq_plan('crossq', smoke=True))]
    with pytest.raises(ValueError, match='identity'):
        fleet.validate_tasks(tasks, tmp_path)


def test_mixed_families_cannot_share_crossq_allowance(tmp_path, monkeypatch):
    runtime = tmp_path / '.runtime'
    runtime.mkdir()
    (runtime / 'usage-ledger.json').write_text(json.dumps(dict(
        ceiling=150., working_limit=130., consumed=39., attempts=[],
        budget_stopped=False, minimum_account_balance=150)))
    tasks = [dict(plan=crossq_plan('crossq', smoke=True)), dict(plan=treasury_plan('full', smoke=True))]
    monkeypatch.setattr(fleet, 'validate_tasks', lambda tasks, root: tasks)
    with pytest.raises(ValueError, match='own shared-budget'):
        fleet.Fleet(tmp_path, object(), tasks, 1, runtime / 'state.json')
