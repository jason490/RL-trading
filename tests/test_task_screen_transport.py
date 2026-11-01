from copy import deepcopy
import tarfile

import pytest

from rl_stocks.runtime.colab.jobs import task_screen_plan, validate_plan
from rl_stocks.runtime.colab.task_screen_artifacts import extract_screen_transport


def test_fixed_screen_contract_rejects_changed_budget_and_steps():
    plan = task_screen_plan('tagged_routed', 'f' * 64)
    validate_plan(plan)
    changed = deepcopy(plan)
    changed['jobs'][0]['steps'] = 250000
    with pytest.raises(ValueError, match='fixed profile-bound'):
        validate_plan(changed)
    changed = deepcopy(plan)
    changed['allocation_deadline_seconds'] = 14400
    with pytest.raises(ValueError, match='fixed profile-bound'):
        validate_plan(changed)
    with pytest.raises(ValueError, match='acceptance identity'):
        task_screen_plan('tagged_routed', '')


def test_incomplete_transport_does_not_create_destination(tmp_path):
    plan = task_screen_plan('tagged_routed', 'f' * 64)
    path = tmp_path / 'empty.tar'
    with tarfile.open(path, 'w'):
        pass
    target = tmp_path / 'extracted'
    with pytest.raises(RuntimeError, match='complete bounded'):
        extract_screen_transport(path, plan, plan['dataset_contract']['identity'], target)
    assert not target.exists()
