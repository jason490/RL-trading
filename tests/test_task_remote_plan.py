from copy import deepcopy

import pytest

from rl_stocks.runtime.colab.jobs import task_smoke_plan, validate_plan
from rl_stocks.runtime.colab.fleet import validate_tasks


@pytest.mark.parametrize('role', ['masked_shared', 'tagged_shared', 'tagged_routed'])
def test_fixed_task_remote_contract(role):
    plan = task_smoke_plan(role)
    validate_plan(plan)
    assert plan['jobs'][0]['steps'] == 1024
    assert plan['accelerator'] == 'CPU'
    for mutation in ('steps', 'device', 'seed', 'smoke'):
        bad = deepcopy(plan)
        bad['jobs'][0][mutation] = {'steps': 100000, 'device': 'cuda', 'seed': 23,
                                   'smoke': False}[mutation]
        with pytest.raises(ValueError, match='fixed task smoke'):
            validate_plan(bad)
    bad = deepcopy(plan)
    bad['type_manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='fixed task smoke'):
        validate_plan(bad)


def test_task_smoke_cannot_allocate_before_controller_integration(tmp_path):
    with pytest.raises(ValueError, match='awaits aggregate budget'):
        validate_tasks([{'id': 'task-smoke', 'dataset': 'unused',
                         'plan': task_smoke_plan('tagged_routed')}], tmp_path)
    with pytest.raises(ValueError):
        task_smoke_plan('unknown')
