import pytest
from rl_stocks.training.task_profile_worker import execute_profile_colab


def test_profile_rejects_local_execution_before_creating_output(tmp_path):
    output = tmp_path / 'artifacts' / 'profile'
    with pytest.raises(RuntimeError, match='owned Colab CPU'):
        execute_profile_colab(None, None, output, task_role='tagged_routed')
    assert not output.exists()


def test_profile_plan_requires_all_roles_and_research_geometry(tmp_path):
    from copy import deepcopy
    from rl_stocks.runtime.colab.jobs import task_profile_plan, validate_plan
    from rl_stocks.runtime.colab.fleet import validate_tasks
    plan = task_profile_plan()
    validate_plan(plan)
    assert len(plan['jobs']) == 3
    assert all(job['steps'] == 4096 and not job['smoke'] for job in plan['jobs'])
    for field, value in [('accelerator', 'A100'), ('task_profile_roles', ['tagged_routed']),
                         ('allocation_deadline_seconds', 14400)]:
        altered = deepcopy(plan)
        altered[field] = value
        with pytest.raises(ValueError, match='fixed three-role'):
            validate_plan(altered)
    with pytest.raises(ValueError, match='awaits complete smoke'):
        validate_tasks([{'id': 'profile', 'dataset': 'unused', 'plan': plan}], tmp_path)
