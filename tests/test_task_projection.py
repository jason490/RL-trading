import pytest
from rl_stocks.training.task_projection import project_screen


def records(learning=10):
    return [dict(task_role=role, learning_seconds=learning, evaluation_seconds=50.)
            for role in ('masked_shared', 'tagged_shared', 'tagged_routed')]


def test_projection_includes_setup_all_evaluations_and_cleanup():
    result = project_screen(records(), remaining_study_units=1.9, aggregate_rate=.08)
    assert result['execution_admitted']
    assert all(r['projected_allocation_seconds'] > 1500 + 750 for r in result['roles'].values())
    assert result['projected_units'] > .15
    assert not result['performance_promotion']
    assert not project_screen(records(200), remaining_study_units=2., aggregate_rate=.08)['execution_admitted']
    assert not project_screen(records(), remaining_study_units=.01, aggregate_rate=.08)['execution_admitted']


@pytest.mark.parametrize('rate', [0, float('nan'), True, -1])
def test_unreliable_rate_rejected(rate):
    with pytest.raises(ValueError):
        project_screen(records(), remaining_study_units=2., aggregate_rate=rate)
