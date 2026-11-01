from types import SimpleNamespace

import pytest

from rl_stocks.training.task_execution import claim_screen, review_spec, verify_execution


def test_review_preserves_initial_failure_and_bounds_subsequent_runs():
    review = review_spec()
    assert review['original_execution_admitted'] is False
    assert review['allocation_seconds'] == {'masked_shared': 7200, 'tagged_shared': 7200, 'tagged_routed': 9000}
    assert review['maximum_incremental_study_units'] == 2
    assert review['learning_protocol_unchanged']


def test_screen_claim_cannot_repeat_or_clear_prior_charges(tmp_path):
    ledger = SimpleNamespace(path=tmp_path / 'ledger.json', account_teardown_balance=170,
        value={'consumed': 39.97, 'budget_stopped': False, 'minimum_account_balance': 150, 'attempts': []})
    claim_screen(ledger, 'rl-stocks-cpu-012345abcdef', 'tagged_routed')
    with pytest.raises(RuntimeError, match='already claimed'):
        claim_screen(ledger, 'rl-stocks-cpu-abcdef012345', 'tagged_routed')
    assert ledger.value['consumed'] == 39.97
    with pytest.raises(ValueError, match='already claimed'):
        verify_execution(tmp_path, ledger)


def test_reviewed_job_cannot_change_profile_duration_or_learning_budget():
    from copy import deepcopy
    from rl_stocks.runtime.colab.jobs import reviewed_task_screen_plan, validate_plan
    for role, seconds in review_spec()['allocation_seconds'].items():
        plan = reviewed_task_screen_plan(role)
        validate_plan(plan)
        assert plan['allocation_deadline_seconds'] == seconds
        for field, value in [('allocation_deadline_seconds', seconds + 3600),
                             ('profile_acceptance_sha256', '0' * 64)]:
            altered = deepcopy(plan)
            altered[field] = value
            with pytest.raises(ValueError, match='fixed reviewed'):
                validate_plan(altered)
        altered = deepcopy(plan)
        altered['jobs'][0]['steps'] = 250000
        with pytest.raises(ValueError, match='fixed reviewed'):
            validate_plan(altered)


def test_screen_suffix_requires_exact_retained_acceptance_and_unclaimed_roles(tmp_path):
    from rl_stocks.training.task_execution import verify_screen_roles
    ledger = SimpleNamespace(path=tmp_path / 'ledger.json')
    verify_screen_roles(tmp_path, ledger, ['masked_shared', 'tagged_shared', 'tagged_routed'])
    with pytest.raises(ValueError, match='matrix'):
        verify_screen_roles(tmp_path, ledger, ['tagged_routed'])
    with pytest.raises(FileNotFoundError):
        verify_screen_roles(tmp_path, ledger, ['tagged_shared', 'tagged_routed'])
    (tmp_path / 'task-screen-tagged_shared-allocation-v1.json').write_text('{}')
    with pytest.raises(ValueError, match='already claimed'):
        verify_screen_roles(tmp_path, ledger, ['tagged_shared', 'tagged_routed'])
