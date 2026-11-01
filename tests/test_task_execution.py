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
    with pytest.raises(FileNotFoundError):
        verify_execution(tmp_path, ledger)
