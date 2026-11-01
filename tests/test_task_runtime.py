from types import SimpleNamespace

import pytest

from rl_stocks.runtime.colab.budget import BudgetStop
from rl_stocks.training.task_runtime import claim_smoke, shared_budget


def ledger(tmp_path):
    return SimpleNamespace(path=tmp_path / 'usage-ledger.json',
        value={'consumed': 39.9, 'budget_stopped': False, 'minimum_account_balance': 150},
        account_teardown_balance=170)


def test_shared_baseline_preserves_charges_and_exhaustion(tmp_path):
    item = ledger(tmp_path)
    assert shared_budget(item) == (2., 39.9)
    item.value['consumed'] += 1
    assert shared_budget(item) == (2., 39.9)
    with pytest.raises(ValueError, match='preserved'):
        shared_budget(item, requested_start=40.9)
    item.value['consumed'] += 1
    with pytest.raises(BudgetStop, match='exhausted'):
        shared_budget(item)
    for value in (True, float('nan'), 3, 0):
        with pytest.raises(ValueError, match='allowance'):
            shared_budget(item, value)


def test_claims_persist_independently_and_cannot_retry(tmp_path):
    item = ledger(tmp_path)
    claim_smoke(item, 'rl-stocks-cpu-012345abcdef', 'masked_shared')
    with pytest.raises(RuntimeError, match='already claimed'):
        claim_smoke(item, 'rl-stocks-cpu-abcdef012345', 'masked_shared')
    claim_smoke(item, 'rl-stocks-cpu-abcdef012345', 'tagged_shared')
    assert shared_budget(item)[1] == 39.9
    with pytest.raises(ValueError, match='owned CPU'):
        claim_smoke(item, 'unrelated', 'tagged_routed')


@pytest.mark.parametrize('key,value', [('budget_stopped', True), ('minimum_account_balance', 0)])
def test_existing_budget_stop_and_account_floor_are_not_cleared(tmp_path, key, value):
    item = ledger(tmp_path)
    item.value[key] = value
    with pytest.raises(BudgetStop, match='protections'):
        shared_budget(item)
    assert not (tmp_path / 'task-conditioning-budget-v1.json').exists()


def test_only_complete_initial_matrix_or_exact_reviewed_suffix(tmp_path):
    from rl_stocks.training.task_runtime import validate_smoke_roles
    item = ledger(tmp_path)
    validate_smoke_roles(tmp_path, item, ['masked_shared', 'tagged_shared', 'tagged_routed'])
    for roles in (['masked_shared'], ['tagged_shared'], ['tagged_shared'] * 3):
        with pytest.raises(ValueError, match='matrix'):
            validate_smoke_roles(tmp_path, item, roles)
    with pytest.raises(FileNotFoundError):
        validate_smoke_roles(tmp_path, item, ['tagged_shared', 'tagged_routed'])


def test_claimed_role_cannot_reenter_complete_initial_matrix(tmp_path):
    from rl_stocks.training.task_runtime import validate_smoke_roles
    item = ledger(tmp_path)
    claim_smoke(item, 'rl-stocks-cpu-012345abcdef', 'masked_shared')
    with pytest.raises(ValueError, match='already claimed'):
        validate_smoke_roles(tmp_path, item, ['masked_shared', 'tagged_shared', 'tagged_routed'])


def test_profile_has_one_persistent_allocation_claim(tmp_path):
    from rl_stocks.training.task_runtime import claim_profile
    item = ledger(tmp_path)
    claim_profile(item, 'rl-stocks-cpu-012345abcdef')
    with pytest.raises(RuntimeError, match='already claimed'):
        claim_profile(item, 'rl-stocks-cpu-abcdef012345')
    assert shared_budget(item) == (2., 39.9)


def test_complete_smoke_admission_requires_retained_evidence(tmp_path):
    from rl_stocks.training.task_runtime import verify_all_smokes
    with pytest.raises(FileNotFoundError):
        verify_all_smokes(tmp_path, ledger(tmp_path))


def test_closed_family_cannot_allocate_or_reset_budget(tmp_path):
    item = ledger(tmp_path)
    (tmp_path / 'task-conditioning-family-closure-v1.json').write_text('{}')
    with pytest.raises(RuntimeError, match='family closed'):
        shared_budget(item)
    assert item.value['consumed'] == 39.9
    assert not item.value['budget_stopped']
