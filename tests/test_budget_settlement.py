"""Delayed post-shutdown metering is monotone and never restarts resources."""
from copy import deepcopy

import pytest

from rl_stocks.runtime.colab.budget import Ledger, Usage, BudgetStop


def stopped(tmp_path):
    now = [0.]
    ledger = Ledger(tmp_path / 'ledger.json', clock=lambda: now[0])
    ledger.value['minimum_account_balance'] = 150
    ledger.begin('owned', Usage(350, 0, 0))
    now[0] = 60.
    ledger.finish(True)
    return ledger, now


def test_delayed_debit_charges_once_and_keeps_operational_clock(tmp_path):
    ledger, now = stopped(tmp_path)
    original = deepcopy(ledger.active)
    now[0] = 3600.
    ledger.settle_stopped('owned', Usage(349.99, 0, 0))
    assert ledger.value['consumed'] == pytest.approx(0.01)
    assert ledger.active['estimated'] == original['estimated']
    assert ledger.active['last_time'] == original['last_time']
    assert ledger.active['end'] == original['end']
    assert ledger.active['status'] == 'stopped'
    assert ledger.settle_stopped('owned', Usage(349.99, 0, 0)) == 0
    assert ledger.settle_stopped('owned', Usage(351, 0, 0)) == 0
    assert ledger.value['consumed'] == pytest.approx(0.01)


@pytest.mark.parametrize('kind', ['session', 'active', 'assignments', 'rate', 'nan'])
def test_unreliable_or_unowned_settlement_rejects_without_mutation(tmp_path, kind):
    ledger, _ = stopped(tmp_path)
    session, usage = 'owned', Usage(349, 0, 0)
    if kind == 'session':
        session = 'other'
    elif kind == 'active':
        ledger.active['status'] = 'active'
    elif kind == 'assignments':
        usage = Usage(349, 0, 1)
    elif kind == 'rate':
        usage = Usage(349, 1, 0)
    else:
        usage = Usage(float('nan'), 0, 0)
    before = deepcopy(ledger.value)
    with pytest.raises(BudgetStop):
        ledger.settle_stopped(session, usage)
    assert ledger.value == before


def test_existing_stop_and_account_protection_survive_settlement(tmp_path):
    ledger, _ = stopped(tmp_path)
    ledger.value['budget_stopped'] = True
    ledger.settle_stopped('owned', Usage(349.99, 0, 0))
    assert ledger.value['budget_stopped'] is True
    assert ledger.value['minimum_account_balance'] == 150


def test_invalid_floor_rejects_before_mutating_charges(tmp_path):
    ledger, _ = stopped(tmp_path)
    ledger.value['minimum_account_balance'] = -1
    before = deepcopy(ledger.value)
    with pytest.raises(BudgetStop):
        ledger.settle_stopped('owned', Usage(349.99, 0, 0))
    assert ledger.value == before
