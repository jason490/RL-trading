from copy import deepcopy
import pytest
from rl_stocks.runtime.colab.budget import Ledger,Usage,BudgetStop


def stopped(tmp_path):
    ledger=Ledger(tmp_path/'ledger.json',clock=lambda:100.)
    ledger.value.update(consumed=37.7872,budget_stopped=True,
        attempts=[dict(session='stopped-fleet',status='stopped',stop_reason='meter unavailable',
                       children={'owned':dict(status='stopped',cleanup_verified=True)})])
    ledger.save();return ledger


def test_review_preserves_usage_and_history_and_enforces_floor(tmp_path):
    ledger=stopped(tmp_path);attempts=deepcopy(ledger.value['attempts'])
    review=ledger.review_resume(Usage(349.97,0,0),'User authorizes resume with floor150','stopped-fleet',150)
    assert ledger.value['consumed']==37.7872 and ledger.value['attempts']==attempts
    assert review['minimum_account_balance']==150 and ledger.account_teardown_balance==170
    assert not ledger.value['budget_stopped']
    with pytest.raises(BudgetStop):ledger.begin('new',Usage(170,0,0))
    ledger.begin('new',Usage(171,0,0));ledger.active['rate']=100.
    with pytest.raises(BudgetStop):ledger.guard(60)
    assert ledger.value['budget_stopped']


@pytest.mark.parametrize('usage,authorization,session,floor',[
    (Usage(349,0,0),'','stopped-fleet',150),
    (Usage(349,0,0),'approved','wrong',150),
    (Usage(349,.08,1),'approved','stopped-fleet',150),
    (Usage(170,0,0),'approved','stopped-fleet',150),
    (Usage(349,0,0),'approved','stopped-fleet',float('nan')),
])
def test_review_failure_leaves_ledger_untouched(tmp_path,usage,authorization,session,floor):
    ledger=stopped(tmp_path);before=deepcopy(ledger.value)
    with pytest.raises(BudgetStop):ledger.review_resume(usage,authorization,session,floor)
    assert ledger.value==before


def test_unresolved_cleanup_and_project_threshold_prevent_review(tmp_path):
    ledger=stopped(tmp_path);ledger.active['children']['owned']['cleanup_verified']=False
    with pytest.raises(BudgetStop):ledger.review_resume(Usage(349,0,0),'approved','stopped-fleet',150)
    ledger.active['children']['owned']['cleanup_verified']=True;ledger.value['consumed']=130
    with pytest.raises(BudgetStop):ledger.review_resume(Usage(349,0,0),'approved','stopped-fleet',150)
