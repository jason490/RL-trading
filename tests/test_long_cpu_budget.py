"""Opt-in long CPU budget plans; default deadlines/account protections remain intact."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_stocks.runtime.colab.budget import Usage,BudgetStop
from rl_stocks.runtime.colab.fleet_budget import FleetBudget
from rl_stocks.runtime.colab import fleet


def test_legacy_default_cannot_admit_long_allocation(tmp_path):
    budget=FleetBudget(tmp_path/'ledger.json',clock=lambda:0.)
    budget.start('fixture',Usage(350.,0.,0),1)
    assert budget.active['deadline']==7200
    with pytest.raises(ValueError):
        budget.admit('owned','CPU',allocation_seconds=21600)
    assert not budget.active['children']


def test_opt_in_requires_complete_plan_before_ledger_mutation(tmp_path):
    path=tmp_path/'ledger.json'
    budget=FleetBudget(path,clock=lambda:0.,allocation_limit_seconds=21600)
    for durations in (None,[],[21601],[True],[21600,10800]):
        with pytest.raises(ValueError):
            budget.start('fixture',Usage(350.,0.,0),1,allocation_plan_seconds=durations)
    assert not path.exists()
    for limit in (True,7200.,21601,7199):
        with pytest.raises(ValueError):
            FleetBudget(path,allocation_limit_seconds=limit)


def test_aggregate_charges_duration_slots_and_six_hour_teardown(tmp_path):
    now=[0.]
    budget=FleetBudget(tmp_path/'ledger.json',clock=lambda:now[0],allocation_limit_seconds=21600)
    budget.ledger.value['minimum_account_balance']=150
    budget.start('fixture',Usage(350.,0.,0),3,allocation_plan_seconds=[21600,10800,7200])
    assert budget.active['deadline']==39600
    for session,duration in [('gru',21600),('sac',10800),('ppo',7200)]:
        budget.admit(session,'CPU',allocation_seconds=duration,teardown_reserve_seconds=360)
    with pytest.raises(ValueError,match='duration slot'):
        budget.admit('extra','CPU',allocation_seconds=21600)
    budget.observe(Usage(350.,.24,3))
    now[0]=3600
    budget.observe(Usage(349.76,.24,3))
    assert budget.ledger.value['consumed']==pytest.approx(.25)  # once at aggregate rate, not once per child
    budget.finish_child('ppo',True)
    budget.finish_child('sac',True)
    with pytest.raises(ValueError,match='no implicit retry'):
        budget.admit('ppo-retry','CPU',allocation_seconds=7200)
    now[0]=21239
    budget.guard('gru',1)
    now[0]=21240
    with pytest.raises(BudgetStop,match='allocation deadline'):
        budget.guard('gru',1)
    assert budget.ledger.value['budget_stopped']
    assert budget.ledger.account_teardown_balance==170


@pytest.mark.parametrize('constraint',['project','account'])
def test_long_duration_never_relaxes_original_credit_thresholds(tmp_path,constraint):
    budget=FleetBudget(tmp_path/'ledger.json',clock=lambda:0.,allocation_limit_seconds=21600)
    budget.ledger.value['minimum_account_balance']=150
    budget.start('fixture',Usage(350.,0.,0),1,allocation_plan_seconds=[21600])
    budget.admit('owned','CPU',allocation_seconds=21600,teardown_reserve_seconds=360)
    budget.active['rate']=1.
    if constraint=='project':
        budget.ledger.value['consumed']=129.99
    else:
        budget.active['last_balance']=170.01
    with pytest.raises(BudgetStop):
        budget.guard('owned',1)
    assert budget.ledger.value['ceiling']==150 and budget.ledger.value['working_limit']==130


@pytest.mark.parametrize('planned,expected_timeout',[(False,5400),(True,20999)])
def test_cli_execution_window_tracks_registered_long_duration_only(tmp_path,monkeypatch,planned,expected_timeout):
    budget=FleetBudget(tmp_path/'ledger.json',clock=lambda:0.,allocation_limit_seconds=21600 if planned else 7200)
    if planned:
        budget.ledger.value['minimum_account_balance']=150
    kwargs=dict(allocation_plan_seconds=[21600]) if planned else {}
    budget.start('fixture',Usage(350.,0.,0),1,**kwargs)
    budget.admit('owned','CPU',allocation_seconds=21600 if planned else 7200,teardown_reserve_seconds=360)
    controller=fleet.Fleet.__new__(fleet.Fleet)
    controller.budget=budget
    controller.root=Path(tmp_path)
    calls=[]
    controller.launch=lambda actor,phase,args,timeout:calls.append((phase,timeout))
    actor=SimpleNamespace(session='owned',task=dict(plan=dict(jobs=[dict(steps=100000)])),
                          exec_deadline=0.,snapshot_due=0.)
    monkeypatch.setattr(fleet.time,'time',lambda:0.)
    monkeypatch.setattr(fleet.time,'monotonic',lambda:100.)
    controller.execute(actor)
    assert calls==[('exec',expected_timeout)]
    assert actor.exec_deadline==100.+expected_timeout


def test_long_plan_requires_persisted_account_protection_before_begin(tmp_path):
    budget=FleetBudget(tmp_path/'ledger.json',allocation_limit_seconds=21600)
    with pytest.raises(BudgetStop,match='account floor150'):
        budget.start('fixture',Usage(350.,0.,0),1,allocation_plan_seconds=[21600])
    assert not budget.ledger.path.exists()
