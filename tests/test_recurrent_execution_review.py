"""Fixture-only authorization, original-budget preservation and no automatic retry."""
from copy import deepcopy
import hashlib
import json

import pytest

from rl_stocks.core.contracts import canonical
from rl_stocks.runtime.colab.budget import Ledger,Usage,BudgetStop
from rl_stocks.runtime.colab.fleet_budget import FleetBudget
from rl_stocks.runtime.colab.jobs import recurrent_reviewed_profile_plan,validate_plan
from rl_stocks.training import recurrent_execution_review as review


def fixture(tmp_path,monkeypatch):
    spec=deepcopy(review.review_spec())
    old_claim=b'fixture immutable original claim'
    old_stop=b'fixture immutable CrossQ stop'
    spec['original_claim_sha256']=hashlib.sha256(old_claim).hexdigest()
    spec['original_crossq_stop_sha256']=hashlib.sha256(old_stop).hexdigest()
    monkeypatch.setattr(review,'review_spec',lambda:spec)
    (tmp_path/'recurrent-comparison-profile-allocation.json').write_bytes(old_claim)
    (tmp_path/'crossq-integrity-stop.json').write_bytes(old_stop)
    (tmp_path/'recurrent-comparison-budget.json').write_bytes(canonical(dict(
        protocol_sha256=spec['research_protocol_sha256'],start_units=spec['original_study_start_units'],
        max_study_incremental_units=2,minimum_account_balance=150)))
    value=dict(ceiling=150.,working_limit=130.,consumed=39.55,budget_stopped=True,minimum_account_balance=150,
        attempts=[dict(session=spec['original_stopped_fleet'],status='stopped',stop_reason='original deadline',
            children={spec['original_child_session']:dict(status='stopped',cleanup_verified=True)})])
    path=tmp_path/'usage-ledger.json'
    path.write_text(json.dumps(value))
    return Ledger(path,clock=lambda:100.),spec


def test_fixed_review_plan_rejects_mutated_deadline_role_or_hardware():
    value=recurrent_reviewed_profile_plan()
    validate_plan(value)
    assert value['allocation_deadline_seconds']==1200 and value['accelerator']=='CPU'
    for key,changed in [('allocation_deadline_seconds',7200),('accelerator','T4'),('execution_review_sha256','0'*64)]:
        bad=deepcopy(value)
        bad[key]=changed
        with pytest.raises(ValueError):
            validate_plan(bad)


def test_no_implicit_resume_then_explicit_fixture_review_preserves_originals(tmp_path,monkeypatch):
    ledger,spec=fixture(tmp_path,monkeypatch)
    original_attempt=deepcopy(ledger.active)
    with pytest.raises(BudgetStop,match='explicit reviewed'):
        review.shared_reviewed_profile_budget(ledger)
    with pytest.raises(BudgetStop,match='authorization'):
        review.authorize_profile_review(ledger,Usage(348.58,0.,0),'')
    assert ledger.value['budget_stopped']
    review.authorize_profile_review(ledger,Usage(348.58,0.,0),'fixture explicit user approval')
    assert ledger.active==original_attempt and ledger.value['consumed']==39.55
    assert ledger.value['minimum_account_balance']==150 and not ledger.value['budget_stopped']
    assert review.shared_reviewed_profile_budget(ledger)==(.1,spec['original_study_start_units'])
    assert (tmp_path/'crossq-integrity-stop.json').read_bytes()==b'fixture immutable CrossQ stop'
    with pytest.raises(ValueError):
        review.shared_reviewed_profile_budget(ledger,requested_start=39.55)
    with pytest.raises(ValueError):
        review.shared_reviewed_profile_budget(ledger,requested_limit=4)
    review.claim_corrected_profile(ledger,'rl-stocks-cpu-fixture')
    with pytest.raises(RuntimeError,match='no automatic retry'):
        review.shared_reviewed_profile_budget(ledger)


def test_review_record_publication_failure_keeps_budget_stopped(tmp_path,monkeypatch):
    ledger,_=fixture(tmp_path,monkeypatch)
    def failed(*args,**kwargs):
        raise OSError('fixture disk failure')
    monkeypatch.setattr(review,'write_once',failed)
    with pytest.raises(OSError):
        review.authorize_profile_review(ledger,Usage(348.58,0.,0),'fixture explicit user approval')
    assert ledger.value['budget_stopped'] and ledger.value['consumed']==39.55
    assert not ledger.value.get('resumption_reviews')
    assert not (tmp_path/'recurrent-comparison-profile-review-authorization.json').exists()


def test_corrected_profile_deadline_still_reserves420_seconds(tmp_path):
    now=[0.]
    budget=FleetBudget(tmp_path/'ledger.json',clock=lambda:now[0])
    budget.start('fixture',Usage(350.,0.,0),1)
    budget.admit('fixture-owned','CPU',allocation_seconds=1200,teardown_reserve_seconds=420)
    now[0]=779
    budget.guard('fixture-owned',1)
    now[0]=780
    with pytest.raises(BudgetStop):
        budget.guard('fixture-owned',1)


def test_fleet_cannot_construct_corrected_profile_before_explicit_review(tmp_path,monkeypatch):
    from rl_stocks.runtime.colab import fleet
    runtime=tmp_path/'.runtime'
    runtime.mkdir()
    ledger,spec=fixture(runtime,monkeypatch)
    tasks=[dict(id='corrected-profile',dataset='unused',plan=recurrent_reviewed_profile_plan())]
    monkeypatch.setattr(fleet,'validate_tasks',lambda tasks,root:tasks)
    with pytest.raises(BudgetStop,match='explicit reviewed'):
        fleet.Fleet(tmp_path,object(),tasks,1,runtime/'state.json')
    review.authorize_profile_review(ledger,Usage(348.58,0.,0),'fixture explicit user approval')
    controller=fleet.Fleet(tmp_path,object(),tasks,1,runtime/'state.json')
    assert controller.study_budget==dict(start_units=spec['original_study_start_units'],limit_units=.1)
    with pytest.raises(ValueError,match='exactly one'):
        fleet.Fleet(tmp_path,object(),tasks,2,runtime/'state2.json')


def test_approval_file_without_matching_ledger_review_cannot_admit(tmp_path,monkeypatch):
    ledger,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(BudgetStop,match='healthy idle'):
        review.authorize_profile_review(ledger,Usage(348.58,1.,1),'fixture explicit user approval')
    assert ledger.value['budget_stopped']
    assert (tmp_path/'recurrent-comparison-profile-review-authorization.json').exists()
    # Even an unrelated/manual clear of the global stop is not scoped approval.
    ledger.value['budget_stopped']=False
    with pytest.raises(ValueError,match='persisted original-ledger resumption'):
        review.shared_reviewed_profile_budget(ledger)
