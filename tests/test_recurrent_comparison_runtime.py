"""Budget, fixed dispatch and hostile archive checks without any allocation."""
from copy import deepcopy
from io import BytesIO
import json
from types import SimpleNamespace
import tarfile

import pytest

from rl_stocks.runtime.colab.jobs import recurrent_comparison_profile_plan,validate_plan
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.runtime.colab.budget import BudgetStop
from rl_stocks.training.recurrent_comparison_runtime import shared_profile_budget,claim_profile


def test_profile_plan_is_fixed():
    plan=recurrent_comparison_profile_plan()
    validate_plan(plan)
    assert plan['version']==41 and plan['allocation_deadline_seconds']==900
    for key,value in [('steps',100000),('device','cuda'),('seed',23),('algorithm','sac')]:
        changed=deepcopy(plan)
        changed['jobs'][0][key]=value
        with pytest.raises(ValueError):
            validate_plan(changed)


def test_original_study_baseline_floor_claim_and_stop_preserved(tmp_path):
    ledger=SimpleNamespace(path=tmp_path/'usage-ledger.json',value={'consumed':39.54})
    stop=tmp_path/'crossq-integrity-stop.json'
    stop.write_text('preserve historical stop')
    assert shared_profile_budget(ledger)==(.1,39.54)
    ledger.value['consumed']=39.58
    assert shared_profile_budget(ledger)==(.1,39.54)
    with pytest.raises(ValueError):
        shared_profile_budget(ledger,requested_start=39.58)
    with pytest.raises(ValueError):
        shared_profile_budget(ledger,requested_limit=2)
    baseline=json.loads((tmp_path/'recurrent-comparison-budget.json').read_text())
    assert baseline['max_study_incremental_units']==2 and baseline['minimum_account_balance']==150
    ledger.value['consumed']=39.65
    with pytest.raises(BudgetStop):
        shared_profile_budget(ledger)
    claim_profile(ledger,'rl-stocks-cpu-test')
    with pytest.raises(RuntimeError,match='no automatic retry'):
        shared_profile_budget(ledger)
    with pytest.raises(RuntimeError):
        claim_profile(ledger,'rl-stocks-cpu-again')
    assert stop.read_text()=='preserve historical stop'


def test_profile_fleet_enforces_single_allocation_and_shared_limit(tmp_path,monkeypatch):
    from rl_stocks.runtime.colab import fleet
    runtime=tmp_path/'.runtime'
    runtime.mkdir()
    (runtime/'usage-ledger.json').write_text(json.dumps(dict(ceiling=150.,working_limit=130.,
        consumed=39.54,attempts=[],budget_stopped=False,minimum_account_balance=150)))
    tasks=[dict(id='profile',dataset='unused',plan=recurrent_comparison_profile_plan())]
    monkeypatch.setattr(fleet,'validate_tasks',lambda tasks,root:tasks)
    with pytest.raises(ValueError,match='exactly one'):
        fleet.Fleet(tmp_path,object(),tasks,2,runtime/'state.json')
    controller=fleet.Fleet(tmp_path,object(),tasks,1,runtime/'state.json')
    assert controller.study_budget==dict(start_units=39.54,limit_units=.1)
    claim_profile(controller.budget.ledger,'rl-stocks-cpu-test')
    with pytest.raises(RuntimeError,match='no automatic retry'):
        fleet.Fleet(tmp_path,object(),tasks,1,runtime/'state2.json')


@pytest.mark.parametrize('kind',['missing','symlink','traversal','duplicate'])
def test_untrusted_archives_reject_before_model_load(tmp_path,kind):
    plan=recurrent_comparison_profile_plan()
    path=tmp_path/'bad.tar'
    with tarfile.open(path,'w') as archive:
        name='../escape' if kind=='traversal' else 'artifacts/completion.json'
        member=tarfile.TarInfo(name)
        if kind=='symlink':
            member.type=tarfile.SYMTYPE
            member.linkname='/etc/passwd'
            archive.addfile(member)
        else:
            member.size=2
            archive.addfile(member,BytesIO(b'{}'))
            if kind=='duplicate':
                archive.addfile(member,BytesIO(b'{}'))
    with pytest.raises(RuntimeError,match='bounded allowlisted'):
        validate_archive(path,plan,plan['dataset_contract']['identity'])
