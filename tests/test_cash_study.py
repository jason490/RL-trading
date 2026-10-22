"""No-fit cash protocol, job, executed-action and saved-record acceptance."""
from copy import deepcopy
import pytest
from test_hybrid_recurrent import Dummy
from rl_stocks.rl.registry.factory import create
from rl_stocks.runtime.colab.jobs import cash_plan,validate_plan
from rl_stocks.training.recurrent_cash import (protocol,arm_spec,overrides,modes,model_record,
    validate_record,validate_initial_pairs,identify_arm,validate_execution_counts,validate_execution_report)


@pytest.mark.parametrize('arm',protocol()['arms'])
@pytest.mark.parametrize('smoke',[True,False])
def test_exact_job_contract(arm,smoke):
    p=cash_plan(arm,smoke=smoke)
    assert validate_plan(p)==p
    assert p['account_contract']=={'minimum_balance':150,'teardown_balance':170}
    for mutate in [lambda p:p['jobs'][0].update(steps=250000),
                   lambda p:p['jobs'][0].update(algorithm='ppo'),
                   lambda p:p['jobs'][0]['overrides']['policy_kwargs'].update(cash_enabled=None),
                   lambda p:p['account_contract'].update(minimum_balance=0)]:
        bad=deepcopy(p);mutate(bad)
        with pytest.raises(ValueError):validate_plan(bad)


def test_actual_initial_models_and_paired_controls():
    records={}
    for arm in protocol()['arms']:
        c=arm_spec(arm);algo='hybrid_recurrent_ppo' if c['distribution']=='direction_protection' else 'recurrent_ppo'
        m=create(algo,Dummy(),smoke=True,overrides=overrides(arm,True))
        r=model_record(m);validate_record(r,arm,True,initial=True);records[arm]=r
        assert identify_arm(dict(algorithm=algo,action_execution=modes(arm),action_distribution=r['action_distribution']))==arm
        bad=deepcopy(r);bad['parameter_count']+=1
        with pytest.raises(ValueError):validate_record(bad,arm,True,initial=True)
        bad=deepcopy(r);bad.update(num_timesteps=512)
        with pytest.raises(ValueError):validate_record(bad,arm,True)
    assert validate_initial_pairs(records)
    records['categorical_cash']['policy_tensor_sha256']='0'*64
    with pytest.raises(ValueError):validate_initial_pairs(records)


def test_cash_counters_require_actual_both_sample_types_and_no_transform():
    c=dict(mode='unrestricted',transitions=512,requested_nonzero=340,processed_nonunit=0,sizing_changed=0,protective_changed=0)
    assert validate_execution_counts(c,'categorical_cash',512)==c
    for change in [dict(requested_nonzero=512),dict(requested_nonzero=0),dict(processed_nonunit=1),dict(sizing_changed=1),dict(protective_changed=1)]:
        with pytest.raises(ValueError):validate_execution_counts(dict(c,**change),'categorical_cash',512)
    b=dict(c,requested_nonzero=512)
    assert validate_execution_counts(b,'categorical_binary',512)==b
    with pytest.raises(ValueError):validate_execution_counts(c,'categorical_binary',512)
    r={'fixture':{'split':'validation','rows':[dict(raw_action=[0.,.2,.3,.4])]}}
    validate_execution_report(r,'categorical_cash')
    with pytest.raises(ValueError):validate_execution_report(r,'categorical_binary')
    r['fixture']['rows'][0]['raw_action'][0]=.2
    with pytest.raises(ValueError):validate_execution_report(r,'categorical_cash')


def test_screen_cannot_pass_only_binary_or_weak_gaussian_control():
    from rl_stocks.reporting.history import screen_gate
    def cell(score):
        row=dict(score=score,mean_max_drawdown=-.2,groups={
            'stock':dict(net_return=score,excess_return=0.,median_excess_return=0.),
            'etf':dict(net_return=score,excess_return=0.)})
        return dict(selected=deepcopy(row),final=deepcopy(row))
    cells={arm:cell(.2) for arm in protocol()['arms']}
    cells['categorical_cash']=cell(.24)
    assert screen_gate(cells,protocol())['passed']
    cells['posthoc_unit']=cell(.25)
    verdict=screen_gate(cells,protocol())
    assert not verdict['passed'] and not verdict['controls']['posthoc_unit']['passed']
    assert verdict['controls']['categorical_binary']['passed']
