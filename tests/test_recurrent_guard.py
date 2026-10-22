from copy import deepcopy
import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure
from test_hybrid_recurrent import Dummy
from rl_stocks.rl.registry.factory import create
from rl_stocks.training.recurrent_guard import protocol,overrides,model_record,validate_record,modes


@pytest.mark.parametrize('smoke',[True,False])
def test_guard_changes_only_target_and_preserves_initial_tensors(smoke):
    records={}
    for arm in protocol()['arms']:
        m=create('recurrent_ppo',Dummy(),smoke=smoke,overrides=overrides(arm,smoke))
        records[arm]=model_record(m);validate_record(records[arm],arm,smoke,initial=True)
        bad=deepcopy(records[arm]);bad['target_kl']=.02
        with pytest.raises(ValueError):validate_record(bad,arm,smoke,initial=True)
    assert records['control']['policy_tensor_sha256']==records['kl_guard']['policy_tensor_sha256']
    c=overrides('control',smoke);g=overrides('kl_guard',smoke)
    assert c.pop('target_kl') is None and g.pop('target_kl')==.01 and c==g
    assert modes('control')==modes('kl_guard')


def test_library_high_kl_guard_stops_before_any_optimizer_step_without_fitting(monkeypatch):
    m=create('recurrent_ppo',Dummy(),smoke=True,overrides={'n_steps':8,'batch_size':8,'target_kl':.01})
    m.set_logger(configure(folder=None,format_strings=[]));before={k:v.clone() for k,v in m.policy.state_dict().items()}
    # Synthetic likelihood mismatch exercises the installed library's guard.
    # No market data and no actual optimizer call or parameter update is allowed.
    for i in range(8):
        m.rollout_buffer.add(np.zeros((1,21),dtype=np.float32),np.zeros((1,4)),np.zeros(1),
            np.array([i==0]),torch.zeros(1),torch.full((1,),100.),lstm_states=m._last_lstm_states)
    m.rollout_buffer.compute_returns_and_advantage(last_values=torch.zeros(1),dones=np.zeros(1))
    def forbidden():raise AssertionError('guard must stop before optimizer.step')
    monkeypatch.setattr(m.policy.optimizer,'step',forbidden)
    m.train()
    assert m.logger.name_to_value['train/approx_kl']>1.5*.01
    assert all(torch.equal(before[k],v) for k,v in m.policy.state_dict().items())
    assert not m.policy.optimizer.state and m.num_timesteps==0
    assert m._n_updates==m.n_epochs  # nominal epochs can advance with zero Adam steps


def test_fixed_guard_protocol_budget_and_unknown_arm_rejection():
    s=protocol()
    assert s['steps_per_cell']==100000 and s['screen_cells']==2
    assert s['execution']['minimum_account_balance']==150 and s['execution']['account_teardown_balance']==170
    assert s['execution']['max_study_incremental_units']==2
    for call in [lambda:overrides('categorical_cash'),lambda:overrides('control',1)]:
        with pytest.raises(ValueError):call()


def test_actual_step_contract_rejects_nominal_counter_substitution():
    # Static contract fixture, not evidence of actual learning.
    for arm in protocol()['arms']:
        m=create('recurrent_ppo',Dummy(),smoke=True,overrides=overrides(arm,True))
        r=model_record(m);r['num_timesteps']=512;o=r['optimizer']
        o.update(num_timesteps=512,sb3_nominal_epoch_counter=40,optimizer_state_count=21,
                 adam_step_min=80,adam_step_max=80,per_parameter_adam_steps={str(i):80 for i in range(21)})
        validate_record(r,arm,True)
        bad=deepcopy(r);bad['optimizer']['adam_step_max']=81
        with pytest.raises(ValueError):validate_record(bad,arm,True)
        bad=deepcopy(r);bad['optimizer']['adam_step_min']=0
        with pytest.raises(ValueError):validate_record(bad,arm,True)
        if arm=='control':
            bad=deepcopy(r);bad['optimizer'].update(adam_step_min=40,adam_step_max=40,per_parameter_adam_steps={str(i):40 for i in range(21)})
            with pytest.raises(ValueError):validate_record(bad,arm,True)


def test_activation_requires_actual_step_reduction_and_does_not_promote():
    from rl_stocks.training.recurrent_guard import activation
    records={}
    for arm,steps in [('control',20000),('kl_guard',12000)]:
        m=create('recurrent_ppo',Dummy(),smoke=False,overrides=overrides(arm))
        r=model_record(m);r['num_timesteps']=100000
        r['optimizer'].update(num_timesteps=100000,sb3_nominal_epoch_counter=1000,optimizer_state_count=21,
            adam_step_min=steps,adam_step_max=steps,per_parameter_adam_steps={str(i):steps for i in range(21)})
        records[arm]=r
    assert activation(records['control'],records['kl_guard'])['guard_activated']
    equal=deepcopy(records['kl_guard']);equal['optimizer'].update(adam_step_min=20000,adam_step_max=20000,
        per_parameter_adam_steps={str(i):20000 for i in range(21)})
    assert activation(records['control'],equal)['guard_activated'] is False


def test_performance_pass_cannot_promote_without_actual_guard_activation():
    from rl_stocks.training.recurrent_guard import screen_verdict
    cells={}
    for arm,score in [('control',.32),('kl_guard',.36)]:
        m=create('recurrent_ppo',Dummy(),smoke=False,overrides=overrides(arm))
        r=model_record(m);r['num_timesteps']=100000
        r['optimizer'].update(num_timesteps=100000,sb3_nominal_epoch_counter=1000,optimizer_state_count=21,
            adam_step_min=20000,adam_step_max=20000,per_parameter_adam_steps={str(i):20000 for i in range(21)})
        row=dict(score=score,mean_max_drawdown=-.2,groups={g:dict(net_return=score,excess_return=.01,median_excess_return=.01) for g in ['stock','etf']})
        cells[arm]=dict(selected=deepcopy(row),final=deepcopy(row),final_model_record=r)
    result=screen_verdict(cells)
    assert result['performance_gates_passed'] and not result['passed'] and result['promoted_candidate'] is None
    cells['kl_guard']['final_model_record']['optimizer'].update(adam_step_min=12000,adam_step_max=12000,
        per_parameter_adam_steps={str(i):12000 for i in range(21)})
    result=screen_verdict(cells)
    assert result['passed'] and result['promoted_candidate']=='kl_guard' and not result['objective_achieved']
