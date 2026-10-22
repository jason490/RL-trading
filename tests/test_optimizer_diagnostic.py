from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from rl_stocks.reporting.optimizer import optimizer_record,training_diagnostics


def fixture():
    p=nn.Linear(2,1);p.optimizer=torch.optim.Adam(p.parameters(),lr=.0003,eps=1e-5)
    for parameter in p.parameters():
        # Synthetic saved state only; no optimizer step or fitting.
        p.optimizer.state[parameter]=dict(step=torch.tensor(200.),exp_avg=torch.zeros_like(parameter),exp_avg_sq=torch.ones_like(parameter))
    return SimpleNamespace(policy=p,_n_updates=10,num_timesteps=1000,n_steps=1000,batch_size=50,n_epochs=10,target_kl=None)


def test_actual_adam_steps_are_distinct_from_sb3_epoch_counter_without_mutation():
    m=fixture();before=[p.clone() for p in m.policy.parameters()]
    x=optimizer_record(m)
    assert x['sb3_nominal_epoch_counter']==10 and x['adam_step_min']==x['adam_step_max']==200
    assert x['optimizer_state_count']==2 and x['all_parameters_and_moments_finite']
    assert all(torch.equal(a,b) for a,b in zip(before,m.policy.parameters()))
    m.policy.optimizer.state.clear();m._n_updates=0
    assert optimizer_record(m)['adam_step_min'] is None


@pytest.mark.parametrize('bad',['nan_parameter','nan_moment','negative_second','fractional_step','shape','foreign','missing_owner'])
def test_corrupt_optimizer_evidence_rejects(bad):
    m=fixture();p=next(m.policy.parameters());state=m.policy.optimizer.state[p]
    if bad=='nan_parameter':
        with torch.no_grad():p.flatten()[0]=float('nan')
    if bad=='nan_moment':state['exp_avg'].flatten()[0]=float('nan')
    if bad=='negative_second':state['exp_avg_sq'].flatten()[0]=-1
    if bad=='fractional_step':state['step']=torch.tensor(1.5)
    if bad=='shape':state['exp_avg']=torch.zeros(1)
    if bad=='foreign':m.policy.optimizer.state[nn.Parameter(torch.ones(1))]=dict(state)
    if bad=='missing_owner':m.policy.optimizer.param_groups[0]['params'].pop()
    with pytest.raises(ValueError):optimizer_record(m)


def point(step=10000,phase='on_step'):
    return dict(steps=step,phase=phase,score=.2,training={'train/approx_kl':.03,'train/clip_fraction':.25,'train/entropy_loss':-4.,'train/std':.8})


def test_last_postupdate_log_supersedes_onstep_and_ranges_are_descriptive():
    final=point(20000,'post_update');final['training']['train/approx_kl']=.01
    x=training_diagnostics([point(),point(20000),final])
    assert len(x['points'])==2 and x['points'][-1]['phase']=='post_update'
    assert x['ranges']['train/approx_kl']['median']==pytest.approx(.02)
    assert 'not all optimizer steps' in x['interpretation']


@pytest.mark.parametrize('field,value',[('train/approx_kl',float('nan')),('train/clip_fraction',1.2),('train/std',0.)])
def test_invalid_training_logs_reject(field,value):
    p=point();p['training'][field]=value
    with pytest.raises(ValueError):training_diagnostics([p])
