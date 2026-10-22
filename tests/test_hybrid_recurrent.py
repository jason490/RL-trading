import numpy as np
import pytest
import torch
from gymnasium import Env, spaces
from sb3_contrib import RecurrentPPO
from rl_stocks.rl.recurrent_ppo.hybrid import DirectionProtectionDistribution, HybridRecurrentPolicy


@pytest.mark.parametrize('cash', [False, True])
def test_joint_likelihood_entropy_and_samples(cash):
    torch.manual_seed(11)
    d=DirectionProtectionDistribution(cash)
    params=torch.zeros(2000,6,requires_grad=True)
    d.proba_distribution(params,torch.zeros(3))
    a=d.sample()
    assert set(a[:,0].tolist())==({-1.,0.,1.} if cash else {-1.,1.})
    expected=d.distribution.log_prob(a[:,1:]).sum(-1)-np.log(3 if cash else 2)
    assert torch.allclose(d.log_prob(a),expected)
    assert torch.allclose(d.entropy(),d.distribution.entropy().sum(-1)+np.log(3 if cash else 2))
    # Verify differentiability of the likelihood used by PPO, without fitting.
    (-d.log_prob(a.detach()).mean()).backward()
    assert torch.isfinite(params.grad).all()
    if not cash:assert torch.count_nonzero(params.grad[:,1])==0
    a[0,0]=.5
    with pytest.raises(ValueError):d.log_prob(a)
    if not cash:
        a[0,0]=0
        with pytest.raises(ValueError):d.log_prob(a)


def test_deterministic_cash_and_binary_mask_preserve_protections():
    p=torch.tensor([[1.,4.,2.,.2,-.3,.7]])
    cash=DirectionProtectionDistribution(True).proba_distribution(p,torch.zeros(3))
    binary=DirectionProtectionDistribution(False).proba_distribution(p,torch.zeros(3))
    assert cash.mode()[0,0]==0 and binary.mode()[0,0]==1
    assert torch.equal(cash.mode()[...,1:],binary.mode()[...,1:])
    with pytest.raises(ValueError):DirectionProtectionDistribution(1)


class Dummy(Env):
    observation_space=spaces.Box(-1,1,(21,),dtype=np.float32)
    action_space=spaces.Box(-1,1,(4,),dtype=np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        return np.zeros(21,dtype=np.float32),{}
    def step(self,action):
        return np.zeros(21,dtype=np.float32),0.,False,False,{}


def model(cash):
    return RecurrentPPO(HybridRecurrentPolicy,Dummy(),seed=11,n_steps=128,batch_size=64,
        policy_kwargs=dict(cash_enabled=cash,net_arch=[32,32],lstm_hidden_size=32),device='cpu')


def test_matched_initialization_optimizer_ownership_and_recurrent_save_load(tmp_path):
    binary=model(False);cash=model(True)
    assert all(torch.equal(v,cash.policy.state_dict()[k]) for k,v in binary.policy.state_dict().items())
    ids={id(p) for g in cash.policy.optimizer.param_groups for p in g['params']}
    assert ids=={id(p) for p in cash.policy.parameters()}
    obs=np.linspace(-.5,.5,21,dtype=np.float32)
    action,state=cash.predict(obs,episode_start=np.array([True]),deterministic=True)
    cash.save(tmp_path/'cash')
    restored=RecurrentPPO.load(tmp_path/'cash',device='cpu')
    assert restored.policy.cash_enabled is True
    action2,state2=restored.predict(obs,episode_start=np.array([True]),deterministic=True)
    assert np.array_equal(action,action2)
    assert all(np.array_equal(a,b) for a,b in zip(state,state2))
    next_action,next_state=cash.predict(obs,state=state,episode_start=np.array([False]),deterministic=True)
    other_action,other_state=restored.predict(obs,state=state2,episode_start=np.array([False]),deterministic=True)
    assert np.array_equal(next_action,other_action)
    assert all(np.array_equal(a,b) for a,b in zip(next_state,other_state))


@pytest.mark.parametrize('cash',[False,True])
def test_actual_recurrent_policy_forward_and_evaluation_agree(cash):
    m=model(cash)
    obs=torch.zeros((1,21));starts=torch.ones(1)
    actions,values,log_prob,_=m.policy.forward(obs,m._last_lstm_states,starts)
    values2,log_prob2,entropy=m.policy.evaluate_actions(obs,actions,m._last_lstm_states,starts)
    assert torch.allclose(log_prob,log_prob2) and torch.allclose(values,values2)
    assert torch.isfinite(entropy).all()
    # Autograd only: no optimizer update or local learning.
    (-log_prob2.mean()+values2.square().mean()).backward()
    assert m.policy.action_net.weight.grad is not None
    assert torch.isfinite(m.policy.action_net.weight.grad).all()


def test_frozen_protocol_and_exact_arm_contracts():
    from rl_stocks.training.recurrent_cash import protocol,arm_spec,overrides
    s=protocol()
    assert len(s['arms'])==4 and s['screen_cells']==4
    assert s['candidate']=='categorical_cash'
    assert s['execution']['minimum_account_balance']==150
    assert s['execution']['account_teardown_balance']==170
    assert overrides('categorical_cash')['policy_kwargs']['cash_enabled'] is True
    assert overrides('categorical_binary',True)['policy_kwargs']['cash_enabled'] is False
    assert 'cash_enabled' not in overrides('posthoc_unit')['policy_kwargs']
    assert arm_spec('gaussian_unit')['training']=='unit_magnitude'
    for call in (lambda:arm_spec('adaptive'),lambda:overrides('categorical_cash',1)):
        with pytest.raises(ValueError):call()


@pytest.mark.parametrize('cash',[False,True])
def test_factory_and_frozen_distribution_contract(tmp_path,cash):
    from dataclasses import asdict
    from test_execution_contract import environment
    from rl_stocks.rl.registry.factory import create
    from rl_stocks.rl.registry.distribution import model_distribution,validate_model_distribution
    from rl_stocks.inference.bundle import freeze,FrozenPolicy
    env=environment('unrestricted');obs,_=env.reset(seed=11)
    m=create('hybrid_recurrent_ppo',env,seed=11,smoke=True,
             overrides={'policy_kwargs':dict(cash_enabled=cash,net_arch=[32,32],lstm_hidden_size=32)})
    spec=model_distribution(m)
    assert spec['cash_enabled'] is cash
    assert validate_model_distribution(m,spec,'hybrid_recurrent_ppo')==spec
    m.save(tmp_path/'policy')
    data=dict(schema='fixture',names=env.names,normalizer=asdict(env.normalizer),
              manifest={'provenance':{'capabilities':{'feed':'fixture'}}})
    metadata=dict(algorithm='hybrid_recurrent_ppo',action_distribution=spec)
    bundle=freeze(tmp_path/'policy.zip',data,metadata,tmp_path/'good')
    frozen=FrozenPolicy(bundle,'fixture','fixture')
    prior=state=None
    for i in range(3):
        action,state=m.predict(obs,state=state,episode_start=np.array([i==0]),deterministic=True)
        actual,prior=frozen.predict_recurrent(obs,state=prior,episode_start=i==0)
        assert np.array_equal(actual,action)
        assert all(np.array_equal(a,b) for a,b in zip(state,prior))
    # Re-freeze malformed metadata with valid checksums: semantic rejection must
    # work independently of file hash integrity.
    bad_specs=[None,dict(spec,cash_enabled=not cash),dict(spec,version=True),dict(spec,extra=1),
               dict(spec,directions=[-1,False,True] if cash else [-1,True])]
    for i,bad in enumerate(bad_specs):
        bad_meta=dict(metadata)
        if bad is None:bad_meta.pop('action_distribution')
        else:bad_meta['action_distribution']=bad
        invalid=freeze(tmp_path/'policy.zip',data,bad_meta,tmp_path/f'bad-{i}')
        with pytest.raises(ValueError):FrozenPolicy(invalid,'fixture','fixture')
    with pytest.raises(ValueError):validate_model_distribution(m,spec,'recurrent_ppo')
    assert m.num_timesteps==0 and not m.policy.optimizer.state


def test_factory_rejects_undeclared_hybrid_policy_configuration():
    from rl_stocks.rl.registry.factory import resolved_parameters
    for algo,over in [('hybrid_recurrent_ppo',None),
                      ('hybrid_recurrent_ppo',{'policy_kwargs':{'cash_enabled':1}}),
                      ('recurrent_ppo',{'policy_kwargs':{'cash_enabled':True}})]:
        with pytest.raises(ValueError):resolved_parameters(algo,smoke=True,overrides=over)


@pytest.mark.parametrize('cash',[False,True])
def test_real_recurrent_buffer_padding_has_finite_masked_likelihood_gradients(cash):
    m=model(cash);buffer=m.rollout_buffer
    # Unequal episode lengths create real library padding, including artificial
    # all-zero actions. Build a buffer only; no learn() or optimizer update.
    for i in range(m.n_steps):
        buffer.add(np.zeros((1,21),dtype=np.float32),np.array([[-1.,.2,.3,.4]],dtype=np.float32),
            np.zeros(1),np.array([i in (0,13,44,89)]),torch.zeros(1),torch.zeros(1),
            lstm_states=m._last_lstm_states)
    buffer.compute_returns_and_advantage(last_values=torch.zeros(1),dones=np.zeros(1))
    padded=0
    for batch in buffer.get(64):
        original=batch.actions.clone();mask=batch.mask>1e-8
        padded+=int((~mask).sum())
        assert torch.all(batch.actions[~mask]==0)
        values,likelihood,entropy=m.policy.evaluate_actions(batch.observations,batch.actions,
            batch.lstm_states,batch.episode_starts)
        assert torch.isfinite(likelihood).all() and torch.isfinite(entropy).all()
        assert torch.equal(original,batch.actions)
        loss=(-likelihood[mask].mean()-entropy[mask].mean()+values.flatten()[mask].square().mean())
        m.policy.zero_grad();loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.policy.parameters() if p.grad is not None)
    assert padded>0 and m.num_timesteps==0 and not m.policy.optimizer.state
    if not cash:
        # Zero direction with real protective outputs is not a padding row.
        with pytest.raises(ValueError):m.policy.evaluate_actions(torch.zeros((1,21)),
            torch.tensor([[0.,.2,.3,.4]]),m._last_lstm_states,torch.ones(1))
