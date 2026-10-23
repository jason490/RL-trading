"""Actual distribution/rollout/save-load checks without optimizer steps or learning."""
from dataclasses import asdict
import math
import numpy as np
import pytest
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from rl_stocks.rl.registry.factory import create,resolved_parameters
from rl_stocks.rl.registry.distribution import model_distribution,validate_model_distribution
from rl_stocks.training.input_clipping import overrides,apply_initialization,model_record
from rl_stocks.reporting.optimizer import optimizer_fingerprint
from rl_stocks.data.generation.datasets import Normalizer,normalizer_from
from rl_stocks.features.registry.assemble import NAMES
from rl_stocks.environment.trading.env import TradingEnv
from test_exposure_replay import fixture


def environment():
    data=fixture();data['names']=list(NAMES);data['normalizer']=asdict(Normalizer(NAMES,(0.,)*7,(1.,)*7))
    for e in data['episodes'].values():e['features']=[[0.]*7 for _ in e['bars']]
    return TradingEnv(list(data['episodes'].values()),normalizer_from(data['normalizer']),data['names'],observation_profile='execution_masked'),data


def models():
    torch.set_num_threads(1);result={}
    for algorithm in ['recurrent_ppo','squashed_recurrent_ppo']:
        env,_=environment();m=create(algorithm,env,seed=11,device='cpu',smoke=True,overrides=overrides('control',True))
        apply_initialization(m,'control',11,True);result[algorithm]=m
    return result


def test_same_parameters_initial_tensors_rng_and_transformed_deterministic_mean():
    ms=models();a,b=ms.values()
    assert resolved_parameters('recurrent_ppo',True,overrides('control',True))==resolved_parameters('squashed_recurrent_ppo',True,overrides('control',True))
    assert model_record(a)['policy_tensor_sha256']==model_record(b)['policy_tensor_sha256']
    assert sum(p.numel() for p in b.policy.parameters())==61513
    obs=torch.zeros(1,21);state=tuple(torch.zeros(1,1,64) for _ in range(2));start=torch.ones(1)
    with torch.no_grad():
        da,_=a.policy.get_distribution(obs,state,start);db,_=b.policy.get_distribution(obs,state,start)
        torch.testing.assert_close(db.mode(),torch.tanh(da.mode()),rtol=0,atol=0)
        torch.testing.assert_close(db.distribution.stddev,da.distribution.stddev,rtol=0,atol=0)
    assert b.policy.squash_output and not b.use_sde


def test_actual_likelihood_jacobian_entropy_and_numerical_saturation():
    m=models()['squashed_recurrent_ppo'];dist=m.policy.action_dist
    means=torch.tensor([[.2,-.3,.4,-.5]]);log_std=torch.zeros(4)
    d=dist.proba_distribution(means,log_std);z=torch.tensor([[.1,-.2,.3,-.4]]);action=torch.tanh(z)
    expected=torch.distributions.Normal(means,torch.ones_like(means)).log_prob(z).sum(1)-torch.log(1-action.square()+1e-6).sum(1)
    torch.testing.assert_close(d.log_prob(action),expected)
    assert d.entropy() is None
    assert torch.isfinite(d.log_prob(torch.tensor([[1.,-1.,1.,-1.]]))).all()
    torch.manual_seed(11);sample=d.sample();assert sample.shape==(1,4) and bool((sample.abs()<=1).all())
    assert torch.isfinite(d.log_prob(sample)).all()


class Continue(BaseCallback):
    def _on_step(self):return True


def test_actual_sb3_rollout_buffer_likelihoods_match_recurrent_evaluation_without_updates():
    m=models()['squashed_recurrent_ppo'];before=model_record(m)['policy_tensor_sha256'];adam=optimizer_fingerprint(m)
    _,callback=m._setup_learn(128,callback=Continue());callback.on_training_start({}, {})
    assert m.collect_rollouts(m.env,callback,m.rollout_buffer,128)
    buffer=m.rollout_buffer
    assert np.isfinite(buffer.actions).all() and np.max(np.abs(buffer.actions))<=1
    with torch.no_grad():
        for batch in buffer.get(64):
            _,logs,entropy=m.policy.evaluate_actions(batch.observations,batch.actions,batch.lstm_states,batch.episode_starts)
            mask=batch.mask>1e-8
            torch.testing.assert_close(logs[mask],batch.old_log_prob[mask],rtol=1e-5,atol=1e-5)
            assert entropy is None and torch.isfinite(logs).all()
    assert m.num_timesteps==128 and not m.policy.optimizer.state
    assert model_record(m)['policy_tensor_sha256']==before and optimizer_fingerprint(m)==adam


def test_saved_model_and_complete_frozen_bundle_preserve_squashing_and_recurrent_states(tmp_path):
    from rl_stocks.inference.bundle import freeze,FrozenPolicy
    from rl_stocks.environment.observations.state import observation_spec
    from rl_stocks.risk.exposure import execution_contract
    m=models()['squashed_recurrent_ppo'];env,data=environment();m.save(tmp_path/'model')
    loaded=RecurrentPPO.load(tmp_path/'model.zip',device='cpu');assert model_record(loaded)==model_record(m)
    dist=model_distribution(m);assert validate_model_distribution(loaded,dist,'squashed_recurrent_ppo')==dist
    payload=dict(schema='fixture',names=data['names'],normalizer=data['normalizer'],manifest={'provenance':{'capabilities':{'feed':'fixture'}}})
    bundle=freeze(tmp_path/'model.zip',payload,dict(algorithm='squashed_recurrent_ppo',action_distribution=dist,observation=observation_spec('execution_masked'),action_execution=execution_contract()),tmp_path/'frozen')
    frozen=FrozenPolicy(bundle,'fixture','fixture');obs,_=env.reset(seed=11);state=fstate=None
    for i in range(2):
        a,state=loaded.predict(obs,state=state,episode_start=np.array([i==0]),deterministic=True)
        b,fstate=frozen.predict_recurrent(obs,state=fstate,episode_start=i==0)
        np.testing.assert_array_equal(a,b)
        for x,y in zip(state,fstate,strict=True):np.testing.assert_array_equal(x,y)
        obs,*_=env.step(np.zeros(4))
    for declared,algorithm in [(None,'squashed_recurrent_ppo'),(dist,'recurrent_ppo'),(dict(dist,jacobian_epsilon=1e-5),'squashed_recurrent_ppo')]:
        with pytest.raises(ValueError):validate_model_distribution(loaded,declared,algorithm)


@pytest.mark.parametrize('kwargs',[dict(use_sde=True),dict(squash_output=False)])
def test_unsupported_exploration_or_unsquashed_overrides_reject(kwargs):
    env,_=environment();params=overrides('control',True);params['policy_kwargs'].update(kwargs)
    with pytest.raises(ValueError):create('squashed_recurrent_ppo',env,smoke=True,overrides=params)
