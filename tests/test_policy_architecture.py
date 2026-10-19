from copy import deepcopy
from dataclasses import asdict
import json
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from test_causal_sampling import episode
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.rl.registry.factory import create,resolved_parameters
from rl_stocks.rl.ppo.architecture import architecture_spec,validate_loaded_architecture
from rl_stocks.training.initialization import initialize_policy


def environment():
    ep=episode();ep['features']=[[0.]*7 for _ in ep['bars']];names=[f'f{i}' for i in range(7)]
    return TradingEnv([ep],Normalizer(tuple(names),(0.,)*7,(1.,)*7),names,observation_profile='execution_full')


def model(variant):
    result=create('ppo',environment(),seed=11,smoke=True,policy_variant=variant)
    initialize_policy(result,'wide');return result


def test_masked_direction_ignores_account_inputs_but_risk_and_value_keep_access():
    masked=model('two_branch_market_exposure');control=model('two_branch_unmasked')
    obs=torch.randn(8,21);obs[:, :7]=obs[0,:7].clone()
    for candidate in (masked,control):
        candidate.policy.set_training_mode(False)
        latent_pi,latent_vf=candidate.policy.mlp_extractor(obs)
        actions=candidate.policy.action_net(latent_pi)
        if candidate is masked:torch.testing.assert_close(actions[:,0],actions[0,0].expand(8),rtol=0,atol=0)
        else:assert torch.std(actions[:,0])>0
        assert torch.std(actions[:,1:],dim=0).max()>0
        assert torch.std(candidate.policy.value_net(latent_vf))>0
    # Parameters and all initial tensors match exactly apart from the fixed mask.
    left,right=masked.policy.state_dict(),control.policy.state_dict()
    assert left.keys()==right.keys()
    for name in left:
        if name.endswith('connection_mask'):continue
        assert torch.equal(left[name],right[name]),name
    assert sum(p.numel() for p in masked.policy.parameters())==sum(p.numel() for p in control.policy.parameters())


def test_mask_has_zero_gradients_and_optimizer_owns_the_wrapped_parameters():
    candidate=model('two_branch_market_exposure');p=candidate.policy
    assert {id(x) for x in p.parameters()}=={id(x) for g in p.optimizer.param_groups for x in g['params']}
    obs=torch.randn(4,21,requires_grad=True)
    latent,_=p.mlp_extractor(obs);action=p.action_net(latent)
    action[:,0].sum().backward()
    assert torch.count_nonzero(obs.grad[:,7:])==0
    assert torch.count_nonzero(obs.grad[:,:7])>0
    assert torch.count_nonzero(p.action_net.weight.grad[0,p.mlp_extractor.branch_width:])==0
    assert 'connection_mask' not in dict(p.named_parameters())
    assert p.action_net.connection_mask.requires_grad is False


def test_native_factory_weights_and_rng_remain_bitwise_unchanged():
    env=environment();native=create('ppo',env,seed=23,smoke=True)
    rng=torch.get_rng_state().clone()
    expected=PPO('MlpPolicy',env,seed=23,device='cpu',verbose=0,**resolved_parameters('ppo',True))
    assert torch.equal(rng,torch.get_rng_state())
    for name,tensor in native.policy.state_dict().items():assert torch.equal(tensor,expected.policy.state_dict()[name])
    obs,_=env.reset(seed=11)
    np.testing.assert_array_equal(native.predict(obs,deterministic=True)[0],expected.predict(obs,deterministic=True)[0])


@pytest.mark.parametrize('variant',['native','two_branch_unmasked','two_branch_market_exposure'])
def test_save_load_freeze_parity_and_architecture_rejection(tmp_path,variant):
    from rl_stocks.inference.bundle import freeze,FrozenPolicy
    candidate=model(variant);env=candidate.get_env().envs[0].unwrapped
    observation,_=env.reset(seed=11);candidate.save(tmp_path/'weights')
    loaded=PPO.load(tmp_path/'weights.zip',device='cpu')
    spec=architecture_spec(variant,7);validate_loaded_architecture(loaded,spec,7)
    np.testing.assert_array_equal(candidate.predict(observation,deterministic=True)[0],loaded.predict(observation,deterministic=True)[0])
    dataset={'schema':'fixture','names':env.names,'normalizer':asdict(env.normalizer),'manifest':{'provenance':{'capabilities':{'feed':'fixture'}}}}
    metadata={'algorithm':'ppo','observation':env.observation_spec,'policy_architecture':spec}
    path=freeze(tmp_path/'weights.zip',dataset,metadata,tmp_path/'frozen');frozen=FrozenPolicy(path,'fixture','fixture')
    np.testing.assert_array_equal(candidate.predict(observation,deterministic=True)[0],frozen.predict(observation))
    if variant!='native':
        with pytest.raises(ValueError):validate_loaded_architecture(loaded,architecture_spec(),7)
        with pytest.raises(ValueError):validate_loaded_architecture(loaded,spec,8)
        loaded.policy.action_net.connection_mask[0,-1]=1-loaded.policy.action_net.connection_mask[0,-1]
        with pytest.raises(ValueError):validate_loaded_architecture(loaded,spec,7)


def test_branch_contract_rejects_wrong_algorithm_profile_or_override():
    for algorithm in ('sac','td3','recurrent_ppo'):
        with pytest.raises(ValueError):resolved_parameters(algorithm,True,policy_variant='two_branch_unmasked',market_features=7)
    with pytest.raises(ValueError):architecture_spec('unknown',7)
    with pytest.raises(ValueError):architecture_spec('two_branch_unmasked',True)
    with pytest.raises(ValueError):resolved_parameters('ppo',True,{'policy_kwargs':{'market_features':8}},'two_branch_unmasked',7)
    env=environment();env.observation_profile='legacy'
    with pytest.raises(ValueError):create('ppo',env,smoke=True,policy_variant='two_branch_unmasked')


def test_architecture_job_plans_lock_profile_budget_algorithm_and_variant():
    from rl_stocks.runtime.colab.jobs import architecture_plan,validate_plan
    for variant in ('native','two_branch_unmasked','two_branch_market_exposure'):
        for smoke in (True,False):
            plan=architecture_plan(variant,smoke=smoke);assert validate_plan(plan)==plan
            for key,value in (('steps',100000),('observation_profile','legacy'),('policy_variant','unknown'),('device','cuda')):
                changed=deepcopy(plan);changed['jobs'][0][key]=value
                with pytest.raises(ValueError):validate_plan(changed)


def test_architecture_promotion_requires_both_controls_without_changing_old_gate():
    import runpy
    from pathlib import Path
    gate=runpy.run_path('scripts/report_observations.py')['dual_screen']
    protocol=json.loads(Path('configs/experiments/exposure_architecture.json').read_text())
    def cell(score):
        point={'score':score,'mean_max_drawdown':-.2,'groups':{g:{'net_return':score} for g in ('stock','etf')}}
        return {'selected':deepcopy(point),'final':deepcopy(point)}
    cells={'native':cell(.20),'two_branch_unmasked':cell(.10),'two_branch_market_exposure':cell(.15)}
    assert not gate(cells,protocol)['passed']
    cells['two_branch_market_exposure']=cell(.23)
    assert gate(cells,protocol)['passed']
    cells['two_branch_market_exposure']['final']['groups']['etf']['net_return']=.19
    assert not gate(cells,protocol)['passed']
