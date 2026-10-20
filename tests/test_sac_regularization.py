from copy import deepcopy
import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import SAC
from rl_stocks.training.sac_regularization import overrides,optimizer_record,validate_optimizer_record,protocol


class Spaces(gym.Env):
    observation_space=gym.spaces.Box(-np.inf,np.inf,(21,),dtype=np.float32)
    action_space=gym.spaces.Box(-1,1,(4,),dtype=np.float32)


def test_actual_optimizers_and_identical_initial_networks_without_fitting():
    records=[]
    for arm in protocol()['arms']:
        model=SAC('MlpPolicy',Spaces(),seed=11,device='cpu',**overrides(arm))
        r=optimizer_record(model);validate_optimizer_record(r,arm);records.append(r)
        assert not model.actor.optimizer.state and not model.critic.optimizer.state
        bad=deepcopy(r);bad['optimizers']['entropy']['weight_decay']=[.0001]
        with pytest.raises(ValueError):validate_optimizer_record(bad,arm)
    assert records[0]['initialization']==records[1]['initialization']
    assert records[0]['optimizers']['actor']!=records[1]['optimizers']['actor']


def test_fixed_arm_and_smoke_architecture():
    assert overrides('weight_decay',True)['policy_kwargs']['net_arch']==[32,32]
    with pytest.raises(ValueError):overrides('sweep')
    with pytest.raises(ValueError):overrides('control',1)


def test_save_load_preserves_actual_optimizer_settings_without_fitting(tmp_path):
    from rl_stocks.training.sac_regularization import validate_final_optimizer_record
    model=SAC('MlpPolicy',Spaces(),seed=11,device='cpu',**overrides('weight_decay'))
    before=optimizer_record(model);model.save(tmp_path/'policy')
    restored=SAC.load(tmp_path/'policy.zip',device='cpu')
    assert optimizer_record(restored)==before
    validate_final_optimizer_record(before,'weight_decay')
    before['initialization']['alpha']=float('nan')
    with pytest.raises(ValueError):validate_final_optimizer_record(before,'weight_decay')


def test_factory_construction_preserves_nested_job_overrides():
    from rl_stocks.rl.registry.factory import create,resolved_parameters
    requested=overrides('weight_decay',True);before=deepcopy(requested)
    model=create('sac',Spaces(),smoke=True,overrides=requested)
    assert requested==before
    assert 'use_sde' in model.policy_kwargs  # SB3 mutates its own private copy.
    assert 'use_sde' not in resolved_parameters('sac',True,requested)['policy_kwargs']
    resolved=resolved_parameters('sac',True,requested)
    resolved['policy_kwargs']['optimizer_kwargs']['weight_decay']=1.
    assert requested==before
