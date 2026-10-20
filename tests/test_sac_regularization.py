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
