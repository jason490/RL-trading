from copy import deepcopy
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
from rl_stocks.inference.crossover import RecurrentCrossover,require_native_parity
from rl_stocks.rl.registry.prediction import PredictionStream


class Fake:
    observation_space=gym.spaces.Box(-np.inf,np.inf,(2,),dtype=np.float32)
    action_space=gym.spaces.Box(-1.,1.,(4,),dtype=np.float32)
    policy=SimpleNamespace(lstm_actor=True)
    def __init__(self,value,increment):self.value=np.array(value,dtype=np.float32);self.increment=increment;self.observations=[]
    def predict(self,obs,state=None,episode_start=None,deterministic=True):
        self.observations.append(obs.copy());obs[:]=999
        if state is None:state=(np.zeros((1,1,2)),np.zeros((1,1,2)))
        for v in state:v[:]+=self.increment
        return self.value.copy(),state


def members():return dict(selected=Fake([.1,.2,.3,.4],1),final=Fake([-.5,.6,.7,.8],10))


@pytest.mark.parametrize('direction,protection',[('selected','selected'),('selected','final'),('final','selected'),('final','final')])
def test_fixed_component_contract_shared_observation_independent_states(direction,protection):
    m=members();p=RecurrentCrossover(m,direction,protection);obs=np.array([1.,2.],np.float32)
    action,state=p.predict(obs,episode_start=np.array([True]));old=deepcopy(state)
    assert action[0]==m[direction].value[0];np.testing.assert_array_equal(action[1:],m[protection].value[1:])
    _,new=p.predict(obs,state=state,episode_start=np.array([False]))
    assert all(np.array_equal(state[k][i],old[k][i]) for k in state for i in range(2))
    assert new['selected'][0][0,0,0]==2 and new['final'][0][0,0,0]==20
    assert all(np.array_equal(v,obs) for model in m.values() for v in model.observations)
    _,reset=p.predict(obs,state=new,episode_start=np.array([True]));assert reset['selected'][0][0,0,0]==1
    stream=PredictionStream(p);np.testing.assert_array_equal(stream.predict(obs),action)


@pytest.mark.parametrize('bad',['stochastic','observation','state','action'])
def test_invalid_inference_contract_rejects(bad):
    m=members();p=RecurrentCrossover(m,'selected','final');kwargs={};obs=np.zeros(2,np.float32)
    if bad=='stochastic':kwargs['deterministic']=False
    if bad=='observation':obs[0]=float('nan')
    if bad=='state':kwargs['episode_start']=np.array([False]);kwargs['state']={}
    if bad=='action':m['final'].value[1]=2
    with pytest.raises(ValueError):p.predict(obs,**kwargs)


def test_native_paths_preserve_exact_labels_counts_and_declared_float_tolerance():
    r={'qty':1,'reason':'stop','rows':[{'equity':100000.,'action':.1}]};x=deepcopy(r)
    x['rows'][0]['equity']+=.00005;require_native_parity(r,x)
    for mutate in [lambda x:x.update(qty=1.0),lambda x:x.update(reason='target'),lambda x:x['rows'][0].update(action=.11),lambda x:x['rows'].append({})]:
        x=deepcopy(r);mutate(x)
        with pytest.raises(ValueError):require_native_parity(r,x)
