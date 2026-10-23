"""Read-only recurrent action-component crossover, for diagnostics only."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from rl_stocks.rl.registry.prediction import recurrent


class RecurrentCrossover:
    def __init__(self,models,direction,protection):
        if set(models)!={'selected','final'} or direction not in models or protection not in models:
            raise ValueError('fixed selected/final sources required')
        self.models=dict(models);self.direction=direction;self.protection=protection
        first=models['selected']
        for model in models.values():
            if (not recurrent(model) or model.observation_space.shape!=first.observation_space.shape
                or model.action_space.shape!=(4,) or not np.all(model.action_space.low==-1)
                or not np.all(model.action_space.high==1)):
                raise ValueError('compatible recurrent four-component policies required')
        self.observation_space=first.observation_space;self.action_space=first.action_space
        self.policy=SimpleNamespace(lstm_actor=True)  # PredictionStream marker, not a trainable policy.

    def predict(self,observation,state=None,episode_start=None,deterministic=True):
        if deterministic is not True:raise ValueError('deterministic crossover required')
        observation=np.asarray(observation,dtype=np.float32)
        if observation.shape!=self.observation_space.shape or not np.isfinite(observation).all():
            raise ValueError('finite compatible observation required')
        start=np.asarray([state is None] if episode_start is None else episode_start)
        if start.shape!=(1,) or start.dtype!=np.bool_:raise ValueError('single boolean episode-start required')
        if not start[0] and (not isinstance(state,dict) or set(state)!=set(self.models)):
            raise ValueError('independent carried recurrent states required')
        actions={};states={}
        for name,model in self.models.items():
            old=None if start[0] else deepcopy(state[name])
            action,new=model.predict(observation.copy(),state=old,episode_start=start.copy(),deterministic=True)
            action=np.asarray(action,dtype=np.float32)
            if action.shape!=(4,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):
                raise ValueError('four bounded finite model outputs required')
            if not isinstance(new,tuple) or len(new)!=2 or any(not np.isfinite(v).all() for v in new):
                raise ValueError('finite hidden/cell state pair required')
            actions[name]=action.copy();states[name]=deepcopy(new)
        result=actions[self.direction].copy();result[1:]=actions[self.protection][1:]
        return result,states


def require_native_parity(expected,actual,*,atol=1e-5,rtol=1e-9):
    """Counts/labels exact; declared finite-float tolerance only, including daily paths."""
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(expected)!=set(actual):raise ValueError('native control keys differ')
        for key in expected:require_native_parity(expected[key],actual[key],atol=atol,rtol=rtol)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(expected)!=len(actual):raise ValueError('native control lengths differ')
        for x,y in zip(expected,actual,strict=True):require_native_parity(x,y,atol=atol,rtol=rtol)
    elif isinstance(expected,float):
        if isinstance(actual,bool) or not isinstance(actual,(int,float)) or not np.isfinite([expected,actual]).all() or not np.isclose(expected,actual,atol=atol,rtol=rtol):
            raise ValueError('native control numeric path differs')
    elif type(expected) is not type(actual) or expected!=actual:
        raise ValueError('native control exact labels/counts differ')
