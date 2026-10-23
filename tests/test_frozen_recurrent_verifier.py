"""Verifier must reject lost memory and caller-owned state mutation."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('frozen_verifier',Path(__file__).resolve().parents[1]/'scripts/verify_frozen_runs.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Model:
    def predict(self, observation, *, state, episode_start, deterministic):
        old=np.zeros((1,1,1)) if episode_start[0] else state[0]
        new=old+float(observation[0])
        return new.flatten(), (new,new.copy())


class Frozen:
    def predict_recurrent(self, observation, *, state, episode_start):
        return Model().predict(observation,state=state,episode_start=np.array([episode_start]),deterministic=True)


def test_sequence_checks_carry_and_independent_reset():
    left=right=None
    for i in range(3):
        action,left,right=module.check_recurrent_prediction(Model(),Frozen(),np.array([1.]),left,right,i==0)
        np.testing.assert_array_equal(action,[i+1.])
    action,_,_=module.check_recurrent_prediction(Model(),Frozen(),np.array([1.]),None,None,True)
    np.testing.assert_array_equal(action,[1.])


def test_lost_memory_rejected():
    class Wrong(Frozen):
        def predict_recurrent(self, observation, *, state, episode_start):
            return super().predict_recurrent(observation,state=None,episode_start=True)
    _,a,b=module.check_recurrent_prediction(Model(),Frozen(),np.array([1.]),None,None,True)
    with pytest.raises(AssertionError):module.check_recurrent_prediction(Model(),Wrong(),np.array([1.]),a,b,False)


def test_input_state_mutation_rejected():
    class Mutating(Frozen):
        def predict_recurrent(self, observation, *, state, episode_start):
            if state is not None:state[0][...] += 1
            return super().predict_recurrent(observation,state=state,episode_start=episode_start)
    _,a,b=module.check_recurrent_prediction(Model(),Frozen(),np.array([1.]),None,None,True)
    with pytest.raises(AssertionError):module.check_recurrent_prediction(Model(),Mutating(),np.array([1.]),a,b,False)


def test_market_mask_verifier_rejects_a_policy_that_accepts_nonzero_channels():
    class Wrong:
        market_context_mode='masked'
        def predict_recurrent(self,*args,**kwargs):return np.zeros(4),None
    with pytest.raises(ValueError,match='nonzero masked market input accepted'):
        module.check_market_mask(Wrong(),np.zeros(24),True)
    assert not module.check_market_mask(Frozen(),np.zeros(24),True)
