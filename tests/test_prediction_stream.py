import unittest
from types import SimpleNamespace
import numpy as np
from rl_stocks.rl.registry.prediction import PredictionStream


class StreamTests(unittest.TestCase):
    def test_memory_is_carried_only_inside_one_episode(self):
        class Model:
            policy=SimpleNamespace(lstm_actor=object())
            def predict(self,obs,state,episode_start,deterministic):
                old=0 if episode_start[0] else state
                new=old+obs
                return new,new
        model=Model();a=PredictionStream(model);b=PredictionStream(model)
        np.testing.assert_array_equal(a.predict(np.array([2.])),[2.])
        np.testing.assert_array_equal(a.predict(np.array([3.])),[5.])
        np.testing.assert_array_equal(b.predict(np.array([3.])),[3.])
        np.testing.assert_array_equal(PredictionStream(model).predict(np.array([2.])),[2.])
