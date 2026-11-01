import pytest
from stable_baselines3 import SAC
import torch

from rl_stocks.runtime.colab.verify_task_screen_export import no_optimization


def test_async_verifier_blocks_fitting_and_restores_apis_after_failure():
    original = SAC.learn
    with pytest.raises(ValueError):
        with no_optimization():
            for method in (SAC.learn, SAC.train, torch.optim.Adam.step, torch.Tensor.backward):
                with pytest.raises(RuntimeError, match='cannot optimize'):
                    method(None)
            raise ValueError('interruption')
    assert SAC.learn is original
