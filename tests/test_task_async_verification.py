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


def test_verification_math_mode_is_explicit_and_does_not_mutate_parent():
    from rl_stocks.runtime.colab.verify_task_screen_export import verification_environment
    parent = {'ATEN_CPU_CAPABILITY': 'avx512', 'MKL_NUM_THREADS': '8', 'custom': 'kept'}
    actual = verification_environment(parent)
    assert actual['ATEN_CPU_CAPABILITY'] == 'avx2'
    assert actual['MKL_CBWR'] == 'AVX2'
    assert actual['MKL_NUM_THREADS'] == actual['OMP_NUM_THREADS'] == '1'
    assert actual['custom'] == 'kept'
    assert parent['ATEN_CPU_CAPABILITY'] == 'avx512'
