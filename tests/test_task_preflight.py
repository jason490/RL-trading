"""The real-data local preflight must prohibit optimization and restore APIs."""
import importlib.util
from pathlib import Path

import pytest
from stable_baselines3 import SAC
import torch


def test_preflight_guard_blocks_all_optimization_and_restores_on_failure():
    spec = importlib.util.spec_from_file_location(
        'task_preflight', Path(__file__).resolve().parents[1] / 'scripts/check_task_real_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = SAC.learn
    with pytest.raises(ValueError, match='interruption'):
        with module.prohibit_optimization():
            for method in (SAC.learn, SAC.train, torch.optim.Adam.step, torch.Tensor.backward):
                with pytest.raises(RuntimeError, match='optimization prohibited'):
                    method(None)
            raise ValueError('interruption')
    assert SAC.learn is original
