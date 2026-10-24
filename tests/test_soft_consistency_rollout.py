import importlib.util
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def module():
    spec = importlib.util.spec_from_file_location(
        'soft_consistency_worker', Path(__file__).parents[1] / 'scripts/diagnose_soft_consistency.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Env:
    i = 0
    episode_index = 0

    def __init__(self, terminated):
        self.terminated = terminated

    def reset(self, seed):
        self.steps = 0
        return np.zeros(1, dtype=np.float32), dict(symbol='SYN', session='2020-01-01')

    def step(self, action):
        self.steps += 1
        return (np.zeros(1, dtype=np.float32), self.steps,
                self.terminated and self.steps == 2, not self.terminated and self.steps == 2, {})


def model():
    def action_log_prob(obs):
        # Consume RNG to verify the worker restores it, even for stochastic actions.
        return torch.rand(1, 4), torch.tensor([-3.])
    return SimpleNamespace(actor=SimpleNamespace(action_log_prob=action_log_prob),
                           critic=lambda obs, action: (torch.tensor([[1.]]), torch.tensor([[2.]])),
                           critic_target=lambda obs, action: (torch.tensor([[4.]]), torch.tensor([[5.]])),
                           log_ent_coef=torch.tensor(np.log(0.2)), gamma=0.5)


@pytest.mark.parametrize('terminated,expected', [(False, 3.45), (True, 2.3)])
def test_worker_soft_tail_uses_target_critic_only_on_timeout_and_restores_rng(terminated, expected):
    m = module()
    before = torch.get_rng_state().clone()
    with torch.inference_mode():
        _, row = m.rollout(model(), Env(terminated), 11, 23, 2, time.monotonic() + 10)
    assert row['target'] == pytest.approx(expected)
    assert row['min_q_minus_soft_target'] == pytest.approx(1 - expected)
    assert row['terminated'] == terminated
    assert torch.equal(before, torch.get_rng_state())


def test_worker_deadline_stops_before_stepping_and_restores_rng():
    m = module()
    before = torch.get_rng_state().clone()
    with pytest.raises(TimeoutError):
        m.rollout(model(), Env(False), 11, 23, 2, time.monotonic() - 1)
    assert torch.equal(before, torch.get_rng_state())
