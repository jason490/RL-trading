import json

from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3.common.buffers import RolloutBuffer

from rl_stocks.training import gae_spec
from rl_stocks.training.vix_spec import parameters as native_parameters


@pytest.mark.parametrize("smoke", [False, True])
def test_original_control_contract_and_one_candidate_difference(smoke):
    control = gae_spec.parameters("control", smoke)
    assert control == native_parameters(smoke)
    candidate = gae_spec.parameters("mc_trace", smoke)
    assert candidate["gae_lambda"] == 1.0
    candidate["gae_lambda"] = 0.95
    assert candidate == control


def test_protocol_mutation_and_unregistered_settings_reject(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        gae_spec.parameters("other")
    with pytest.raises(ValueError):
        gae_spec.parameters("control", 1)
    spec = gae_spec.protocol()
    spec["gae_lambda"]["mc_trace"] = 0.99
    path = tmp_path / "configs/experiments/gae_trace_preregistration.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(gae_spec, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        gae_spec.protocol()


def returns(values, lam):
    rewards = np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.01], dtype=np.float32)
    rewards[2] += 0.99 * 0.04  # First episode timeout reward already bootstrapped by SB3.
    buffer = RolloutBuffer(
        6,
        spaces.Box(-1, 1, (1,), dtype=np.float32),
        spaces.Box(-1, 1, (1,), dtype=np.float32),
        gamma=0.99,
        gae_lambda=lam,
        device="cpu",
    )
    for i in range(6):
        buffer.add(
            np.array([0.0]),
            np.array([0.0]),
            rewards[i],
            i in [0, 3],
            torch.tensor([values[i]]),
            torch.tensor([0.0]),
        )
    buffer.compute_returns_and_advantage(torch.tensor([0.03]), np.array([False]))
    return rewards, buffer.returns[:, 0], buffer.advantages[:, 0]


def test_lambda_one_telescopes_intermediate_values_and_respects_episode_boundaries():
    values = np.array([0.01, 0.05, -0.02, 0.07, -0.04, 0.02], dtype=np.float32)
    rewards, result, advantages = returns(values, 1.0)
    expected = []
    for i in range(6):
        end = 3 if i < 3 else 6
        expected.append(
            sum(0.99 ** (j - i) * float(rewards[j]) for j in range(i, end))
            + (0 if i < 3 else 0.99 ** (6 - i) * 0.03)
        )
    np.testing.assert_allclose(result, expected, atol=2e-8)
    _, changed, changed_advantages = returns(-values, 1.0)
    np.testing.assert_allclose(result, changed, atol=2e-8)
    np.testing.assert_allclose(advantages, result - values, atol=2e-8)
    assert not np.allclose(advantages, changed_advantages)
    _, control, _ = returns(values, 0.95)
    assert not np.allclose(control, result)
    assert gae_spec.protocol()["execution"]["minimum_account_balance"] == 150
