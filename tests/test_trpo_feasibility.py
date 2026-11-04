import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import pytest
import torch

from rl_stocks.training.trpo_feasibility import create, initial_parity, protocol


class ContractEnv(gym.Env):
    observation_space = Box(-np.inf, np.inf, (21,), dtype=np.float32)
    action_space = Box(-1.0, 1.0, (4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(21, dtype=np.float32), {}

    def step(self, action):
        raise AssertionError("engineering constructor/inference must not collect or train")


@pytest.mark.parametrize("seed", [11, 23, 37])
def test_shared_initialization_and_save_load_prediction_without_training(
    tmp_path, seed, monkeypatch
):
    from stable_baselines3 import PPO
    from sb3_contrib import TRPO

    def forbidden(*a, **k):
        raise AssertionError("no local optimization")

    monkeypatch.setattr(PPO, "learn", forbidden)
    monkeypatch.setattr(TRPO, "learn", forbidden)
    control = create("ppo", ContractEnv(), seed=seed)
    candidate = create("trpo", ContractEnv(), seed=seed)
    proof = initial_parity(control, candidate)
    assert proof["all_initial_arrays_equal"] and proof["parameters"] == 11465
    assert candidate.n_steps == 1000 and candidate.batch_size == 50
    assert candidate.cg_max_steps == 15 and candidate.target_kl == 0.01
    x = np.random.default_rng(5).normal(size=(7, 21)).astype(np.float32)
    before = {k: v.clone() for k, v in candidate.policy.state_dict().items()}
    a, _ = candidate.predict(x, deterministic=True)
    b, _ = control.predict(x, deterministic=True)
    np.testing.assert_array_equal(a, b)
    path = tmp_path / "untrained.zip"
    candidate.save(path)
    restored = TRPO.load(path, device="cpu")
    np.testing.assert_array_equal(a, restored.predict(x, deterministic=True)[0])
    np.testing.assert_array_equal(a, candidate.predict(x, deterministic=True)[0])
    assert all(torch.equal(v, candidate.policy.state_dict()[k]) for k, v in before.items())
    assert candidate.num_timesteps == 0 and restored.num_timesteps == 0
    assert not candidate.policy.optimizer.state and not restored.policy.optimizer.state
    with torch.no_grad():
        candidate.policy.log_std[0] += 1
    with pytest.raises(ValueError, match="initialization"):
        initial_parity(control, candidate)


def test_fixed_contract_and_smoke_geometry_without_generic_registration():
    from rl_stocks.rl.registry.factory import CLASSES

    assert "trpo" not in CLASSES
    assert protocol()["colab_allocation_admitted"] is False
    candidate = create("trpo", ContractEnv(), smoke=True)
    assert candidate.n_steps == 128 and candidate.batch_size == 32
    for algo, seed, smoke in [
        ("sac", 11, False),
        ("trpo", True, False),
        ("trpo", 99, False),
        ("trpo", 11, 1),
    ]:
        with pytest.raises(ValueError):
            create(algo, ContractEnv(), seed=seed, smoke=smoke)
    env = ContractEnv()
    env.action_space = Box(-1.0, 1.0, (3,), dtype=np.float32)
    with pytest.raises(ValueError):
        create("trpo", env)
    env = ContractEnv()
    env.observation_space = Box(-np.inf, np.inf, (22,), dtype=np.float32)
    with pytest.raises(ValueError):
        create("trpo", env)
