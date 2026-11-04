"""Unregistered, fixed TRPO/PPO constructors for engineering acceptance only."""

import json
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box

from rl_stocks.core.contracts import digest

PROTOCOL_SHA256 = "ba19cae92e039c0f6d86e01d8309203899835596daefa6abc1ea4245b706c7ab"


def protocol():
    path = Path(__file__).resolve().parents[3] / "configs/experiments/trpo_feasibility.json"
    value = json.loads(path.read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError("fixed TRPO feasibility protocol required")
    return value


def create(algorithm, env, *, seed=11, smoke=False):
    if (
        algorithm not in ("ppo", "trpo")
        or type(seed) is not int
        or seed not in (11, 23, 37)
        or type(smoke) is not bool
    ):
        raise ValueError("fixed engineering algorithm/seed/mode required")
    obs, action = env.observation_space, env.action_space
    if (
        not isinstance(obs, Box)
        or obs.shape != (21,)
        or obs.dtype != np.float32
        or not isinstance(action, Box)
        or action.shape != (4,)
        or action.dtype != np.float32
        or not np.array_equal(action.low, np.full(4, -1.0))
        or not np.array_equal(action.high, np.ones(4))
    ):
        raise ValueError("full21-field observation and four-component action contract required")
    from rl_stocks.training.dependencies import verify_environment

    verify_environment()
    import torch
    from stable_baselines3 import PPO
    from sb3_contrib import TRPO

    spec = protocol()
    params = dict(spec["shared_parameters"])
    params.update(spec["ppo_parameters" if algorithm == "ppo" else "trpo_parameters"])
    if smoke:
        params.update(n_steps=spec["smoke"]["n_steps"], batch_size=spec["smoke"]["batch_size"])
    policy = dict(
        net_arch={"pi": [64, 64], "vf": [64, 64]},
        activation_fn=torch.nn.Tanh,
        ortho_init=True,
        log_std_init=0.0,
    )
    cls = PPO if algorithm == "ppo" else TRPO
    return cls("MlpPolicy", env, seed=seed, device="cpu", policy_kwargs=policy, **params)


def initial_parity(control, candidate):
    import torch
    from stable_baselines3 import PPO
    from sb3_contrib import TRPO

    if type(control) is not PPO or type(candidate) is not TRPO:
        raise ValueError("exact native PPO and TRPO required")
    if (
        control.num_timesteps != 0
        or candidate.num_timesteps != 0
        or control.policy.optimizer.state
        or candidate.policy.optimizer.state
    ):
        raise ValueError("untrained states required")
    left, right = control.policy.state_dict(), candidate.policy.state_dict()
    if set(left) != set(right) or any(not torch.equal(left[k], right[k]) for k in left):
        raise ValueError("complete common actor/value/log-std initialization required")
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in left.values()):
        raise ValueError("finite float32 policy required")
    return {
        "arrays": len(left),
        "parameters": sum(p.numel() for p in candidate.policy.parameters()),
        "all_initial_arrays_equal": True,
        "training_performed": False,
    }
