import copy
import tempfile
from pathlib import Path
import unittest

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import torch

from rl_stocks.training.initialization import ARMS, initialize_policy, initialization_spec
from rl_stocks.runtime.colab.jobs import initialization_plan, validate_plan


class InitializationTests(unittest.TestCase):
    def test_native_control_is_bitwise_unchanged_and_does_not_advance_rng(self):
        model = PPO('MlpPolicy', gym.make('Pendulum-v1'), n_steps=8, batch_size=8, seed=11, device='cpu')
        weights = {k: v.clone() for k, v in model.policy.state_dict().items()}
        rng = torch.get_rng_state().clone()
        self.assertFalse(initialize_policy(model)['applied'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in model.policy.state_dict().items():
            self.assertTrue(torch.equal(value, weights[key]))
        model.get_env().close()

    def test_factorial_changes_only_distribution_initial_state_not_direction_or_weights(self):
        class FourActions(gym.Env):
            action_space = gym.spaces.Box(-1., 1., (4,), dtype=np.float32)
            observation_space = gym.spaces.Box(-1., 1., (12,), dtype=np.float32)
        torch.set_num_threads(1)
        for arm in ARMS[1:]:
            model = PPO('MlpPolicy', FourActions(), n_steps=8, batch_size=8, seed=11, device='cpu')
            before = {k: v.clone() for k, v in model.policy.state_dict().items()}
            initialize_policy(model, arm)
            for key, value in model.policy.state_dict().items():
                if key not in ('action_net.bias', 'log_std'):
                    self.assertTrue(torch.equal(value, before[key]), key)
            observation = np.zeros(12, dtype=np.float32)
            deterministic = model.predict(observation, deterministic=True)[0]
            self.assertEqual(deterministic[0], 0.)
            distribution = model.policy.get_distribution(torch.zeros((2000, 12))).get_actions()
            self.assertTrue((distribution[:, 0] < 0).any())
            self.assertTrue((distribution[:, 0] > 0).any())
            self.assertTrue(all(p.requires_grad for p in model.policy.parameters()))
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'policy.zip'
                model.save(target)
                loaded = PPO.load(target, device='cpu')
                np.testing.assert_array_equal(deterministic, loaded.predict(observation, deterministic=True)[0])
            model.get_env().close()

    def test_strict_bounded_plans_and_incompatible_algorithms(self):
        for arm in ARMS:
            for smoke in (True, False):
                plan = initialization_plan(arm, smoke=smoke)
                self.assertEqual(validate_plan(plan), plan)
                for key, value in [('steps', 250000), ('initialization', 'unknown'), ('device', 'cuda')]:
                    bad = copy.deepcopy(plan)
                    bad['jobs'][0][key] = value
                    with self.assertRaises(ValueError): validate_plan(bad)
        for arguments in [('wide', 'sac'), ('unknown', 'ppo')]:
            with self.assertRaises(ValueError): initialization_spec(*arguments)
        for seed in (True, 19):
            with self.assertRaises(ValueError): initialization_plan('wide', seed)
        with self.assertRaises(ValueError): initialization_plan('wide', 23, True)
