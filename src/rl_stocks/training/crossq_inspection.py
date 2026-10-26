"""Untrained implementation audit; never calls learn, train or optimizer steps."""

import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import random
import tempfile

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sb3_contrib import CrossQ
from sb3_contrib.crossq.policies import CrossQPolicy
from sb3_contrib.common.torch_layers import BatchRenorm
from stable_baselines3 import SAC
import torch

from rl_stocks.core.contracts import digest

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SHA256 = "1e6311f784d511499a569c9308eb60fe85e01489fb578e27eede74f6c67e28c1"


def protocol():
    value = json.loads((ROOT / 'configs/experiments/crossq_implementation_audit.json').read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError('fixed untrained CrossQ inspection protocol required')
    return value


class SpacesOnly(gym.Env):
    def __init__(self, spec):
        self.observation_space = spaces.Box(-np.inf, np.inf, (spec['observation_size'],), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (spec['action_size'],), dtype=np.float32)

    def reset(self, **kwargs):
        raise RuntimeError('inspection must not collect transitions')

    def step(self, action):
        raise RuntimeError('inspection must not collect transitions')


def fingerprint(model):
    values = {k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
              for k, v in model.policy.state_dict().items()}
    return digest(values)


def _record(model):
    layers = {name: dict(features=layer.num_features, steps=int(layer.steps),
                        warmup_steps=layer.warmup_steps, eps=layer.eps, momentum=layer.momentum)
              for name, layer in model.policy.named_modules() if isinstance(layer, BatchRenorm)}
    optimizers = {}
    for name, optimizer in [('actor', model.actor.optimizer), ('critic', model.critic.optimizer),
                            ('entropy', model.ent_coef_optimizer)]:
        group = optimizer.param_groups[0]
        optimizers[name] = dict(state_count=len(optimizer.state), betas=list(group['betas']),
                                eps=group['eps'], learning_rate=group['lr'])
    return dict(actor_parameters=sum(p.numel() for p in model.actor.parameters()),
                critic_parameters=sum(p.numel() for p in model.critic.parameters()),
                target_critic_present=hasattr(model, 'critic_target'),
                target_critic_parameters=sum(p.numel() for p in model.critic_target.parameters())
                    if hasattr(model, 'critic_target') else 0,
                normalization=layers, optimizers=optimizers,
                num_timesteps=model.num_timesteps, updates=model._n_updates,
                policy_tensor_and_buffer_sha256=fingerprint(model))


def audit():
    spec = protocol()
    versions = {name: importlib.metadata.version(name) for name in ['stable-baselines3', 'sb3-contrib', 'torch']}
    if versions['stable-baselines3'] != spec['installed_sb3'] or versions['sb3-contrib'] != spec['installed_sb3_contrib']:
        raise ValueError('review installed implementation before repeating audit')
    numpy_state, python_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[]):
        try:
            records = {}
            observations = np.random.default_rng(123).normal(size=(spec['prediction_observations'], spec['observation_size'])).astype(np.float32)
            for name, cls in [('sac', SAC), ('crossq', CrossQ)]:
                env = SpacesOnly(spec)
                model = cls('MlpPolicy', env, seed=spec['seed'], device='cpu',
                            policy_kwargs=dict(net_arch=spec['net_arch']), **spec['common_overrides'])
                before = _record(model)
                assert not any(row['state_count'] for row in before['optimizers'].values())
                assert before['num_timesteps'] == before['updates'] == 0
                predicted = model.predict(observations, deterministic=True)[0]
                assert predicted.shape == (spec['prediction_observations'], spec['action_size'])
                assert np.isfinite(predicted).all()
                assert _record(model) == before
                with tempfile.TemporaryDirectory(prefix='crossq-untrained-') as directory:
                    path = Path(directory) / 'model.zip'
                    replay = Path(directory) / 'replay.pkl'
                    model.save(path)
                    model.save_replay_buffer(replay)
                    loaded = cls.load(path, device='cpu')
                    loaded.load_replay_buffer(replay)
                    np.testing.assert_array_equal(predicted, loaded.predict(observations, deterministic=True)[0])
                    assert _record(loaded) == before
                    assert loaded.replay_buffer.pos == 0 and not loaded.replay_buffer.full
                records[name] = dict(**before, exact_untrained_prediction_parity=True,
                                     normalization_and_weights_unchanged=True, empty_replay_restored=True)
                env.close()
            defaults = inspect.signature(CrossQ.__init__).parameters
            source = {label: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                      for label, cls in [('algorithm', CrossQ), ('policy', CrossQPolicy), ('normalization', BatchRenorm)]}
            return dict(protocol_sha256=digest(spec), versions=versions, implementation_sha256=source,
                        crossq_defaults={k: defaults[k].default for k in ['learning_rate', 'policy_delay', 'learning_starts', 'buffer_size']},
                        records=records, original_untrained_models=2, restored_untrained_models=2,
                        transitions=0, optimizer_steps=0,
                        actual_learned_acceptance=False, hardware_throughput_profile=False,
                        provider_or_dataset_access=False, colab_units=0, performance_protocol_registered=False)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
