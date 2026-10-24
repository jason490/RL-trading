from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

from test_policy_architecture import environment
from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.inference.bundle import FrozenPolicy, freeze
from rl_stocks.rl.registry.distribution import model_distribution
from rl_stocks.rl.registry.factory import create, resolved_parameters
from rl_stocks.rl.tqc.geometry import model_spec


def candidate(smoke=True, dropped=2):
    return create('tqc', environment(), seed=11, smoke=smoke,
                  overrides={'top_quantiles_to_drop_per_net': dropped})


def test_quantile_heads_shapes_and_retained_targets():
    m = candidate()
    spec = model_spec(m)
    assert spec['n_critics'] == 2 and spec['n_quantiles'] == 25
    assert spec['retained_target_quantiles'] == 46
    with torch.no_grad():
        obs, action = torch.zeros(3, 21), torch.zeros(3, 4)
        assert m.critic(obs, action).shape == (3, 2, 25)
        assert m.critic_target(obs, action).shape == (3, 2, 25)
    assert model_distribution(m) == {'version': 1, 'kind': 'squashed_gaussian', 'components': 4}
    assert m._n_updates == 0 and m.num_timesteps == 0
    assert model_spec(candidate(dropped=0))['retained_target_quantiles'] == 50


@pytest.mark.parametrize('overrides', [
    {'top_quantiles_to_drop_per_net': -1}, {'top_quantiles_to_drop_per_net': 25},
    {'top_quantiles_to_drop_per_net': True}, {'policy_kwargs': {'n_critics': 0}},
    {'policy_kwargs': {'n_critics': True}}, {'policy_kwargs': {'n_quantiles': 0}},
    {'policy_kwargs': {'n_quantiles': 2}}, {'policy_kwargs': {'n_quantiles': False}},
    {'use_sde': True}, {'policy_kwargs': {'use_sde': True}},
])
def test_invalid_quantile_or_exploration_geometry_rejected(overrides):
    with pytest.raises(ValueError):
        resolved_parameters('tqc', overrides=overrides)


def test_policy_override_keeps_explicit_default_quantile_contract():
    params = resolved_parameters('tqc', overrides={'policy_kwargs': {'net_arch': [16, 16]}})
    assert params['policy_kwargs'] == {'n_critics': 2, 'n_quantiles': 25, 'net_arch': [16, 16]}


@pytest.mark.parametrize('smoke', [True, False])
def test_shared_actor_initialization_and_tqc_control_tensors_without_updates(smoke):
    sac = create('sac', environment(), seed=11, smoke=smoke)
    untruncated = candidate(smoke, 0)
    truncated = candidate(smoke, 2)
    assert fingerprint(sac.actor.state_dict()) == fingerprint(untruncated.actor.state_dict())
    assert fingerprint(untruncated.policy.state_dict()) == fingerprint(truncated.policy.state_dict())
    assert fingerprint(untruncated.actor.optimizer.state_dict()) == fingerprint(truncated.actor.optimizer.state_dict())
    assert fingerprint(untruncated.critic.optimizer.state_dict()) == fingerprint(truncated.critic.optimizer.state_dict())
    assert fingerprint(untruncated.ent_coef_optimizer.state_dict()) == fingerprint(truncated.ent_coef_optimizer.state_dict())
    assert all(m._n_updates == 0 for m in (sac, untruncated, truncated))


def test_native_sac_factory_weights_and_rng_unchanged():
    env = environment()
    native = create('sac', env, seed=23, smoke=True)
    rng = torch.get_rng_state().clone()
    original_params = dict(buffer_size=100000, batch_size=64, learning_starts=64,
                           train_freq=1, gradient_steps=1, tau=0.005, ent_coef='auto',
                           learning_rate=0.0003, gamma=0.99, policy_kwargs={'net_arch': [32, 32]})
    assert resolved_parameters('sac', True) == original_params
    expected = SAC('MlpPolicy', env, seed=23, device='cpu', verbose=0, **original_params)
    assert torch.equal(rng, torch.get_rng_state())
    assert fingerprint(native.policy.state_dict()) == fingerprint(expected.policy.state_dict())
    assert 'n_quantiles' not in resolved_parameters('sac')['policy_kwargs']


def payload_and_metadata(model):
    env = environment()
    payload = dict(schema='tqc-fixture', names=env.names, normalizer=asdict(env.normalizer),
                   manifest={'provenance': {'capabilities': {'feed': 'fixture'}}})
    metadata = dict(algorithm='tqc', observation=env.observation_spec,
                    action_distribution=model_distribution(model), critic_geometry=model_spec(model))
    return payload, metadata


def test_untrained_frozen_parity_and_immutable_normalization(tmp_path):
    m = candidate()
    m.save(tmp_path / 'model.zip')
    payload, metadata = payload_and_metadata(m)
    bundle = freeze(tmp_path / 'model.zip', payload, metadata, tmp_path / 'frozen')
    frozen = FrozenPolicy(bundle, payload['schema'], 'fixture')
    before = deepcopy(asdict(frozen.normalizer))
    state_before = fingerprint(frozen._model.policy.state_dict())
    rng = torch.get_rng_state().clone()
    observation, _ = environment().reset(seed=11)
    expected = m.predict(observation, deterministic=True)[0]
    for _ in range(3):
        np.testing.assert_array_equal(frozen.predict(observation), expected)
    assert asdict(frozen.normalizer) == before
    assert fingerprint(frozen._model.policy.state_dict()) == state_before
    assert torch.equal(rng, torch.get_rng_state()) and frozen._model._n_updates == 0
    assert frozen.critic_geometry == model_spec(m)
    with pytest.raises(ValueError, match='observation shape'):
        frozen.predict(np.zeros(20))


@pytest.mark.parametrize('field,value', [('n_quantiles', 24), ('retained_target_quantiles', 45), ('version', True)])
def test_internally_checksummed_wrong_critic_metadata_rejected(tmp_path, field, value):
    m = candidate()
    m.save(tmp_path / 'model.zip')
    payload, metadata = payload_and_metadata(m)
    metadata['critic_geometry'][field] = value
    bundle = freeze(tmp_path / 'model.zip', payload, metadata, tmp_path / 'frozen')
    with pytest.raises(ValueError, match='critic geometry'):
        FrozenPolicy(bundle, payload['schema'], 'fixture')


def test_missing_quantile_declaration_or_mislabeled_algorithm_rejected(tmp_path):
    m = candidate()
    m.save(tmp_path / 'model.zip')
    payload, metadata = payload_and_metadata(m)
    del metadata['critic_geometry']
    bundle = freeze(tmp_path / 'model.zip', payload, metadata, tmp_path / 'missing')
    with pytest.raises(ValueError, match='critic geometry'):
        FrozenPolicy(bundle, payload['schema'], 'fixture')
    metadata['algorithm'] = 'sac'
    bundle = freeze(tmp_path / 'model.zip', payload, metadata, tmp_path / 'mislabeled')
    with pytest.raises(ValueError, match='critic geometry'):
        FrozenPolicy(bundle, payload['schema'], 'fixture')


def test_actual_target_head_mismatch_rejected():
    m = candidate()
    m.critic_target.n_quantiles = 24
    with pytest.raises(ValueError, match='head geometry'):
        model_spec(m)


def test_nonempty_adam_serialization_fixture_without_any_optimizer_step(tmp_path):
    m = candidate()
    # Explicit serialization fixture, not learned optimizer acceptance.
    for optimizer in (m.actor.optimizer, m.critic.optimizer, m.ent_coef_optimizer):
        parameter = optimizer.param_groups[0]['params'][0]
        optimizer.state[parameter] = dict(step=torch.tensor(3.), exp_avg=torch.ones_like(parameter),
                                          exp_avg_sq=torch.full_like(parameter, 0.5))
    states = [fingerprint(o.state_dict()) for o in (m.actor.optimizer, m.critic.optimizer, m.ent_coef_optimizer)]
    m.save(tmp_path / 'fixture.zip')
    from sb3_contrib import TQC
    restored = TQC.load(tmp_path / 'fixture.zip', device='cpu')
    actual = [fingerprint(o.state_dict()) for o in
              (restored.actor.optimizer, restored.critic.optimizer, restored.ent_coef_optimizer)]
    assert actual == states and restored._n_updates == m._n_updates == 0


def test_full_sac_parameters_still_match_prior_frozen_state_protocol():
    path = Path(__file__).parents[1] / 'configs/experiments/sac_execution_state.json'
    original = json.loads(path.read_text())['parameters']
    original['policy_kwargs'] = {'net_arch': original.pop('net_arch')}
    assert resolved_parameters('sac') == original
