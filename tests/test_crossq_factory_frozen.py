from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
from sb3_contrib import CrossQ
import torch

from test_policy_architecture import environment
from rl_stocks.inference.bundle import freeze, FrozenPolicy
from rl_stocks.rl.crossq.geometry import model_spec, normalization_record
from rl_stocks.rl.registry.factory import create, resolved_parameters
from rl_stocks.rl.registry.distribution import model_distribution
from rl_stocks.training.crossq_spec import parameters
from rl_stocks.training.crossq_assets import validate_assets


@pytest.fixture(autouse=True)
def forbid_learning(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail('local optimization attempted')
    monkeypatch.setattr(CrossQ, 'learn', forbidden)
    monkeypatch.setattr(CrossQ, 'train', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)


def candidate():
    return create('crossq', environment(), seed=11, smoke=True)


def bundle(root, mutate=None):
    model = candidate()
    # Manual positive counters/moments cover saved inference state without fitting.
    from rl_stocks.rl.crossq.geometry import layers
    for layer in layers(model).values():
        layer.steps.fill_(1344)
        layer.ra_mean.fill_(0.01)
        layer.ra_var.fill_(1.1)
    env = environment()
    payload = dict(schema='crossq-fixture', names=env.names, normalizer=asdict(env.normalizer),
                   manifest={'provenance': {'capabilities': {'feed': 'fixture'}}})
    metadata = dict(algorithm='crossq', observation=env.observation_spec,
                    action_distribution=model_distribution(model), crossq_geometry=model_spec(model),
                    crossq_normalization=normalization_record(model))
    if mutate:
        mutate(metadata)
    path = root / 'owned-model.zip'
    model.save(path)
    return model, freeze(path, payload, metadata, root / 'frozen')


def test_factory_keeps_full_geometry_and_rejects_unregistered_settings():
    assert resolved_parameters('crossq', True) == parameters('crossq', True)
    assert model_spec(candidate())['normalization_layers'] == 9
    for overrides in [dict(policy_delay=1), dict(learning_rate=0.001),
                      dict(policy_kwargs={'net_arch': [32, 32]})]:
        with pytest.raises(ValueError, match='preregistered'):
            resolved_parameters('crossq', True, overrides)
    # Current registry changes cannot invalidate frozen original generation snapshots.
    assert validate_assets()['passed']


def test_actual_frozen_parity_preserves_parameters_buffers_and_inputs(tmp_path):
    model, path = bundle(tmp_path)
    frozen = FrozenPolicy(path, 'crossq-fixture', 'fixture')
    before = deepcopy(frozen._model.policy.state_dict())
    normalizer = deepcopy(frozen.normalizer)
    for seed in range(5):
        observation = np.random.default_rng(seed).normal(size=21).astype(np.float32)
        original = observation.copy()
        expected = model.predict(observation, deterministic=True)[0]
        np.testing.assert_array_equal(frozen.predict(observation), expected)
        np.testing.assert_array_equal(observation, original)
    assert all(torch.equal(v, before[k]) for k, v in frozen._model.policy.state_dict().items())
    assert frozen.normalizer == normalizer
    for bad in [np.zeros(20), np.full(21, np.nan)]:
        with pytest.raises(ValueError):
            frozen.predict(bad)


@pytest.mark.parametrize('change', ['geometry', 'normalization', 'missing'])
def test_rehashed_bundle_cannot_hide_wrong_normalization_contract(tmp_path, change):
    def mutate(metadata):
        if change == 'geometry':
            metadata['crossq_geometry']['renorm_warmup_steps'] = 100000
        elif change == 'normalization':
            first = next(iter(metadata['crossq_normalization']))
            metadata['crossq_normalization'][first]['steps'] += 1
        else:
            metadata.pop('crossq_normalization')
    _, path = bundle(tmp_path, mutate)
    with pytest.raises(ValueError, match='CrossQ'):
        FrozenPolicy(path, 'crossq-fixture', 'fixture')


def test_mutating_prediction_is_rejected_and_running_state_restored(tmp_path, monkeypatch):
    _, path = bundle(tmp_path)
    frozen = FrozenPolicy(path, 'crossq-fixture', 'fixture')
    before = normalization_record(frozen._model)
    original = frozen._model.predict
    def mutate(*a, **kw):
        frozen._model.actor.latent_pi[0].ra_mean.add_(1)
        return original(*a, **kw)
    monkeypatch.setattr(frozen._model, 'predict', mutate)
    with pytest.raises(ValueError, match='mutated'):
        frozen.predict(np.zeros(21))
    assert normalization_record(frozen._model) == before


def test_generic_training_stays_blocked_before_environment_or_model(monkeypatch):
    from rl_stocks.training import runner
    monkeypatch.setattr(runner, 'read_dataset', lambda *a: pytest.fail('read before guard'))
    with pytest.raises(ValueError, match='remains disabled'):
        runner.train('missing', 'crossq')


def test_public_manifest_edit_cannot_rebind_frozen_statistics(tmp_path):
    _, path = bundle(tmp_path)
    frozen = FrozenPolicy(path, 'crossq-fixture', 'fixture')
    first = next(iter(frozen.manifest['metadata']['crossq_normalization']))
    frozen.manifest['metadata']['crossq_normalization'][first]['steps'] += 1
    # Verified expectation was captured at load, independently of mutable display metadata.
    action = frozen.predict(np.zeros(21))
    assert action.shape == (4,) and np.isfinite(action).all()
