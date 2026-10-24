"""Actual untrained off-policy tensor/RNG artifacts; never learn or optimizer.step."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from scripts.build_distributional_initializations import ModelSpaces
from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.rl.registry.factory import CLASSES, create
from rl_stocks.training import distributional_assets as assets
from rl_stocks.training.distributional_critics import (
    apply_initialization, identify_parameters, model_record, overrides, parsed_global_rng,
    protocol, rng_record, validate_initial_comparison,
)


def model(arm='tqc_truncated', seed=11, smoke=True):
    return create(protocol()['arms'][arm]['algorithm'], ModelSpaces(), seed=seed, smoke=smoke,
                  overrides=overrides(arm, smoke))


@pytest.mark.parametrize('seed,smoke', [(11, True), (11, False), (23, False), (37, False)])
def test_portable_native_actor_critics_and_three_rng_streams(seed, smoke):
    records, draws = {}, []
    for index, arm in enumerate(protocol()['arms']):
        m = model(arm, seed, smoke)
        owners = {name: [id(p) for group in opt.param_groups for p in group['params']]
                  for name, opt in [('actor', m.actor.optimizer), ('critic', m.critic.optimizer),
                                     ('entropy', m.ent_coef_optimizer)]}
        with torch.no_grad():
            for p in m.policy.parameters():
                p.fill_(0.123 * (index + 1))
        torch.manual_seed(900 + index)
        np.random.seed(901 + index)
        random.seed(902 + index)
        proof = apply_initialization(m, arm, seed, smoke)
        assert proof == assets.initialization_spec(arm, seed, smoke)
        records[arm] = model_record(m)
        assert records[arm]['rng'] == proof['global_rng_sha256']
        assert records[arm]['policy_tensor_sha256'] == proof['policy_tensor_sha256']
        for name, opt in [('actor', m.actor.optimizer), ('critic', m.critic.optimizer),
                          ('entropy', m.ent_coef_optimizer)]:
            assert owners[name] == [id(p) for group in opt.param_groups for p in group['params']]
            assert not opt.state
        draws.append((torch.randn(16), np.random.normal(size=16), [random.random() for _ in range(16)]))
    assert validate_initial_comparison(records, smoke)['sampling_rng_identical']
    for draw in draws[1:]:
        assert torch.equal(draw[0], draws[0][0])
        np.testing.assert_array_equal(draw[1], draws[0][1])
        assert draw[2] == draws[0][2]


@pytest.mark.parametrize('arm', ['sac_native', 'tqc_truncated'])
def test_actual_untrained_save_load_keeps_tensors_optimizers_and_initialization_proof(tmp_path, arm):
    m = model(arm)
    proof = apply_initialization(m, arm, smoke=True)
    before = model_record(m)
    m.save(tmp_path / 'initial.zip')
    restored = CLASSES[protocol()['arms'][arm]['algorithm']].load(tmp_path / 'initial.zip', device='cpu')
    after = model_record(restored)
    # Loading a checkpoint is not a promise to restore process-global sampling RNG.
    assert {k: v for k, v in before.items() if k != 'rng'} == {k: v for k, v in after.items() if k != 'rng'}
    assert restored._research_initialization == proof
    assert restored._n_updates == restored.num_timesteps == 0


@pytest.mark.parametrize('bad', ['actor', 'critic', 'entropy', 'replay', 'steps', 'updates',
                                  'seed', 'gamma', 'twice'])
def test_only_fresh_matching_offpolicy_models_can_restore_without_mutation(bad):
    m = model()
    if bad in ('actor', 'critic', 'entropy'):
        optimizer = m.ent_coef_optimizer if bad == 'entropy' else getattr(m, bad).optimizer
        optimizer.state[optimizer.param_groups[0]['params'][0]] = {'step': torch.tensor(1.)}
    if bad == 'replay':
        m.replay_buffer.pos = 1
    if bad == 'steps':
        m.num_timesteps = 1
    if bad == 'updates':
        m._n_updates = 1
    if bad == 'seed':
        m.seed = 23
    if bad == 'gamma':
        m.gamma = 0.95
    if bad == 'twice':
        apply_initialization(m, 'tqc_truncated', smoke=True)
    before, rng = fingerprint(m.policy.state_dict()), rng_record()
    with pytest.raises(ValueError):
        apply_initialization(m, 'tqc_truncated', smoke=True)
    assert fingerprint(m.policy.state_dict()) == before and rng_record() == rng


@pytest.mark.parametrize('bad', ['numpy_keys', 'numpy_gauss', 'python_cache'])
def test_bad_global_rng_rejected_before_tensor_or_global_mutation(monkeypatch, bad):
    entry, group = assets.initialization_entry('tqc_truncated')
    group = deepcopy(group)
    if bad == 'numpy_keys':
        group['numpy']['keys'][0] = True
    if bad == 'numpy_gauss':
        group['numpy']['has_gauss'] = True
    if bad == 'python_cache':
        group['python'][2] = 'invalid'
    with pytest.raises(ValueError):
        parsed_global_rng(group)
    monkeypatch.setattr(assets, 'initialization_entry', lambda *args: (entry, group))
    m = model()
    before, rng = fingerprint(m.policy.state_dict()), rng_record()
    with pytest.raises(ValueError):
        apply_initialization(m, 'tqc_truncated', smoke=True)
    assert fingerprint(m.policy.state_dict()) == before and rng_record() == rng


def test_locked_manifest_and_exact_parameter_arm_resolution():
    manifest = assets.manifest()
    assert len(manifest['artifacts']) == 12 and len(manifest['groups']) == 4
    for smoke in (True, False):
        for arm, spec in protocol()['arms'].items():
            params = overrides(arm, smoke)
            assert identify_parameters(params, spec['algorithm'], smoke) == arm
            if spec['algorithm'] == 'tqc':
                params['policy_kwargs']['n_quantiles'] = 24
            else:
                params['gamma'] = 0.95
            with pytest.raises(ValueError):
                identify_parameters(params, spec['algorithm'], smoke)
    with pytest.raises(ValueError):
        assets.initialization_entry('sac_native', 23, True)
    with pytest.raises(ValueError):
        assets.initialization_entry('sac_native', True, False)


@pytest.mark.parametrize('bad', ['episode_unit', 'decoupled_adam', 'scalar_critic_count'])
def test_actual_effective_optimizer_and_critic_contracts_reject_before_mutation(bad):
    m = model()
    if bad == 'episode_unit':
        from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
        m.train_freq = TrainFreq(1, TrainFrequencyUnit.EPISODE)
    if bad == 'decoupled_adam':
        m.actor.optimizer.defaults['decoupled_weight_decay'] = True
    if bad == 'scalar_critic_count':
        params = overrides('sac_native', True)
        params['policy_kwargs']['n_critics'] = 3
        m = create('sac', ModelSpaces(), seed=11, smoke=True, overrides=params)
    before, rng = fingerprint(m.policy.state_dict()), rng_record()
    with pytest.raises(ValueError):
        apply_initialization(m, 'sac_native' if bad == 'scalar_critic_count' else 'tqc_truncated', smoke=True)
    assert fingerprint(m.policy.state_dict()) == before and rng_record() == rng
