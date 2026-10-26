from copy import deepcopy

import pytest
from sb3_contrib import CrossQ
from stable_baselines3 import SAC
import torch

from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.rl.crossq.geometry import layers
from rl_stocks.rl.registry.factory import create
from rl_stocks.training.crossq_inspection import SpacesOnly
from rl_stocks.training.crossq_initialization import apply_initialization
from rl_stocks.training.crossq_state import model_record, validate_record
from rl_stocks.training.crossq_spec import parameters, expected_updates
from rl_stocks.training.recurrent_restore import same_restored


@pytest.fixture(autouse=True)
def no_optimization(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail('local optimization attempted')
    for cls in [SAC, CrossQ]:
        monkeypatch.setattr(cls, 'learn', forbidden)
        monkeypatch.setattr(cls, 'train', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)


def model(arm, seed=11, smoke=True):
    return create(arm, SpacesOnly(dict(observation_size=21, action_size=4)), seed=seed,
                  smoke=smoke, overrides=parameters(arm, smoke))


@pytest.mark.parametrize('seed', [11, 23, 37])
def test_actual_pinned_full_starts_and_all_original_rng_streams(seed):
    records = {}
    for arm in ['sac', 'crossq']:
        candidate = model(arm, seed, False)
        owners = [id(p) for p in candidate.policy.parameters()]
        apply_initialization(candidate, arm, seed, False)
        records[arm] = model_record(candidate)
        validate_record(records[arm], arm, False, initial=True)
        assert owners == [id(p) for p in candidate.policy.parameters()]
        with pytest.raises(ValueError):
            apply_initialization(candidate, arm, seed, False)
    assert records['sac']['rng'] == records['crossq']['rng']


@pytest.mark.parametrize('change', ['gamma', 'optimizer', 'normalization', 'normalization_mean', 'seed'])
def test_initialization_rejects_wrong_settings_before_tensor_or_rng_mutation(change):
    candidate = model('crossq')
    if change == 'gamma':
        candidate.gamma = True
    elif change == 'optimizer':
        candidate.actor.optimizer.param_groups[0]['betas'] = (0.9, 0.999)
    elif change == 'normalization':
        candidate.actor.latent_pi[0].steps.fill_(1)
    elif change == 'normalization_mean':
        candidate.actor.latent_pi[0].ra_mean.fill_(0.1)
    else:
        candidate.seed = 23
    before = fingerprint(candidate.policy.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        apply_initialization(candidate, 'crossq', 11, True)
    assert fingerprint(candidate.policy.state_dict()) == before
    assert torch.equal(rng, torch.get_rng_state())


def populated(arm, steps=4096, phase='post_update'):
    candidate = model(arm)
    apply_initialization(candidate, arm, smoke=True)
    counts = expected_updates(arm, steps, phase, True)
    candidate.num_timesteps, candidate._n_updates = steps, counts['critic']
    for name, optimizer in [('actor', candidate.actor.optimizer), ('critic', candidate.critic.optimizer),
                            ('entropy', candidate.ent_coef_optimizer)]:
        for group in optimizer.param_groups:
            for parameter in group['params']:
                optimizer.state[parameter] = dict(step=torch.tensor(float(counts[name])),
                    exp_avg=torch.full_like(parameter, 0.01), exp_avg_sq=torch.full_like(parameter, 0.02))
    if arm == 'crossq':
        for name, layer in layers(candidate).items():
            layer.steps.fill_(counts['actor_normalization' if name.startswith('actor.') else 'critic_normalization'])
            layer.ra_mean.fill_(0.1)
            layer.ra_var.fill_(1.1)
    with torch.no_grad():
        candidate.actor.mu.weight.add_(0.001)
        candidate.log_ent_coef.fill_(-0.1)
    return candidate


@pytest.mark.parametrize('arm', ['sac', 'crossq'])
def test_actual_adam_and_normalization_serialization_without_optimizer_steps(tmp_path, arm):
    candidate = populated(arm)
    before = validate_record(model_record(candidate), arm, True)
    path = tmp_path / 'manual-state.zip'
    candidate.save(path)
    loaded = type(candidate).load(path, device='cpu')
    after = validate_record(model_record(loaded), arm, True)
    assert same_restored(after, before)


@pytest.mark.parametrize('change', ['adam', 'normalization', 'missing_optimizer', 'moment'])
def test_nominal_step_cannot_hide_wrong_actual_saved_state(change):
    candidate = populated('crossq')
    record = model_record(candidate)
    if change == 'adam':
        name = next(iter(record['optimizers']['actor']['per_parameter_steps']))
        record['optimizers']['actor']['per_parameter_steps'][name] -= 1
    elif change == 'normalization':
        name = next(iter(record['normalization']))
        record['normalization'][name]['steps'] -= 1
    elif change == 'missing_optimizer':
        record['optimizers'].pop('entropy')
    else:
        state = next(iter(candidate.actor.optimizer.state.values()))
        state['exp_avg_sq'].fill_(-1)
        with pytest.raises(ValueError):
            model_record(candidate)
        return
    with pytest.raises(ValueError):
        validate_record(deepcopy(record), 'crossq', True)


@pytest.mark.parametrize('arm', ['sac', 'crossq'])
def test_on_step_checkpoint_excludes_current_optimizer_update(arm):
    candidate = populated(arm, phase='on_step')
    record = model_record(candidate)
    validate_record(record, arm, True, phase='on_step')
    with pytest.raises(ValueError, match='gradient count'):
        validate_record(record, arm, True, phase='post_update')
