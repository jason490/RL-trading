"""Serialization/counter fixtures only; no actual learning or optimizer.step."""
from copy import deepcopy

import pytest
import torch

from test_distributional_initialization import model
from rl_stocks.rl.registry.factory import CLASSES
from rl_stocks.training.distributional_critics import apply_initialization, protocol
from rl_stocks.training.distributional_records import model_record, validate_record


def fixture(arm='tqc_truncated'):
    m = model(arm)
    apply_initialization(m, arm, smoke=True)
    for optimizer in (m.actor.optimizer, m.critic.optimizer, m.ent_coef_optimizer):
        for group in optimizer.param_groups:
            for p in group['params']:
                optimizer.state[p] = dict(step=torch.tensor(960.), exp_avg=torch.ones_like(p),
                                          exp_avg_sq=torch.full_like(p, 0.5))
    m.num_timesteps = 1024
    m._n_updates = 960
    return m


@pytest.mark.parametrize('arm', list(protocol()['arms']))
def test_actual_three_optimizer_serialization_and_all_parameter_steps(tmp_path, arm):
    m = model(arm)
    apply_initialization(m, arm, smoke=True)
    initial = model_record(m)
    validate_record(initial, arm, True, initial=True)
    m = fixture(arm)
    record = model_record(m)
    validate_record(record, arm, True)
    m.save(tmp_path / 'fixture.zip')
    restored = CLASSES[protocol()['arms'][arm]['algorithm']].load(tmp_path / 'fixture.zip', device='cpu')
    actual = model_record(restored)
    validate_record(actual, arm, True)
    assert {k: v for k, v in actual.items() if k != 'rng'} == {k: v for k, v in record.items() if k != 'rng'}
    assert [record['adam_state'][key]['state_count'] for key in ('actor', 'critic', 'entropy')] == [8, 12, 1]


@pytest.mark.parametrize('bad', ['updates', 'step_min', 'missing_state', 'parameter_count', 'initial_proof', 'alpha', 'seed'])
def test_semantically_inconsistent_counters_or_provenance_rejected(bad):
    record = deepcopy(model_record(fixture()))
    if bad == 'updates':
        record['updates'] = 959
    if bad == 'step_min':
        record['adam_state']['critic']['step_min'] = 959
    if bad == 'missing_state':
        record['adam_state']['entropy']['per_parameter_steps'] = {}
    if bad == 'parameter_count':
        record['adam_state']['actor']['parameter_count'] += 1
    if bad == 'initial_proof':
        record['shared_initialization']['artifact_sha256'] = '0' * 64
    if bad == 'alpha':
        record['initialization']['alpha'] = 0
    if bad == 'seed':
        record['seed'] = 23
    with pytest.raises(ValueError):
        validate_record(record, 'tqc_truncated', True)


@pytest.mark.parametrize('bad', ['shape', 'nonfinite', 'negative_second'])
def test_actual_invalid_adam_moments_rejected(bad):
    m = fixture()
    p = m.actor.optimizer.param_groups[0]['params'][0]
    state = m.actor.optimizer.state[p]
    if bad == 'shape':
        state['exp_avg'] = torch.zeros(1)
    if bad == 'nonfinite':
        state['exp_avg'].fill_(float('nan'))
    if bad == 'negative_second':
        state['exp_avg_sq'].fill_(-1)
    with pytest.raises(ValueError):
        model_record(m)
