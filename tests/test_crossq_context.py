from copy import deepcopy

import pytest

from rl_stocks.training import crossq_context as context
from rl_stocks.training.crossq_spec import parameters, protocol, PROTOCOL_SHA256


def request(arm='crossq'):
    return dict(dataset_id=protocol()['dataset'], algorithm=arm, overrides=parameters(arm, True),
                seed=11, smoke=True, steps=4096, every=1024, initialization='standard',
                policy_variant='native', observation_profile='execution_full', start_sampling='legacy_uniform',
                reward_kind='net_log_equity', reward_scale=1.0, training_exposure_mode='unrestricted',
                evaluation_exposure_mode='unrestricted', final_test=False, compact_checkpoints=True,
                milestones=[4096], device='cpu')


@pytest.mark.parametrize('arm', ['sac', 'crossq'])
def test_actual_fixed_requests_and_explicit_dispatch(arm):
    assert context.validate_request(**request(arm)) == arm
    assert context.validate_dispatch(arm, PROTOCOL_SHA256)
    assert not context.validate_dispatch('sac', None)
    with pytest.raises(ValueError, match='explicit'):
        context.validate_dispatch('crossq', None)


@pytest.mark.parametrize('field,value', [('steps', 1024), ('every', 512), ('seed', True),
    ('smoke', 1), ('final_test', True), ('device', 'cuda'), ('compact_checkpoints', False),
    ('evaluation_exposure_mode', 'unit_magnitude'), ('reward_scale', 2.0), ('milestones', [1024, 4096])])
def test_unregistered_contract_changes_reject(field, value):
    values = request()
    values[field] = value
    with pytest.raises(ValueError):
        context.validate_request(**values)


def test_algorithm_parameter_mismatch_and_invalid_payload_reject():
    values = request('sac')
    values['overrides'] = parameters('crossq', True)
    with pytest.raises(ValueError):
        context.validate_request(**values)
    with pytest.raises(ValueError):
        context.validate_payload({})


def test_invalid_registered_runner_stops_before_environment_or_model(monkeypatch):
    from rl_stocks.training import runner, dependencies
    monkeypatch.setattr(dependencies, 'verify_environment', lambda: {})
    monkeypatch.setattr(runner, 'read_dataset', lambda p: {})
    def forbidden(*a, **k):
        pytest.fail('constructed before request validation')
    monkeypatch.setattr(runner, 'TradingEnv', forbidden)
    monkeypatch.setattr(runner, 'create', forbidden)
    values = deepcopy(request())
    dataset = values.pop('dataset_id')
    algorithm = values.pop('algorithm')
    values['evaluation_every'] = values.pop('every')
    values['steps'] = 1024
    with pytest.raises(ValueError, match='transition'):
        runner.train(dataset, algorithm, study_protocol_sha256=PROTOCOL_SHA256, **values)
