"""Actual construction and locked gates; no fitting or optimizer updates."""
from copy import deepcopy
import numpy as np
import pytest
from test_sac_regularization import Spaces
from test_history_gate import cells
from rl_stocks.rl.registry.factory import create, CLASSES
from rl_stocks.training.recurrent_state import protocol, arm_spec, overrides, model_record, validate_record
from rl_stocks.reporting.history import screen_gate
from rl_stocks.rl.registry.prediction import PredictionStream


@pytest.mark.parametrize('smoke', [False, True])
def test_actual_construction_and_recurrent_initial_equality(smoke):
    records = {}
    for arm in protocol()['arms']:
        requested = overrides(arm, smoke)
        before = deepcopy(requested)
        model = create(arm_spec(arm)['algorithm'], Spaces(), smoke=smoke, overrides=requested)
        record = model_record(model)
        validate_record(record, arm, smoke, initial=True)
        assert not model.policy.optimizer.state
        assert requested == before
        records[arm] = record
        for key, value in [('observation_size', 12), ('gamma', .95), ('num_timesteps', 100000), ('critic_lstm', False if arm != 'feedforward_full' else True)]:
            bad = deepcopy(record); bad[key] = value
            with pytest.raises(ValueError):
                validate_record(bad, arm, smoke, initial=True)
    assert records['recurrent_masked'] == records['recurrent_full']
    assert records['recurrent_full']['policy_tensor_sha256'] != records['feedforward_full']['policy_tensor_sha256']


@pytest.mark.parametrize('arm', ['feedforward_full', 'recurrent_masked', 'recurrent_full'])
def test_untrained_save_load_and_sequence_reset(tmp_path, arm):
    algorithm = arm_spec(arm)['algorithm']
    model = create(algorithm, Spaces(), smoke=True, overrides=overrides(arm, True))
    model.save(tmp_path/'policy')
    restored = CLASSES[algorithm].load(tmp_path/'policy.zip', device='cpu')
    assert model_record(restored) == model_record(model)
    observations = np.random.default_rng(11).normal(size=(8, 21)).astype(np.float32)
    expected = PredictionStream(model)
    actual = PredictionStream(restored)
    actions = []
    for obs in observations:
        action = expected.predict(obs)
        np.testing.assert_array_equal(action, actual.predict(obs))
        actions.append(action)
    reset = PredictionStream(restored)
    for obs, action in zip(observations, actions):
        np.testing.assert_array_equal(action, reset.predict(obs))
    assert model.num_timesteps == restored.num_timesteps == 0


def test_gate_requires_both_controls_and_absolute_group_targets():
    old = cells()
    panel = dict(feedforward_full=old['original'], recurrent_masked=old['recent_etf'], recurrent_full=old['older_etf'])
    assert screen_gate(panel, protocol())['passed']
    for mutate in [lambda p: p['recurrent_masked']['final'].update(score=.21),
                   lambda p: p['recurrent_full']['selected']['groups']['stock'].update(excess_return=-.03),
                   lambda p: p['recurrent_full']['final']['groups']['etf'].update(excess_return=-.03)]:
        bad=deepcopy(panel); mutate(bad)
        assert not screen_gate(bad, protocol())['passed']
    assert not screen_gate(panel, protocol())['objective_achieved']


def test_no_sweeps_or_implicit_smoke():
    with pytest.raises(ValueError): overrides('larger_memory')
    with pytest.raises(ValueError): overrides('recurrent_full', 1)
    spec = overrides('recurrent_full')
    spec['policy_kwargs']['net_arch'][0] = 999
    assert overrides('recurrent_full')['policy_kwargs']['net_arch'] == [64, 64]
