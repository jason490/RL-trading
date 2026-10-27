"""Common simulated evaluation on synthetic validation bars; no training or APIs."""
from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta

import numpy as np
import pytest
import torch

from test_core import bar
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.evaluation.recurrent_sac import evaluate_recurrent_sac
from rl_stocks.evaluation.runner import baselines
from rl_stocks.rl.recurrent_sac.networks import RecurrentSACNetworks
from rl_stocks.rl.recurrent_sac.prediction import FrozenRecurrentActor
from rl_stocks.rl.registry.prediction import PredictionStream


def fixture():
    names = [f'x{i}' for i in range(7)]
    episodes = {}
    for symbol in ('SPY', 'QQQ'):
        bars = []
        for i in range(5):
            value = asdict(bar(str(date(2021, 1, 4)+timedelta(days=i)),
                              op=100+i, high=102+i, low=99+i, close=101+i))
            value['symbol'] = symbol
            bars.append(value)
        episodes[symbol] = dict(symbol=symbol, split='validation', bars=bars,
                                 features=[[float(i)/10]*7 for i in range(5)], atr=[2.]*5)
    dataset = dict(schema='daily-indicators-v1', names=names, episodes=episodes,
                   normalizer=asdict(Normalizer(tuple(names), (0.,)*7, (1.,)*7)))
    with torch.random.fork_rng():
        torch.manual_seed(11)
        networks = RecurrentSACNetworks(21, 8)
    return networks, dataset


def test_common_path_baseline_cost_direction_and_private_frozen_stream_parity():
    networks, dataset = fixture()
    original_data = deepcopy(dataset)
    original_weights = deepcopy(networks.state_dict())
    rng = torch.get_rng_state().clone()
    actual = evaluate_recurrent_sac(networks, dataset, exposure_mode='unrestricted')
    normalizer = Normalizer(tuple(dataset['names']), (0.,)*7, (1.,)*7)
    frozen = FrozenRecurrentActor(networks, observation_schema='execution-full-v1-normalized21')
    for symbol, result in actual.items():
        env = TradingEnv([dataset['episodes'][symbol]], normalizer, dataset['names'],
                         observation_profile='execution_full')
        try:
            obs, _ = env.reset(seed=11, options={'episode': 0})
            state = None
            for i, row in enumerate(result['rows']):
                expected, state = frozen.predict(obs, observation_schema=frozen.observation_schema,
                                                  state=state, episode_start=i == 0)
                np.testing.assert_array_equal(row['raw_action'], expected)
                obs, _, _, _, _ = env.step(expected)
            assert result['baselines'] == baselines(dataset['episodes'][symbol], env.config)
            assert result['metrics']['net_return'] == pytest.approx(env.sim.portfolio.equity(env.bars[-1].close)/100000-1)
            assert result['directional'] and result['events'] == env.sim.portfolio.events
        finally:
            env.close()
    assert dataset == original_data and torch.equal(rng, torch.get_rng_state())
    for key, value in networks.state_dict().items():
        assert torch.equal(value, original_weights[key])
    assert all(p.grad is None for p in networks.parameters())


def test_stream_state_resets_per_episode_and_unit_exposure_filter_uses_common_contract():
    networks, dataset = fixture()
    frozen = FrozenRecurrentActor(networks, observation_schema='execution-full-v1-normalized21')
    a, b = PredictionStream(frozen), PredictionStream(frozen)
    first = a.predict(np.zeros(21, dtype=np.float32))
    carried = a.predict(np.zeros(21, dtype=np.float32))
    fresh = b.predict(np.zeros(21, dtype=np.float32))
    np.testing.assert_array_equal(first, fresh)
    assert not np.array_equal(first, carried)
    results = evaluate_recurrent_sac(networks, dataset, exposure_mode='unit_magnitude')
    for result in results.values():
        for row in result['rows']:
            assert row['exposure_mode'] == 'unit_magnitude'
            assert abs(row['raw_action'][0]) in (0, 1)
            assert row['raw_action'][1:] == row['policy_action'][1:]


@pytest.mark.parametrize('change', ['test_route', 'retired_episode', 'retired_date', 'schema',
                                   'names', 'normalizer', 'date_order', 'mode'])
def test_invalid_or_retired_evaluation_rejects_before_forward(monkeypatch, change):
    networks, dataset = fixture()
    kwargs = dict(exposure_mode='unrestricted')
    if change == 'test_route':
        kwargs['split'] = 'test'
    elif change == 'retired_episode':
        dataset['episodes']['SPY']['split'] = 'test'
    elif change == 'retired_date':
        dataset['episodes']['SPY']['bars'][0]['session'] = '2024-01-01'
    elif change == 'schema':
        dataset['schema'] = 'unknown'
    elif change == 'names':
        dataset['names'] = dataset['names'][:-1]
    elif change == 'normalizer':
        dataset['names'] = list(reversed(dataset['names']))
    elif change == 'date_order':
        dataset['episodes']['SPY']['bars'][1]['session'] = '2021-01-04'
    else:
        kwargs['exposure_mode'] = 'unregistered'
    def forbidden(*args, **kw):
        pytest.fail('policy forward called before invalid input rejected')
    monkeypatch.setattr(FrozenRecurrentActor, 'predict', forbidden)
    with pytest.raises(ValueError):
        evaluate_recurrent_sac(networks, dataset, **kwargs)


def test_common_public_test_route_cannot_bypass_new_family_retired_guard():
    from rl_stocks.evaluation.runner import evaluate
    networks, dataset = fixture()
    frozen = FrozenRecurrentActor(networks, observation_schema='execution-full-v1-normalized21')
    with pytest.raises(ValueError, match='validation only'):
        evaluate(frozen, dataset, split='test', observation_profile='execution_full')
