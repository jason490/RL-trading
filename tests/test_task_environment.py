from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check

from test_causal_sampling import episode
from test_core import bar
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.evaluation.task_conditioning import EpisodeTaskPredictor, evaluate_task_training
from rl_stocks.rl.sac.task_factory import create_task_sac
from rl_stocks.training.task_environment import build_task_environment


@pytest.fixture(autouse=True)
def no_local_optimization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('local optimization attempted')
    monkeypatch.setattr(SAC, 'learn', forbidden)
    monkeypatch.setattr(SAC, 'train', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)


def payload():
    symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'A']
    names = [f'f{i}' for i in range(7)]
    episodes = {}
    for symbol in symbols:
        ep = episode()
        ep['symbol'] = symbol
        for item in ep['bars']:
            item['symbol'] = symbol
        for session in ('2017-12-29', '2018-01-02', '2020-12-02'):
            item = asdict(bar(session, op=100., high=101., low=99., close=100.,
                              adjustments='split_adjusted_with_dividends'))
            item['symbol'] = symbol
            ep['bars'].append(item)
            ep['atr'].append(2.)
        ep['features'] = [[0.] * 7 for _ in ep['bars']]
        episodes[symbol] = ep
    data = {'schema': 'fixture', 'names': names,
            'normalizer': asdict(Normalizer(tuple(names), (0.,) * 7, (1.,) * 7)),
            'episodes': episodes,
            'manifest': {'included_splits': ['train', 'validation'],
                         'evaluation_symbols': symbols, 'unverified_override': False,
                         'universe_config': {'instruments': [
                             {'symbol': s, 'kind': 'stock' if s == 'A' else 'broad_etf'}
                             for s in symbols]}}}
    evidence = {'schema': 'historical-instrument-types-v1', 'dataset': digest(data),
                'historical_evidence_verified': True, 'records': [
                    {'symbol': s, 'kind': 'stock' if s == 'A' else 'etf',
                     'source': 'synthetic historical fixture only',
                     'known_from': '2010-01-01', 'evidence_kind': 'historical_primary_source'}
                    for s in symbols]}
    return data, evidence


def test_typed_environments_pass_checkers_and_match_sampling():
    data, evidence = payload()
    before = deepcopy(data)
    envs = [build_task_environment(data, evidence, task_role=role, smoke=True)
            for role in ('masked_shared', 'tagged_shared', 'tagged_routed')]
    try:
        sb3_check(envs[1])
        check_env(envs[1], skip_render_check=True)
        for env in envs:
            env.reset(seed=23)
        for _ in range(8):
            states = [env.reset() for env in envs]
            assert len({info['symbol'] for _, info in states}) == 1
            assert all(np.array_equal(states[0][0][:-1], obs[:-1]) for obs, _ in states)
            assert states[0][0][-1] == 0
            expected = -1 if states[1][1]['symbol'] == 'A' else 1
            assert states[1][0][-1] == states[2][0][-1] == expected
        assert data == before
    finally:
        for env in envs:
            env.close()


def test_fixed_training_replay_uses_exact_group_tags(monkeypatch):
    data, evidence = payload()
    env = build_task_environment(data, evidence, task_role='tagged_routed', smoke=True)
    try:
        model = create_task_sac(env, task_role='tagged_routed', smoke=True)
        tags = []
        original = model.predict

        def record(obs, **kwargs):
            tags.append(float(obs[-1]))
            return original(obs, **kwargs)

        monkeypatch.setattr(model, 'predict', record)
        result = evaluate_task_training(model, data, evidence)
        assert len(result) == 5
        assert all(len(cell['rows']) == 2 for cell in result.values())
        assert tags == [1.] * 8 + [-1.] * 2
        assert all(cell['split'] == 'train' for cell in result.values())
        assert all(len(row['policy_action']) == 4 for cell in result.values()
                   for row in cell['rows'])
    finally:
        env.close()


def test_episode_predictor_rejects_unavailable_and_exhausted_streams():
    data, evidence = payload()
    env = build_task_environment(data, evidence, task_role='tagged_shared', smoke=True)
    try:
        model = create_task_sac(env, task_role='tagged_shared', smoke=True)
        ep = data['episodes']['SPY']
        short = {**ep, 'bars': ep['bars'][:2]}
        adapter = EpisodeTaskPredictor(model, short, evidence['records'][0])
        with pytest.raises(ValueError, match='full21'):
            adapter.predict(np.zeros(22))
        adapter.predict(np.zeros(21))
        with pytest.raises(ValueError, match='exhausted'):
            adapter.predict(np.zeros(21))
        late = {**evidence['records'][0], 'known_from': '2020-01-01'}
        with pytest.raises(ValueError, match='unavailable'):
            EpisodeTaskPredictor(model, short, late).predict(np.zeros(21))
    finally:
        env.close()
