"""Append verified instrument type after existing frozen preprocessing."""
from types import MappingProxyType
from datetime import date

import gymnasium as gym
import numpy as np

from rl_stocks.rl.sac.task_policy import check_role


class InstrumentTypeObservation(gym.Wrapper):
    def __init__(self, env, records, *, task_role):
        super().__init__(env)
        self.task_role = check_role(task_role)
        if env.observation_space.shape != (21,):
            raise ValueError('instrument type requires full21 observation schema')
        registry = {}
        for record in records:
            symbol = record['symbol']
            kind = record['kind']
            if (symbol in registry or kind not in ('stock', 'etf')
                    or not record.get('source')):
                raise ValueError('invalid or duplicate instrument type provenance')
            known = date.fromisoformat(record['known_from'])
            registry[symbol] = (kind, known, record['source'])
        self.registry = MappingProxyType(registry)
        for episode in env.unwrapped.episodes:
            symbol = episode['symbol']
            if symbol not in self.registry:
                raise ValueError('unknown instrument type')
            if self.registry[symbol][1] > date.fromisoformat(episode['bars'][0]['session']):
                raise ValueError('instrument classification unavailable at episode start')
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (22,), dtype=np.float32)
        self._symbol = None

    def _append(self, observation):
        symbol = self.env.unwrapped.episode['symbol']
        if symbol != self._symbol:
            raise ValueError('instrument changed within episode')
        kind = self.registry[symbol][0]
        tag = 0 if self.task_role == 'masked_shared' else (1 if kind == 'etf' else -1)
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (21,) or not np.isfinite(value).all():
            raise ValueError('invalid full21 observation')
        return np.concatenate((value, np.asarray([tag], dtype=np.float32)))

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._symbol = self.env.unwrapped.episode['symbol']
        return self._append(observation), info

    def step(self, action):
        if self._symbol is None:
            raise RuntimeError('reset required')
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._append(observation), reward, terminated, truncated, info
