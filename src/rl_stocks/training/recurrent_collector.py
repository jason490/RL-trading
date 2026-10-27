"""Single raw TradingEnv collection into episode-safe replay; no learner calls."""
from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.rl.recurrent_sac.collection import OnlineRecurrentActor
from rl_stocks.training.recurrent_payload import RecurrentPayloadStore, _vector
from rl_stocks.training.recurrent_replay import TransitionRef, _integer


@dataclass(frozen=True)
class CollectedTransition:
    reference: TransitionRef
    observation: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    warmup: bool
    info: dict


class RecurrentCollector:
    """Own one training environment's collection state, with explicit raw reset semantics.

    The caller owns closing the environment. Any failure after collection starts
    poisons this collector: the simulator may have advanced before validation
    failed, so silently retrying the transition would corrupt replay lineage.
    """

    def __init__(self, env: TradingEnv, networks, store: RecurrentPayloadStore, *,
                 seed: int, warmup_steps: int, max_episode_steps: int = 512):
        _integer('seed', seed)
        _integer('warmup_steps', warmup_steps)
        _integer('max_episode_steps', max_episode_steps, 1)
        if not isinstance(env, TradingEnv) or any(ep['split'] != 'train' for ep in env.episodes):
            raise ValueError('raw training-only TradingEnv required')
        if (env.observation_profile != 'execution_full' or env.exposure_execution != 'unrestricted'
                or env.reward_kind != 'net_log_equity' or env.reward_scale != 1.):
            raise ValueError('fixed execution-full/unrestricted/net-log-equity contract required')
        if (type(env.episode_steps) is not int or not 1 <= env.episode_steps <= max_episode_steps
                or env.observation_space.shape != (networks.observation_size,)
                or store.observation_size != networks.observation_size or env.action_space.shape != (4,)):
            raise ValueError('bounded episode and observation/action geometry required')
        self.env, self.store = env, store
        self.policy = OnlineRecurrentActor(networks, max_history=max_episode_steps)
        self.rng = np.random.default_rng(seed)
        self.seed, self.warmup_steps = seed, warmup_steps
        self.transitions = 0
        self.episode = self.step = 0
        self.failed = False
        self._observation = None
        self._symbol = None

    def _reset(self):
        observation, info = self.env.reset(seed=self.seed if self.transitions == 0 else None)
        if type(info.get('symbol')) is not str or not info['symbol']:
            raise ValueError('reset requires explicit symbol identity')
        self._observation = _vector(observation, self.store.observation_size, 'reset observation')
        self._symbol = info['symbol']

    def collect(self, *, policy_revision: int, deterministic=False) -> CollectedTransition:
        if self.failed:
            raise RuntimeError('collector failed after potential environment advance; no retry')
        _integer('policy revision', policy_revision)
        if type(deterministic) is not bool:
            raise ValueError('explicit deterministic boolean required')
        try:
            if self._observation is None:
                self._reset()
            observation = self._observation
            warmup = self.transitions < self.warmup_steps
            random_action = self.rng.uniform(-1, 1, size=4).astype(np.float32) if warmup else None
            action = self.policy.action(observation, revision=policy_revision, episode_start=self.step == 0,
                                        random_action=random_action, deterministic=deterministic)
            nxt, reward, terminated, truncated, info = self.env.step(action.copy())
            nxt = _vector(nxt, self.store.observation_size, 'true final/next observation')
            ref = self.store.append_raw(observation=observation, action=action, reward=reward,
                                        next_observation=nxt, environment=0, episode=self.episode,
                                        step=self.step, symbol=self._symbol,
                                        terminated=terminated, truncated=truncated)
            result = CollectedTransition(ref, observation.copy(), action.copy(), nxt.copy(),
                                          float(reward), terminated, truncated, warmup, deepcopy(info))
            self.transitions += 1
            self.step += 1
            if terminated or truncated:
                # Defer raw reset to the next collect call; replay already owns true final obs.
                self._observation = None
                self.episode += 1
                self.step = 0
            else:
                self._observation = nxt
            return result
        except BaseException:
            self.failed = True
            raise
