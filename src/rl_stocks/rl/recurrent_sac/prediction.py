"""Private frozen actor over already-normalized versioned observations."""
from copy import deepcopy

import numpy as np
import torch

from rl_stocks.training.recurrent_payload import _vector


class FrozenRecurrentActor:
    def __init__(self, networks, *, observation_schema: str):
        if type(observation_schema) is not str or not observation_schema:
            raise ValueError('versioned normalized observation schema required')
        self._actor = deepcopy(networks.actor).cpu().eval().requires_grad_(False)
        self._observation_size = networks.observation_size
        self._hidden_size = networks.hidden_size
        self._schema = observation_schema

    @property
    def observation_schema(self):
        return self._schema

    def predict(self, observation, *, observation_schema: str, state=None, episode_start=False):
        if observation_schema != self._schema or type(episode_start) is not bool:
            raise ValueError('observation schema or reset flag mismatch')
        obs = _vector(observation, self._observation_size, 'normalized observation')
        if state is not None:
            state = np.asarray(state)
            if (state.dtype != np.float32 or state.shape != (1, 1, self._hidden_size)
                    or not np.isfinite(state).all()):
                raise ValueError('finite float32 single-lane recurrent state required')
        with torch.no_grad():
            hidden = None if episode_start or state is None else torch.tensor(state)
            features, nxt = self._actor.memory(torch.tensor(obs).reshape(1, 1, -1), hidden)
            mean = self._actor.mean(self._actor.head(features))
            action = mean.tanh().reshape(4).numpy().copy()
            carried = nxt.numpy().copy()
        if not np.isfinite(action).all() or not np.isfinite(carried).all():
            raise ValueError('nonfinite frozen actor output')
        return action, carried
