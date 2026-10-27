"""Live observation-only actor history for offline training collection."""
import numpy as np
import torch

from rl_stocks.training.recurrent_payload import _vector
from rl_stocks.training.recurrent_replay import _integer


class OnlineRecurrentActor:
    """Track one episode; rebuild its complete observed prefix after parameter updates.

    Unlike FrozenRecurrentActor this references the online actor. The owner must
    increment revision whenever actor parameters change. Observation history is
    bounded by the registered episode limit; it contains no future observations.
    """

    def __init__(self, networks, *, max_history: int):
        _integer('max_history', max_history, 1)
        if any(p.device.type != 'cpu' for p in networks.actor.parameters()):
            raise ValueError('CPU collection actor required')
        self._actor = networks.actor
        self._observation_size = networks.observation_size
        self._max_history = max_history
        self._history = []
        self._hidden = None
        self._revision = None

    def action(self, observation, *, revision: int, episode_start: bool,
               random_action=None, deterministic=False):
        _integer('revision', revision)
        if type(episode_start) is not bool or type(deterministic) is not bool:
            raise ValueError('explicit reset/deterministic booleans required')
        if self._revision is not None and revision < self._revision:
            raise ValueError('actor revision cannot decrease')
        obs = _vector(observation, self._observation_size, 'normalized observation')
        if random_action is not None:
            random_action = _vector(random_action, 4, 'warmup action')
            if (np.abs(random_action) > 1).any():
                raise ValueError('warmup action outside normalized bounds')
        previous = [] if episode_start else self._history
        if len(previous) >= self._max_history:
            raise ValueError('episode observation history limit reached before reset')
        history = [*previous, obs]
        with torch.no_grad():
            if episode_start or self._revision != revision or self._hidden is None:
                # Rebuild with current parameters, including observations during warmup.
                inputs = torch.tensor(np.stack(history)).unsqueeze(0)
                features, hidden = self._actor.memory(inputs)
            else:
                inputs = torch.tensor(obs).reshape(1, 1, -1)
                features, hidden = self._actor.memory(inputs, self._hidden)
            if random_action is not None:
                result = random_action.copy()
            elif deterministic:
                result = self._actor.mean(self._actor.head(features[:, -1:])).tanh().reshape(4).numpy().copy()
            else:
                result = self._actor.sample(features[:, -1:])[0].reshape(4).numpy().copy()
        if not np.isfinite(result).all() or not bool(torch.isfinite(hidden).all()):
            raise ValueError('nonfinite collection action or history')
        self._history, self._hidden, self._revision = history, hidden.detach(), revision
        return result
