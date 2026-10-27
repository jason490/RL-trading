"""Copied recurrent transition payloads and explicit raw/SB3 boundary adapters."""
from dataclasses import dataclass
from numbers import Real

import numpy as np

from rl_stocks.training.recurrent_replay import (
    RecurrentReplayIndex, SequenceRef, TransitionRef, _integer,
)


def _vector(value, size, name):
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind not in 'fiu':
        raise ValueError(f'{name} requires a real vector of shape {(size,)}')
    with np.errstate(over='ignore'):
        copied = raw.astype(np.float32, copy=True)
    if not np.isfinite(copied).all():
        raise ValueError(f'{name} must be finite float32')
    copied.flags.writeable = False
    return copied


@dataclass(frozen=True)
class SequenceBatch:
    # Observations [batch,time+1,features] include the final observation.
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    loss_mask: np.ndarray
    bootstrap_mask: np.ndarray
    reset_after: np.ndarray
    insertion_ids: np.ndarray
    burn_in: int


class RecurrentPayloadStore:
    """Single-writer store; append_raw receives an unreset Gymnasium next observation.

    No normalization updates, hidden states or padding are performed. The separate
    append_sb3 adapter consumes one lane of SB3 VecEnv output, including terminal
    observations. It deliberately does not accept arbitrary Gymnasium autoreset APIs.
    """

    def __init__(self, capacity: int, observation_size: int):
        _integer('observation_size', observation_size, 1)
        self.index = RecurrentReplayIndex(capacity)
        self.observation_size = observation_size
        self._payloads = [None] * capacity
        self._last: dict[int, tuple[TransitionRef, np.ndarray]] = {}

    def append_raw(self, *, observation, action, reward, next_observation,
                   environment: int, episode: int, step: int, symbol: str,
                   terminated: bool, truncated: bool) -> TransitionRef:
        obs = _vector(observation, self.observation_size, 'observation')
        nxt = _vector(next_observation, self.observation_size, 'next observation')
        act = _vector(action, 4, 'action')
        if np.any(np.abs(act) > 1):
            raise ValueError('four-component action must be within [-1,1]')
        if isinstance(reward, (bool, np.bool_)) or not isinstance(reward, Real):
            raise ValueError('finite real reward required')
        with np.errstate(over='ignore'):
            rew = np.float32(reward)
        if not np.isfinite(rew):
            raise ValueError('finite float32 reward required')
        _integer('environment', environment)
        previous = self._last.get(environment)
        if previous is not None and not previous[0].ended and not np.array_equal(previous[1], obs):
            raise ValueError('observation does not continue the previous transition')
        ref = self.index.append(environment=environment, episode=episode, step=step,
                                symbol=symbol, terminated=terminated, truncated=truncated)
        self._payloads[ref.insertion_id % self.index.capacity] = (ref, obs, act, rew, nxt)
        self._last[environment] = (ref, nxt)
        return ref

    def append_sb3(self, *, observation, action, reward, returned_observation,
                   done: bool, info: dict, **identity) -> TransitionRef:
        if type(done) is not bool or type(info) is not dict:
            raise ValueError('explicit SB3 done boolean and info dictionary required')
        timeout = info.get('TimeLimit.truncated', False)
        if type(timeout) is not bool or (timeout and not done):
            raise ValueError('invalid SB3 time-limit flag')
        if done and 'terminal_observation' not in info:
            raise ValueError('SB3 ended transition requires terminal_observation')
        if not done and 'terminal_observation' in info:
            raise ValueError('terminal observation on ongoing transition')
        return self.append_raw(observation=observation, action=action, reward=reward,
                               next_observation=info['terminal_observation'] if done
                               else returned_observation,
                               terminated=done and not timeout, truncated=done and timeout,
                               **identity)

    def gather(self, sequences: tuple[SequenceRef, ...]) -> SequenceBatch:
        if type(sequences) is not tuple or not sequences:
            raise ValueError('nonempty tuple of sequences required')
        # Resolve every snapshot before constructing a result. Caller synchronizes writes.
        slots = [self.index.resolve(s) for s in sequences]
        burn_in, length = sequences[0].burn_in, len(slots[0])
        if any(s.burn_in != burn_in or len(p) != length for s, p in zip(sequences, slots)):
            raise ValueError('mixed sequence geometry')
        rows = []
        for seq, positions in zip(sequences, slots):
            payloads = [self._payloads[p] for p in positions]
            if any(p is None or p[0] != ref for p, ref in zip(payloads, seq.transitions)):
                raise ValueError('payload/reference mismatch')
            rows.append(payloads)
        observations = np.stack([np.stack([row[0][1], *(p[4] for p in row)]) for row in rows])
        values = dict(observations=observations,
                      actions=np.stack([np.stack([p[2] for p in row]) for row in rows]),
                      rewards=np.asarray([[p[3] for p in row] for row in rows], dtype=np.float32),
                      loss_mask=np.asarray([s.loss_mask for s in sequences], dtype=bool),
                      bootstrap_mask=np.asarray([s.bootstrap_mask for s in sequences], dtype=bool),
                      reset_after=np.asarray([s.reset_after for s in sequences], dtype=bool),
                      insertion_ids=np.asarray([[t.insertion_id for t in s.transitions]
                                                for s in sequences], dtype=np.int64))
        for value in values.values():
            value.flags.writeable = False
        return SequenceBatch(**values, burn_in=burn_in)
