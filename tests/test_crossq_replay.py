import pickle

import numpy as np
import pytest
from stable_baselines3.common.buffers import ReplayBuffer

from rl_stocks.training.crossq_inspection import SpacesOnly
from rl_stocks.training.crossq_replay import replay_record


def replay():
    env = SpacesOnly(dict(observation_size=21, action_size=4))
    buffer = ReplayBuffer(100000, env.observation_space, env.action_space, device='cpu')
    for index in range(4):
        buffer.add(np.full((1, 21), index, dtype=np.float32),
                   np.full((1, 21), index + 1, dtype=np.float32),
                   np.zeros((1, 4), dtype=np.float32), np.array([0.1]),
                   np.array([index == 3]), [{'TimeLimit.truncated': index == 3}])
    return buffer


def test_actual_replay_prefix_and_timeout_pickle_parity_without_sampling():
    buffer = replay()
    before = replay_record(buffer, 4)
    restored = pickle.loads(pickle.dumps(buffer))
    assert replay_record(restored, 4) == before
    assert before['timeout_count'] == before['done_count'] == 1
    assert before['terminal_count'] == 0
    restored.observations[99999] = np.nan  # Unfilled capacity is not evidence.
    assert replay_record(restored, 4) == before
    restored.observations[0, 0, 0] += 1
    assert replay_record(restored, 4) != before


@pytest.mark.parametrize('change', ['count', 'timeout', 'nonfinite', 'action', 'disabled_timeout'])
def test_replay_mask_schema_and_filled_data_errors_reject(change):
    buffer = replay()
    if change == 'count':
        buffer.pos -= 1
    elif change == 'timeout':
        buffer.timeouts[0] = 1
    elif change == 'nonfinite':
        buffer.rewards[0] = np.nan
    elif change == 'action':
        buffer.actions[0] = 1.1
    else:
        buffer.handle_timeout_termination = False
    with pytest.raises(ValueError):
        replay_record(buffer, 4)
