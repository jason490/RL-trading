from types import SimpleNamespace

import pytest

from rl_stocks.training.treasury_sampling import SamplingPaths, validate_paths


class Env:
    def __init__(self, offset=0):
        self.episode = {'symbol': 'A'}
        self.episode_index = 0
        self.episode_steps = 2
        self.start_sampler = SimpleNamespace(bounds=[3])
        self.bars = [SimpleNamespace(session=str(i + offset)) for i in range(5)]
        self.resets = self.steps = 0

    def reset(self, *, seed=None):
        self.resets += 1
        self.i = 1
        return [1], {'seed': seed}

    def step(self, action):
        self.steps += 1
        self.i += 1
        return [self.i], 0, False, self.i == 3, {'action': action}


def record(offset=0):
    env = Env(offset)
    audit = SamplingPaths(env)
    assert env.reset(seed=11) == ([1], {'seed': 11})
    for action in [0.1, -0.2]:
        assert env.step(action)[-1] == {'action': action}
    assert env.resets == 1 and env.steps == 2
    return audit.snapshot()


def test_exact_observed_paths_without_extra_calls_and_constant_record_size():
    first = record()
    assert first == record()
    assert first['window_sha256'] != record(1)['window_sha256']
    assert first['date_sha256'] != record(1)['date_sha256']
    assert validate_paths(first, dict(starts=1, transitions=2), 2) == first
    assert len(first) == 6


@pytest.mark.parametrize('field', ['starts', 'transitions', 'window_sha256', 'measured'])
def test_missing_or_forged_path_fields_reject(field):
    value = record()
    value.pop(field)
    with pytest.raises(ValueError):
        validate_paths(value, dict(starts=1, transitions=2), 2)
