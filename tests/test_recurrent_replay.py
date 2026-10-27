"""Independent chronological oracle and boundary checks; no learner or market data."""
from dataclasses import replace

import numpy as np
import pytest

from rl_stocks.training.recurrent_replay import RecurrentReplayIndex, SequenceRef


def add(index, step, *, environment=0, episode=0, symbol='SPY', **flags):
    return index.append(environment=environment, episode=episode, step=step, symbol=symbol, **flags)


def ids(sequence):
    return tuple(t.insertion_id for t in sequence.transitions)


def test_overwrite_counterexample_preserves_chronology_and_physical_wrap():
    index = RecurrentReplayIndex(8)
    for step in range(11):
        add(index, step)
    windows = index.windows(burn_in=1, unroll=2)
    assert {ids(w) for w in windows} == {tuple(range(s, s + 3)) for s in range(3, 9)}
    assert (9, 10, 3) not in {ids(w) for w in windows}
    wrapped = next(w for w in windows if ids(w) == (6, 7, 8))
    assert index.resolve(wrapped) == (6, 7, 0)
    assert wrapped.loss_mask == (False, True, True)


def test_terminal_and_truncation_bootstrap_and_episode_symbol_isolation():
    index = RecurrentReplayIndex(20)
    add(index, 0)
    add(index, 1, terminated=True)
    add(index, 0, episode=1, symbol='QQQ')
    add(index, 1, episode=1, symbol='QQQ', truncated=True)
    windows = index.windows(burn_in=0, unroll=2)
    assert len(windows) == 2
    assert windows[0].bootstrap_mask == (True, False)
    assert windows[1].bootstrap_mask == (True, True)
    assert all(w.reset_after == (False, True) for w in windows)
    assert all(w.loss_mask == (True, True) for w in windows)
    with pytest.raises(ValueError, match='boundary'):
        index.resolve(SequenceRef((windows[0].transitions[-1], windows[1].transitions[0]), 0))


def test_interleaved_environments_and_overwrites_match_independent_history_oracle():
    index = RecurrentReplayIndex(17)
    history = []
    steps = [0, 0, 0]
    episodes = [0, 0, 0]
    rng = np.random.default_rng(12)
    for _ in range(120):
        env = int(rng.integers(3))
        end = steps[env] == 5
        ref = add(index, steps[env], environment=env, episode=episodes[env], truncated=end)
        history.append(ref)
        steps[env] = 0 if end else steps[env] + 1
        episodes[env] += int(end)
        retained = history[-17:]
        expected = set()
        for first in retained:
            # Oracle selects exact step identities rather than neighboring storage slots.
            chain = [next((t for t in retained if t.environment == first.environment
                           and t.episode == first.episode and t.step == first.step + k), None)
                     for k in range(3)]
            if all(t is not None for t in chain):
                expected.add(tuple(t.insertion_id for t in chain))
        actual = index.windows(burn_in=1, unroll=2)
        assert {ids(w) for w in actual} == expected
        for window in actual:
            index.resolve(window)


def test_stale_snapshot_forged_lineage_and_incomplete_prefix_reject():
    index = RecurrentReplayIndex(3)
    for step in range(3):
        add(index, step)
    window = index.windows(burn_in=1, unroll=2)[0]
    forged = replace(window, transitions=(replace(window.transitions[0], symbol='OTHER'),
                                          *window.transitions[1:]))
    with pytest.raises(ValueError):
        index.resolve(forged)
    add(index, 3)
    with pytest.raises(ValueError, match='overwritten'):
        index.resolve(window)
    fresh = index.windows(burn_in=1, unroll=2)[0]
    assert ids(fresh) == (1, 2, 3)
    with pytest.raises(ValueError):
        index.resolve(replace(fresh, burn_in=3))


@pytest.mark.parametrize('kwargs', [dict(step=2), dict(step=1, episode=1),
                                   dict(step=1, symbol='QQQ'), dict(step=True),
                                   dict(step=1, terminated=1)])
def test_bad_append_is_atomic(kwargs):
    index = RecurrentReplayIndex(4)
    add(index, 0)
    before = index.windows(burn_in=0, unroll=1)
    with pytest.raises(ValueError):
        add(index, **kwargs)
    assert index.windows(burn_in=0, unroll=1) == before
    assert add(index, 1).insertion_id == 1


def test_reset_requires_explicit_end_and_new_episode_id():
    index = RecurrentReplayIndex(4)
    add(index, 0, terminated=True)
    with pytest.raises(ValueError):
        add(index, 0)
    add(index, 0, episode=1)
    with pytest.raises(ValueError):
        add(index, 0, episode=2)


def test_sampling_reproducible_without_global_rng_or_padding():
    index = RecurrentReplayIndex(10)
    add(index, 0)
    with pytest.raises(ValueError, match='no complete'):
        index.sample(batch_size=2, burn_in=1, unroll=2, rng=np.random.default_rng(1))
    for step in range(1, 8):
        add(index, step)
    global_before = np.random.get_state()
    a = index.sample(batch_size=20, burn_in=2, unroll=3, rng=np.random.default_rng(33))
    b = index.sample(batch_size=20, burn_in=2, unroll=3, rng=np.random.default_rng(33))
    assert a == b
    assert all(len(w.transitions) == 5 and sum(w.loss_mask) == 3 for w in a)
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]


@pytest.mark.parametrize('capacity', [0, -1, True, 2.0])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        RecurrentReplayIndex(capacity)


@pytest.mark.parametrize('kwargs', [dict(burn_in=-1, unroll=1), dict(burn_in=0, unroll=0),
                                   dict(burn_in=1, unroll=4), dict(burn_in=True, unroll=1)])
def test_invalid_window_geometry(kwargs):
    with pytest.raises(ValueError):
        RecurrentReplayIndex(4).windows(**kwargs)
