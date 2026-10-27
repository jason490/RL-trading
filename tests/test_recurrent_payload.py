import numpy as np
import pytest

from rl_stocks.training.recurrent_payload import RecurrentPayloadStore


def append(store, step, **overrides):
    values = dict(observation=[step, 1], next_observation=[step + 1, 1],
                  action=[0, .1, .2, .3], reward=.25, environment=0, episode=0,
                  step=step, symbol='SPY', terminated=False, truncated=False)
    values.update(overrides)
    return store.append_raw(**values)


def test_shifted_history_final_observation_masks_and_copied_inputs():
    store = RecurrentPayloadStore(8, 2)
    obs = np.array([0., 1.], dtype=np.float32)
    append(store, 0, observation=obs)
    obs[:] = 99
    append(store, 1)
    append(store, 2, truncated=True)
    batch = store.gather(store.index.windows(burn_in=1, unroll=2))
    np.testing.assert_array_equal(batch.observations, [[[0, 1], [1, 1], [2, 1], [3, 1]]])
    np.testing.assert_array_equal(batch.loss_mask, [[False, True, True]])
    np.testing.assert_array_equal(batch.bootstrap_mask, [[True, True, True]])
    np.testing.assert_array_equal(batch.reset_after, [[False, False, True]])
    assert batch.actions.shape == (1, 3, 4) and batch.rewards.dtype == np.float32
    with pytest.raises(ValueError):
        batch.observations[0, 0] = 77
    # Even deliberately making the gathered copy writable cannot mutate storage.
    batch.observations.flags.writeable = True
    batch.observations[:] = 77
    again = store.gather(store.index.windows(burn_in=1, unroll=2))
    assert again.observations[0, 0, 0] == 0


@pytest.mark.parametrize('timeout', [False, True])
def test_sb3_reset_observation_cannot_replace_true_final(timeout):
    store = RecurrentPayloadStore(4, 2)
    store.append_sb3(observation=[0, 1], action=[0]*4, reward=1,
                     returned_observation=[999, 999], done=True,
                     info={'terminal_observation': np.array([7, 8], dtype=np.float32),
                           'TimeLimit.truncated': timeout},
                     environment=0, episode=0, step=0, symbol='SPY')
    batch = store.gather(store.index.windows(burn_in=0, unroll=1))
    np.testing.assert_array_equal(batch.observations[0, -1], [7, 8])
    assert batch.bootstrap_mask[0, 0] == timeout
    assert batch.reset_after[0, 0]


@pytest.mark.parametrize('override', [dict(next_observation=[float('nan'), 1]),
                                    dict(observation=[5, 1]), dict(action=[2, 0, 0, 0]),
                                    dict(reward=True), dict(reward=float('inf')),
                                    dict(next_observation=['1', '2'])])
def test_invalid_payload_or_discontinuous_observation_is_atomic(override):
    store = RecurrentPayloadStore(4, 2)
    append(store, 0)
    before = store.index.windows(burn_in=0, unroll=1)
    with pytest.raises(ValueError):
        append(store, 1, **override)
    assert store.index.windows(burn_in=0, unroll=1) == before
    assert append(store, 1).insertion_id == 1


def test_stale_gather_and_mixed_geometry_reject():
    store = RecurrentPayloadStore(3, 2)
    for step in range(3):
        append(store, step)
    old = store.index.windows(burn_in=1, unroll=2)
    mixed = old + store.index.windows(burn_in=0, unroll=2)
    with pytest.raises(ValueError, match='geometry'):
        store.gather(mixed)
    append(store, 3)
    with pytest.raises(ValueError, match='overwritten'):
        store.gather(old)


def test_sb3_missing_terminal_observation_rejects_without_inserting():
    store = RecurrentPayloadStore(3, 2)
    with pytest.raises(ValueError, match='terminal_observation'):
        store.append_sb3(observation=[0, 1], action=[0]*4, reward=0,
                         returned_observation=[999]*2, done=True, info={},
                         environment=0, episode=0, step=0, symbol='SPY')
    assert not store.index.windows(burn_in=0, unroll=1)


def test_actual_trading_env_dummy_vecenv_final_observation():
    from test_causal_sampling import episode
    from rl_stocks.data.generation.datasets import Normalizer
    from rl_stocks.environment.trading.env import TradingEnv
    from stable_baselines3.common.vec_env import DummyVecEnv

    ep = episode()
    ep['features'] = [[float(i)] for i in range(len(ep['bars']))]
    env = DummyVecEnv([lambda: TradingEnv([ep], Normalizer(('x',), (0.,), (1.,)),
                                         ['x'], episode_steps=2)])
    try:
        obs = env.reset()
        store = RecurrentPayloadStore(8, obs.shape[1])
        for step in range(2):
            action = np.zeros((1, 4), dtype=np.float32)
            nxt, reward, done, infos = env.step(action)
            store.append_sb3(observation=obs[0], action=action[0], reward=reward[0],
                             returned_observation=nxt[0], done=bool(done[0]), info=infos[0],
                             environment=0, episode=0, step=step, symbol=ep['symbol'])
            obs = nxt
        assert done[0] and infos[0]['TimeLimit.truncated']
        batch = store.gather(store.index.windows(burn_in=1, unroll=1))
        np.testing.assert_array_equal(batch.observations[0, -1], infos[0]['terminal_observation'])
        assert batch.observations[0, -1, 0] == 2 and obs[0, 0] == 0
        assert batch.bootstrap_mask[0, -1]
    finally:
        env.close()
