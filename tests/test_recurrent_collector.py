"""Synthetic environment and untrained collection; no optimizer or provider access."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from test_causal_sampling import episode
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.rl.recurrent_sac.collection import OnlineRecurrentActor
from rl_stocks.rl.recurrent_sac.networks import RecurrentSACNetworks
from rl_stocks.training.recurrent_collector import RecurrentCollector
from rl_stocks.training.recurrent_payload import RecurrentPayloadStore


def setup(*, warmup=2, episode_steps=3):
    ep = episode()
    ep['features'] = [[float(i)/100]*7 for i in range(len(ep['bars']))]
    names = [f'x{i}' for i in range(7)]
    env = TradingEnv([ep], Normalizer(tuple(names), (0.,)*7, (1.,)*7), names,
                     episode_steps=episode_steps, observation_profile='execution_full')
    with torch.random.fork_rng():
        torch.manual_seed(11)
        networks = RecurrentSACNetworks(21, 8)
    store = RecurrentPayloadStore(30, 21)
    collector = RecurrentCollector(env, networks, store, seed=11, warmup_steps=warmup,
                                    max_episode_steps=episode_steps)
    return env, networks, store, collector


def test_actual_raw_environment_warmup_terminal_storage_and_episode_reset():
    env, networks, store, collector = setup()
    normalizer = deepcopy(env.normalizer)
    before = deepcopy(networks.state_dict())
    try:
        transitions = [collector.collect(policy_revision=0, deterministic=True) for _ in range(6)]
        assert [t.warmup for t in transitions] == [True, True, False, False, False, False]
        assert [t.reference.episode for t in transitions] == [0, 0, 0, 1, 1, 1]
        assert [t.reference.step for t in transitions] == [0, 1, 2, 0, 1, 2]
        assert transitions[2].truncated and transitions[5].truncated
        assert transitions[2].next_observation[0] == np.float32(.03)
        assert transitions[3].observation[0] == 0
        windows = store.index.windows(burn_in=1, unroll=2)
        assert len(windows) == 2
        batch = store.gather(windows)
        assert batch.observations[0, -1, 0] == np.float32(.03)
        assert batch.bootstrap_mask.all()
        rng = np.random.default_rng(11)
        for result in transitions[:2]:
            np.testing.assert_array_equal(result.action, rng.uniform(-1, 1, 4).astype(np.float32))
        assert env.normalizer == normalizer
        for key, value in networks.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        assert all(p.grad is None for p in networks.parameters())
    finally:
        env.close()


def test_history_refresh_after_revision_uses_entire_observed_prefix():
    env, networks, _, collector = setup(warmup=2, episode_steps=5)
    try:
        first = collector.collect(policy_revision=0, deterministic=True)
        second = collector.collect(policy_revision=0, deterministic=True)
        # Manual data-independent tensor perturbation, never an optimizer update.
        with torch.no_grad():
            networks.actor.memory.bias_ih_l0.add_(.01)
        result = collector.collect(policy_revision=1, deterministic=True)
        inputs = torch.tensor(np.stack([first.observation, second.observation, result.observation])).unsqueeze(0)
        with torch.no_grad():
            features, _ = networks.actor.memory(inputs)
            expected = networks.actor.mean(networks.actor.head(features[:, -1:])).tanh().reshape(4).numpy()
        np.testing.assert_array_equal(result.action, expected)
    finally:
        env.close()


def test_fresh_episode_deterministic_action_does_not_carry_prior_symbol_state():
    env, networks, _, collector = setup(warmup=0, episode_steps=2)
    try:
        first = collector.collect(policy_revision=0, deterministic=True)
        collector.collect(policy_revision=0, deterministic=True)
        reset = collector.collect(policy_revision=0, deterministic=True)
        fresh = OnlineRecurrentActor(networks, max_history=2)
        expected = fresh.action(reset.observation, revision=0, episode_start=True, deterministic=True)
        np.testing.assert_array_equal(reset.action, expected)
        np.testing.assert_array_equal(first.action, reset.action)
    finally:
        env.close()


def test_untrained_stochastic_collection_repeats_with_saved_torch_rng():
    rng = torch.get_rng_state().clone()
    try:
        sequences = []
        for _ in range(2):
            env, _, _, collector = setup(warmup=1, episode_steps=3)
            try:
                torch.set_rng_state(rng)
                sequences.append([collector.collect(policy_revision=0).action for _ in range(5)])
            finally:
                env.close()
        np.testing.assert_array_equal(sequences[0], sequences[1])
    finally:
        torch.set_rng_state(rng)


def test_invalid_environment_boundary_failure_poisoned_without_retry(monkeypatch):
    env, _, store, collector = setup(warmup=0)
    original = env.step
    def bad(action):
        nxt, reward, terminated, truncated, info = original(action)
        nxt[0] = float('nan')
        return nxt, reward, terminated, truncated, info
    monkeypatch.setattr(env, 'step', bad)
    try:
        with pytest.raises(ValueError, match='finite'):
            collector.collect(policy_revision=0)
        assert collector.failed and collector.transitions == 0 and env.steps == 1
        assert not store.index.windows(burn_in=0, unroll=1)
        with pytest.raises(RuntimeError, match='no retry'):
            collector.collect(policy_revision=0)
        assert env.steps == 1
    finally:
        env.close()


@pytest.mark.parametrize('change', ['split', 'profile', 'execution', 'reward', 'scale', 'length'])
def test_nonmatching_training_contract_rejects_before_collection(change):
    env, networks, store, _ = setup()
    if change == 'split':
        env.episodes[0]['split'] = 'validation'
    elif change == 'profile':
        env.observation_profile = 'legacy'
    elif change == 'execution':
        env.exposure_execution = 'unit_magnitude'
    elif change == 'reward':
        env.reward_kind = 'other'
    elif change == 'scale':
        env.reward_scale = 10
    else:
        env.episode_steps = None
    try:
        with pytest.raises(ValueError):
            RecurrentCollector(env, networks, store, seed=11, warmup_steps=0, max_episode_steps=3)
        assert not hasattr(env, 'steps')
    finally:
        env.close()


def test_online_actor_enforces_history_limit_and_monotone_revision():
    _, networks, _, _ = setup()
    policy = OnlineRecurrentActor(networks, max_history=1)
    policy.action(np.zeros(21), revision=1, episode_start=True, deterministic=True)
    with pytest.raises(ValueError, match='limit'):
        policy.action(np.zeros(21), revision=1, episode_start=False)
    with pytest.raises(ValueError, match='decrease'):
        policy.action(np.zeros(21), revision=0, episode_start=True)


def test_warmup_does_not_draw_global_torch_noise_and_returned_arrays_do_not_alias_store():
    env, _, store, collector = setup(warmup=2)
    before = torch.get_rng_state().clone()
    try:
        result = collector.collect(policy_revision=0)
        assert torch.equal(before, torch.get_rng_state())
        result.observation[:] = 99
        result.action[:] = 99
        result.next_observation[:] = 99
        replay = store.gather(store.index.windows(burn_in=0, unroll=1))
        assert replay.observations[0, 0, 0] == 0
        assert replay.observations[0, 1, 0] == np.float32(.01)
        assert (np.abs(replay.actions) <= 1).all()
    finally:
        env.close()
