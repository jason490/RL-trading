"""Untrained forward/analytic fixtures only; no backward or optimizer calls."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

from rl_stocks.rl.recurrent_sac.networks import RecurrentSACNetworks
from rl_stocks.training.recurrent_payload import RecurrentPayloadStore
from rl_stocks.training.recurrent_sac_losses import (
    actor_objective, bellman_target, critic_objective, frozen_parameters,
    replay_tensors, temperature_loss,
)


@pytest.fixture
def fixture():
    store = RecurrentPayloadStore(20, 3)
    for episode in range(2):
        for step in range(5):
            store.append_raw(observation=[step, episode, .2], next_observation=[step+1, episode, .2],
                             action=[.1, .2, -.3, .4], reward=float(step)/10, environment=0,
                             episode=episode, step=step, symbol='SPY',
                             terminated=episode == 0 and step == 4,
                             truncated=episode == 1 and step == 4)
    batch = store.gather(store.index.windows(burn_in=2, unroll=3))
    with torch.random.fork_rng():
        torch.manual_seed(11)
        networks = RecurrentSACNetworks(3, 8)
    return networks, batch


def test_bellman_numpy_oracle_terminal_and_truncated():
    rewards = torch.tensor([[1., 2., 3.]])
    bootstrap = torch.tensor([[True, False, True]])
    q1 = torch.tensor([[2., 100., 6.]])
    q2 = torch.tensor([[4., 200., 5.]])
    logp = torch.tensor([[-1., -2., -3.]])
    actual = bellman_target(rewards, bootstrap, q1, q2, logp, torch.tensor(.2), .9)
    expected = np.array([[1 + .9*(2+.2), 2, 3+.9*(5+.6)]])
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6)
    assert not actual.requires_grad


def test_native_sb3_density_with_batch_time_flattening(fixture):
    networks, _ = fixture
    features = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8)/50
    noise = torch.full((2, 3, 4), .25)
    with torch.no_grad():
        actions, logp = networks.actor.sample(features, noise)
        latent = networks.actor.head(features)
        mean = networks.actor.mean(latent)
        logstd = networks.actor.log_std(latent).clamp(-20, 2)
        gaussian = mean + logstd.exp()*noise
        native = SquashedDiagGaussianDistribution(4)
        native.proba_distribution(mean.reshape(-1, 4), logstd.reshape(-1, 4))
        expected = native.log_prob(torch.tanh(gaussian).reshape(-1, 4), gaussian.reshape(-1, 4))
    torch.testing.assert_close(logp.reshape(-1), expected, rtol=0, atol=0)
    assert actions.shape == (2, 3, 4) and logp.shape == (2, 3)
    assert torch.all(actions.abs() <= 1) and torch.isfinite(logp).all()


def test_losses_independent_manual_full_history_oracle_and_no_mutation(fixture):
    networks, batch = fixture
    replay = replay_tensors(batch)
    before = deepcopy(networks.state_dict())
    rng = torch.get_rng_state().clone()
    noise = torch.zeros(2, 3, 4)
    with torch.no_grad():
        critic_loss, targets, current = critic_objective(networks, replay, torch.tensor(.1),
                                                       next_noise=noise)
        actor_loss, logp, actions = actor_objective(networks, replay, torch.tensor(.1), noise=noise)
        # Independent complete episode forward; omit helper chunk/shift logic.
        ah, _ = networks.actor.memory(replay.observations)
        next_actions, next_logp = networks.actor.sample(ah[:, 3:], noise)
        target_q = []
        policy_q = []
        manual_current = []
        for target, critic in zip(networks.targets, networks.critics):
            th, _ = target.memory(replay.observations)
            ch, _ = critic.memory(replay.observations)
            target_q.append(target(th[:, 3:], next_actions))
            policy_q.append(critic(ch[:, 2:-1], actions))
            manual_current.append(critic(ch[:, 2:-1], replay.actions[:, 2:]))
        expected_targets = (replay.rewards[:, 2:] + .99*replay.bootstrap[:, 2:]
                            * (torch.minimum(*target_q) - .1*next_logp))
        expected_critic = .5*sum((q-expected_targets).square().mean() for q in manual_current)
        expected_actor = (.1*logp - torch.minimum(*policy_q)).mean()
    torch.testing.assert_close(targets, expected_targets)
    torch.testing.assert_close(critic_loss, expected_critic)
    torch.testing.assert_close(actor_loss, expected_actor)
    assert all(torch.isfinite(q).all() for q in current)
    for key, value in networks.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in networks.parameters())


def test_gradient_graph_ownership_without_local_differentiation(fixture):
    networks, batch = fixture
    replay = replay_tensors(batch)
    alpha_log = torch.tensor(0., requires_grad=True)
    noise = torch.zeros(2, 3, 4)
    closs, targets, _ = critic_objective(networks, replay, alpha_log.exp(), next_noise=noise)
    aloss, logp, _ = actor_objective(networks, replay, alpha_log.exp(), noise=noise)
    eloss = temperature_loss(alpha_log, logp)
    assert closs.requires_grad and aloss.requires_grad and eloss.requires_grad
    assert not targets.requires_grad
    assert all(p.requires_grad for p in networks.critics.parameters())
    assert not any(p.requires_grad for p in networks.targets.parameters())
    assert all(p.grad is None for p in networks.parameters()) and alpha_log.grad is None
    expected = -(alpha_log.detach().item() * (logp.detach().numpy() - 4)).mean()
    assert eloss.detach().item() == pytest.approx(expected)


def test_temperature_numpy_value_and_frozen_flags_restore_on_error(fixture):
    networks, _ = fixture
    logp = torch.tensor([[-.2, -.4]], requires_grad=True)
    actual = temperature_loss(torch.tensor(.3, requires_grad=True), logp)
    assert actual.detach().item() == pytest.approx(-.3*np.mean([-.2-4, -.4-4]))
    first = next(networks.critics.parameters())
    first.requires_grad_(False)
    before = [p.requires_grad for p in networks.critics.parameters()]
    with pytest.raises(RuntimeError):
        with frozen_parameters(networks.critics):
            assert not any(p.requires_grad for p in networks.critics.parameters())
            raise RuntimeError('fixture')
    assert before == [p.requires_grad for p in networks.critics.parameters()]


@pytest.mark.parametrize('field', ['loss', 'reset', 'bootstrap', 'dtype', 'action'])
def test_corrupt_replay_contract_rejects(fixture, field):
    _, batch = fixture
    if field == 'loss':
        value = batch.loss_mask.copy()
        value[0, 0] = True
        bad = replace(batch, loss_mask=value)
    elif field == 'reset':
        value = batch.reset_after.copy()
        value[0, 0] = True
        bad = replace(batch, reset_after=value)
    elif field == 'bootstrap':
        value = batch.bootstrap_mask.copy()
        value[0, 0] = False
        bad = replace(batch, bootstrap_mask=value)
    elif field == 'dtype':
        bad = replace(batch, observations=batch.observations.astype(np.float64))
    else:
        value = batch.actions.copy()
        value[0, 0, 0] = 2
        bad = replace(batch, actions=value)
    with pytest.raises(ValueError):
        replay_tensors(bad)
