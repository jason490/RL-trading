"""No optimizer calls: construction, routing, target and serialization checks."""
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_stocks.rl.sac.task_policy import TaskSACPolicy, RoutedHead


def policy(role):
    torch.manual_seed(11)
    return TaskSACPolicy(spaces.Box(-np.inf, np.inf, (22,), dtype=np.float32),
                         spaces.Box(-1, 1, (4,), dtype=np.float32), lambda _: .0003,
                         net_arch=[64, 64], task_role=role)


def observations():
    x = torch.zeros((4, 22))
    x[:, -1] = torch.tensor([-1, 1, 1, -1])
    return x


def test_shared_routed_initial_parity_and_rng():
    shared = policy('tagged_shared')
    rng = torch.get_rng_state().clone()
    routed = policy('tagged_routed')
    assert torch.equal(rng, torch.get_rng_state())
    x = observations()
    with torch.no_grad():
        assert torch.equal(shared(x, deterministic=True), routed(x, deterministic=True))
        for a, b in zip(shared.critic(x, torch.zeros(4, 4)),
                        routed.critic(x, torch.zeros(4, 4)), strict=True):
            assert torch.equal(a, b)
        for a, b in zip(routed.critic.parameters(), routed.critic_target.parameters(), strict=True):
            assert torch.equal(a, b)
    assert sum(p.numel() for p in routed.parameters()) > sum(
        p.numel() for p in shared.parameters())


def test_masked_tagged_initial_tensors():
    masked, tagged = policy('masked_shared'), policy('tagged_shared')
    for key, value in masked.state_dict().items():
        assert torch.equal(value, tagged.state_dict()[key])
    x = observations()
    with pytest.raises(ValueError, match='tag'):
        masked(x)
    with pytest.raises(ValueError, match='tag'):
        tagged(torch.zeros(1, 22))


def test_routing_is_per_row_and_not_cached():
    head = RoutedHead(torch.nn.Linear(2, 1))
    with torch.no_grad():
        head.stock.weight.zero_()
        head.stock.bias.fill_(3)
        head.etf.weight.zero_()
        head.etf.bias.fill_(7)
        tag = observations()[:, -1:]
        assert head(torch.zeros(4, 2), tag).flatten().tolist() == [3, 7, 7, 3]
        assert head(torch.zeros(4, 2), -tag).flatten().tolist() == [7, 3, 3, 7]


@pytest.mark.parametrize('role', ['masked_shared', 'tagged_shared', 'tagged_routed'])
def test_policy_save_load(tmp_path, role):
    model = policy(role)
    x = observations() if role != 'masked_shared' else torch.zeros(4, 22)
    path = tmp_path / 'policy.pt'
    model.save(path)
    restored = TaskSACPolicy.load(path)
    assert restored.task_role == role
    with torch.no_grad():
        assert torch.equal(model(x, deterministic=True), restored(x, deterministic=True))
    optimizer_parameters = {id(p) for group in model.actor.optimizer.param_groups
                            for p in group['params']}
    assert optimizer_parameters == {id(p) for p in model.actor.parameters()}


def test_reject_unknown_role():
    with pytest.raises(ValueError, match='role'):
        policy('unknown')


def test_type_wrapper_provenance_and_episode_integrity():
    import gymnasium as gym
    from rl_stocks.environment.observations.task import InstrumentTypeObservation

    class Fixture(gym.Env):
        observation_space = spaces.Box(-np.inf, np.inf, (21,), dtype=np.float32)
        action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        episodes = [{'symbol': 'A', 'bars': [{'session': '2010-01-04'}]}]

        def reset(self, **kwargs):
            self.episode = self.episodes[0]
            return np.arange(21, dtype=np.float32), {}

        def step(self, action):
            return np.arange(21, dtype=np.float32), 0., False, True, {}

    record = {'symbol': 'A', 'kind': 'stock', 'known_from': '2009-01-01',
              'source': 'synthetic fixture, not production classification'}
    wrapper = InstrumentTypeObservation(Fixture(), [record], task_role='tagged_shared')
    record['kind'] = 'etf'
    observation, _ = wrapper.reset()
    assert observation[-1] == -1
    assert np.array_equal(observation[:-1], np.arange(21, dtype=np.float32))
    assert wrapper.step(np.zeros(4))[0][-1] == -1
    with pytest.raises(TypeError):
        wrapper.registry['A'] = ('etf', None, None)
    wrapper.unwrapped.episode = {'symbol': 'OTHER'}
    with pytest.raises(ValueError, match='changed'):
        wrapper.step(np.zeros(4))
    with pytest.raises(ValueError, match='unknown'):
        InstrumentTypeObservation(Fixture(), [], task_role='tagged_shared')
    with pytest.raises(ValueError, match='unavailable'):
        InstrumentTypeObservation(Fixture(), [{**record, 'known_from': '2020-01-01'}],
                                  task_role='tagged_shared')


def test_actual_actor_and_both_critics_route_and_reject_bad_replay_tags():
    model = policy('tagged_routed')
    x = observations()
    with torch.no_grad():
        model.actor.mu.stock.weight.zero_()
        model.actor.mu.etf.weight.zero_()
        model.actor.mu.stock.bias.fill_(-.5)
        model.actor.mu.etf.bias.fill_(.5)
        assert torch.equal(model.actor(x, deterministic=True)[:, 0] > 0, x[:, -1] == 1)
        for index, q in enumerate(model.critic.q_networks):
            q[-1].stock.weight.zero_()
            q[-1].etf.weight.zero_()
            q[-1].stock.bias.fill_(index + 2)
            q[-1].etf.bias.fill_(index + 5)
        for index, values in enumerate(model.critic(x, torch.zeros(4, 4))):
            assert values.flatten().tolist() == [index + 2, index + 5, index + 5, index + 2]
        invalid = x.clone()
        invalid[1, -1] = 0
        with pytest.raises(ValueError, match='tag'):
            model.critic_target(invalid, torch.zeros(4, 4))
        invalid[1, -1] = float('nan')
        with pytest.raises(ValueError, match='nonfinite'):
            model.actor(invalid)
