"""Native untrained GRU forwards only: no backward, optimizer, learner or data API."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from rl_stocks.training.recurrent_history import ROLES, encode_history, sac_histories


@pytest.fixture
def fixture():
    with torch.random.fork_rng():
        torch.manual_seed(11)
        encoders = {role: nn.GRU(3, 5, batch_first=True) for role in ROLES}
        obs = torch.randn(2, 9, 3)
    return encoders, obs


@pytest.mark.parametrize('burn_in', [0, 1, 4, 7])
def test_shifted_chunk_matches_full_causal_forward_and_detaches_prefix(fixture, burn_in):
    encoders, obs = fixture
    encoder = encoders['actor']
    before = deepcopy(encoder.state_dict())
    state = torch.get_rng_state().clone()
    result = encode_history(encoder, obs, burn_in=burn_in)
    with torch.no_grad():
        full, final = encoder(obs)
    torch.testing.assert_close(result.current, full[:, burn_in:-1], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(result.next, full[:, burn_in + 1:], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(result.final_hidden, final, rtol=1e-6, atol=1e-7)
    assert result.current.shape == (2, 8 - burn_in, 5)
    assert not result.current.requires_grad and result.burn_in_hidden.grad_fn is None
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert torch.equal(state, torch.get_rng_state())
    assert all(p.grad is None for p in encoder.parameters())


def test_next_input_must_not_restart_from_current_prefix_state(fixture):
    encoders, obs = fixture
    encoder = encoders['actor']
    result = encode_history(encoder, obs, burn_in=3)
    with torch.no_grad():
        wrong, _ = encoder(obs[:, 4:], result.burn_in_hidden)
    assert not torch.allclose(result.next, wrong, rtol=1e-6, atol=1e-7)


def test_future_observations_cannot_change_earlier_features(fixture):
    encoders, obs = fixture
    encoder = encoders['actor']
    changed = obs.clone()
    changed[:, 6:] += 100
    a = encode_history(encoder, obs, burn_in=2)
    b = encode_history(encoder, changed, burn_in=2)
    torch.testing.assert_close(a.current[:, :4], b.current[:, :4], rtol=0, atol=0)
    assert not torch.equal(a.next[:, 3:], b.next[:, 3:])


def test_roles_and_batch_lanes_reconstruct_independently(fixture):
    encoders, obs = fixture
    results = sac_histories(encoders, obs, burn_in=3)
    for role, result in results.items():
        for lane in range(2):
            alone = encode_history(encoders[role], obs[lane:lane+1], burn_in=3)
            torch.testing.assert_close(result.current[lane:lane+1], alone.current,
                                       rtol=1e-6, atol=1e-7)
    assert not torch.equal(results['critic_1'].current, results['actor'].current)
    # Changing one role cannot change another role, nor leave a carried cache.
    replacement = dict(encoders)
    replacement['actor'] = deepcopy(encoders['actor'])
    with torch.no_grad():
        replacement['actor'].bias_ih_l0.add_(1)
    altered = sac_histories(replacement, obs, burn_in=3)
    torch.testing.assert_close(altered['critic_1'].current, results['critic_1'].current,
                               rtol=0, atol=0)
    assert not torch.equal(altered['actor'].current, results['actor'].current)


def test_graph_boundary_without_running_backward(fixture):
    encoders, obs = fixture
    obs.requires_grad_(True)
    results = sac_histories(encoders, obs, burn_in=3, track_grad=True)
    for role, result in results.items():
        assert result.current.requires_grad == (not role.startswith('target_'))
        assert result.burn_in_hidden.grad_fn is None and not result.burn_in_hidden.requires_grad
        assert all(p.grad is None for p in encoders[role].parameters())
    assert obs.grad is None
    with torch.no_grad():
        result = encode_history(encoders['actor'], obs, burn_in=3, track_grad=True)
    assert not result.current.requires_grad


def test_tied_encoder_storage_is_rejected(fixture):
    encoders, obs = fixture
    encoders['critic_1'] = encoders['actor']
    with pytest.raises(ValueError, match='independent'):
        sac_histories(encoders, obs, burn_in=2)


@pytest.mark.parametrize('burn_in', [-1, True, 8, 9])
def test_invalid_burn_in(fixture, burn_in):
    encoders, obs = fixture
    with pytest.raises(ValueError):
        encode_history(encoders['actor'], obs, burn_in=burn_in)


@pytest.mark.parametrize('kind', ['nan', 'double', 'shape', 'empty', 'wrong_width'])
def test_invalid_inputs_reject(fixture, kind):
    encoders, obs = fixture
    if kind == 'nan':
        obs[0, 0, 0] = float('nan')
    elif kind == 'double':
        obs = obs.double()
    elif kind == 'shape':
        obs = obs[0]
    elif kind == 'empty':
        obs = obs[:0]
    else:
        obs = obs[:, :, :2]
    with pytest.raises(ValueError):
        encode_history(encoders['actor'], obs, burn_in=2)


@pytest.mark.parametrize('kwargs', [dict(bidirectional=True, batch_first=True),
                                   dict(batch_first=False),
                                   dict(batch_first=True, num_layers=2, dropout=.1)])
def test_unsupported_recurrent_semantics_reject(fixture, kwargs):
    _, obs = fixture
    with torch.random.fork_rng():
        encoder = nn.GRU(3, 5, **kwargs)
    with pytest.raises(ValueError):
        encode_history(encoder, obs, burn_in=2)


def test_payload_final_history_and_new_episode_start_do_not_share_memory(fixture):
    from rl_stocks.training.recurrent_payload import RecurrentPayloadStore

    encoders, _ = fixture
    store = RecurrentPayloadStore(10, 3)
    for episode in range(2):
        base = episode * 100
        for step in range(3):
            store.append_raw(observation=[base + step, 1, 2], action=[0]*4, reward=0.,
                             next_observation=[base + step + 1, 1, 2], environment=0,
                             episode=episode, step=step, symbol=('SPY', 'QQQ')[episode],
                             terminated=False, truncated=step == 2)
    batch = store.gather(store.index.windows(burn_in=1, unroll=2))
    inputs = torch.tensor(batch.observations)
    results = sac_histories(encoders, inputs, burn_in=batch.burn_in)
    for role in ROLES:
        with torch.no_grad():
            independent, _ = encoders[role](inputs[1:2])
        torch.testing.assert_close(results[role].current[1:2], independent[:, 1:-1],
                                   rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(results[role].next[1:2], independent[:, 2:],
                                   rtol=1e-6, atol=1e-7)
    assert inputs[1, -1, 0] == 103  # true final observation; no subsequent reset
