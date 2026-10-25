"""Actual SB3 buffers filled manually; no environment learning or optimizer step."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from test_gae_initialization import model
from rl_stocks.training.gae_rollouts import validate_arrays, read_arrays, GaeRolloutAudit


def fixture(arm):
    m = model(arm)
    b = m.rollout_buffer
    n = m.n_steps
    raw = np.sin(np.arange(n, dtype=np.float32) / 11).reshape(n, 1) * 0.01
    values = np.cos(np.arange(n, dtype=np.float32) / 7) * 0.03
    dones = np.zeros((n, 1), dtype=bool)
    dones[[31, 63]] = True
    timeouts = np.zeros_like(dones)
    timeouts[31] = True
    tails = np.zeros((n, 1), dtype=np.float32)
    tails[31] = 0.04
    starts = np.zeros((n, 1), dtype=np.float32)
    starts[[0, 32, 64]] = 1
    state = RNNStates(
        pi=(torch.zeros((1, 1, 64)), torch.zeros((1, 1, 64))),
        vf=(torch.zeros((1, 1, 64)), torch.zeros((1, 1, 64))),
    )
    for i in range(n):
        reward = raw[i] + np.float32(m.gamma) * tails[i]
        b.add(
            np.zeros((1, 21), dtype=np.float32),
            np.zeros((1, 4), dtype=np.float32),
            reward,
            starts[i],
            torch.tensor([values[i]]),
            torch.tensor([0.0]),
            lstm_states=state,
        )
    last = np.array([0.02], dtype=np.float32)
    b.compute_returns_and_advantage(torch.from_numpy(last), np.array([False]))
    result = {
        name: np.array(getattr(b, name), copy=True)
        for name in ["rewards", "values", "episode_starts", "returns", "advantages"]
    }
    result.update(
        raw_rewards=raw,
        timeout_values=tails,
        step_dones=dones,
        timeouts=timeouts,
        dones=np.array([False]),
        last_values=last,
    )
    return result


@pytest.mark.parametrize("arm", ["control", "mc_trace"])
def test_real_buffer_gae_timeout_and_mc_reconstruction(arm):
    arrays = fixture(arm)
    before = deepcopy(arrays)
    record = validate_arrays(arrays, arm, True)
    assert record["timeouts"] == 1 and record["episode_starts"] == 3
    assert record["candidate_mc_identity_checked"] is (arm == "mc_trace")
    if arm == "mc_trace":
        assert record["maximum_segmented_mc_return_error"] < 2e-6
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], before[name])


@pytest.mark.parametrize(
    "change",
    [
        lambda a: a["episode_starts"].__setitem__((32, 0), 0),
        lambda a: a["timeouts"].__setitem__((5, 0), True),
        lambda a: a["returns"].__setitem__((1, 0), 1.0),
        lambda a: a["advantages"].__setitem__((1, 0), float("nan")),
        lambda a: a["rewards"].__setitem__((31, 0), a["raw_rewards"][31, 0]),
        lambda a: a.update(values=a["values"].astype(np.float64)),
        lambda a: a.update(extra=np.zeros(1)),
    ],
)
def test_checksum_valid_arrays_cannot_hide_math_mask_or_schema_errors(change):
    arrays = fixture("mc_trace")
    change(arrays)
    with pytest.raises((ValueError, AssertionError)):
        validate_arrays(arrays, "mc_trace", True)


@pytest.mark.parametrize("bad_target", [False, True])
def test_actual_callback_records_pre_update_arrays_without_touching_model(
    tmp_path, monkeypatch, bad_target
):
    from rl_stocks.training.gae_initialization import model_record

    m = model("mc_trace")
    m.policy.set_training_mode(False)

    def forbidden(*args, **kwargs):
        pytest.fail("local optimization attempted")

    monkeypatch.setattr(type(m), "learn", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    callback = GaeRolloutAudit(tmp_path, "mc_trace", True)
    callback.init_callback(m)
    callback._on_rollout_start()
    obs = np.zeros((1, 21), dtype=np.float32)
    state = m._last_lstm_states
    for i in range(m.n_steps):
        raw = np.array([0.001], dtype=np.float32)
        done = np.array([i == 31])
        info = {}
        if done[0]:
            info = {"TimeLimit.truncated": True, "terminal_observation": obs[0]}
        callback.locals = dict(rewards=raw, dones=done, infos=[info], lstm_states=state)
        assert callback._on_step()
        reward = raw + np.float32(m.gamma) * callback.tails[-1]
        m.rollout_buffer.add(
            obs,
            np.zeros((1, 4), dtype=np.float32),
            reward,
            np.array([i in [0, 32]]),
            torch.tensor([0.01]),
            torch.tensor([0.0]),
            lstm_states=state,
        )
    m._last_obs = obs
    m._last_episode_starts = np.array([False])
    m.num_timesteps = 128
    with torch.no_grad():
        last = m.policy.predict_values(
            torch.from_numpy(obs), state.vf, torch.tensor([False], dtype=torch.float32)
        )
    m.rollout_buffer.compute_returns_and_advantage(last, np.array([False]))
    before = model_record(m)
    if bad_target:
        import json

        m.rollout_buffer.returns[0, 0] += 1.0
        with pytest.raises(AssertionError):
            callback._on_rollout_end()
        failure = json.loads((tmp_path / "gae-rollout-failure.json").read_text())
        assert (tmp_path / failure["arrays_file"]).is_file()
        assert failure["before_optimizer_update"] and not callback.records
        assert model_record(m) == before
        return
    callback._on_rollout_end()
    after = model_record(m)
    assert before == after and not m.policy.optimizer.state
    record = callback.records[0]
    raw = (tmp_path / record["arrays_file"]).read_bytes()
    assert validate_arrays(read_arrays(raw), "mc_trace", True)["candidate_mc_identity_checked"]
    assert record["before_optimizer_update"] and record["steps"] == 128
    with pytest.raises(ValueError):
        callback._on_rollout_end()
    with pytest.raises(ValueError):
        callback.summary()


def test_buffer_may_begin_mid_episode_without_new_start():
    arrays = fixture("mc_trace")
    arrays["episode_starts"][0, 0] = 0.0
    assert validate_arrays(arrays, "mc_trace", True)["episode_starts"] == 2
