"""Independent target arithmetic and split/window isolation; no optimization."""

from copy import deepcopy

from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3.common.buffers import RolloutBuffer

from rl_stocks.reporting.recurrent_values import targets, error_metrics, windows


@pytest.mark.parametrize("terminated", [False, True])
def test_targets_match_sb3_gae_and_independent_discounted_sum(terminated):
    rewards = np.array([0.01, -0.02, 0.03, -0.01], dtype=np.float32)
    values = np.array([0.02, 0.015, 0.025, 0.005], dtype=np.float32)
    tail = 0.0 if terminated else 0.012
    result = targets(rewards, values, 0.99, 0.95, tail, terminated)
    buffer = RolloutBuffer(
        4,
        spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
        spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",
    )
    for i in range(4):
        adjusted = rewards[i] + (0.99 * tail if i == 3 and not terminated else 0)
        buffer.add(
            np.array([0.0]),
            np.array([0.0]),
            adjusted,
            i == 0,
            torch.tensor([values[i]]),
            torch.tensor([0.0]),
        )
    buffer.compute_returns_and_advantage(torch.tensor([0.0]), np.array([True]))
    np.testing.assert_allclose(result["lambda_return"], buffer.returns[:, 0], atol=1e-8)
    for i in range(4):
        discounted = sum(0.99**j * float(rewards[i + j]) for j in range(4 - i))
        assert result["observed_zero_tail"][i] == pytest.approx(discounted)
        assert result["bootstrapped_return"][i] == pytest.approx(
            discounted + 0.99 ** (4 - i) * tail
        )
    assert error_metrics(values, values)["rmse"] == 0
    assert error_metrics(values, values)["explained_variance"] == 1
    assert error_metrics([1, 1], [2, 2])["explained_variance"] is None


def test_terminal_tail_and_nonfinite_inputs_reject():
    with pytest.raises(ValueError):
        targets([1], [1], 0.99, 0.95, 1.0, True)
    with pytest.raises(ValueError):
        targets([np.nan], [1], 0.99, 0.95, 0.0, True)
    with pytest.raises(ValueError):
        error_metrics([1], [np.inf])


def test_window_selects_only_fixed_training_suffix_without_mutation():
    episode = dict(
        symbol="A",
        split="train",
        bars=[dict(session=f"2020-12-0{i}") for i in [1, 2, 3, 4]],
        features=[[i] * 7 for i in range(4)],
        atr=[1.0] * 4,
    )
    data = dict(
        schema="daily-indicators-v1",
        names=list(range(7)),
        episodes={"train": episode, "validation": dict(episode, split="validation")},
    )
    before = deepcopy(data)
    spec = dict(panel=["A"], window_end="2020-12-04", window_transitions=2)
    selected = windows(data, spec)["A"]
    assert selected["bars"] == episode["bars"][-3:] and data == before
    bad = deepcopy(data)
    bad["episodes"]["train"]["split"] = "validation"
    with pytest.raises(ValueError):
        windows(bad, spec)
    bad = deepcopy(data)
    bad["episodes"]["train"]["bars"][-1]["session"] = "2024-01-01"
    with pytest.raises(ValueError):
        windows(bad, spec)
    bad = deepcopy(data)
    bad["episodes"]["train"]["features"].pop()
    with pytest.raises(ValueError):
        windows(bad, spec)


def test_actual_stochastic_collector_matches_critic_and_never_optimizes(monkeypatch):
    import importlib.util
    from dataclasses import asdict
    from pathlib import Path
    import time
    from test_vix_context import construct
    from rl_stocks.core.contracts import Bar
    from rl_stocks.training.vix_context import model_record
    from rl_stocks.training.vix_restoration import same_restored

    spec = importlib.util.spec_from_file_location(
        "value_audit", Path(__file__).parents[1] / "scripts/diagnose_recurrent_values.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    model = construct("legacy")

    def forbidden(*args, **kwargs):
        pytest.fail("local optimization attempted")

    monkeypatch.setattr(type(model), "learn", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    bars = []
    for i in range(4):
        day = f"2020-12-0{i + 1}"
        bars.append(
            asdict(
                Bar(
                    symbol="A",
                    session=day,
                    open_time=day + "T14:30:00+00:00",
                    event_time=day + "T21:00:00+00:00",
                    available_time=day + "T21:15:00+00:00",
                    retrieved_time="2026-10-25T00:00:00+00:00",
                    provider="fixture",
                    feed="fixture",
                    revision="fixture",
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                    volume=1000.0,
                )
            )
        )
    episode = dict(
        symbol="A", split="train", bars=bars, features=[[0.0] * 7 for _ in bars], atr=[1.0] * 4
    )
    data = dict(
        names=[str(i) for i in range(7)],
        normalizer=dict(names=[str(i) for i in range(7)], means=[0.0] * 7, scales=[1.0] * 7),
    )
    contract = dict(
        window_transitions=3,
        observation_profile="execution_masked",
        exposure_execution="unrestricted",
        gamma=0.99,
        gae_lambda=0.95,
    )
    before = model_record(model)
    with torch.random.fork_rng(devices=[]):
        result = script.collect(model, episode, data, contract, 17, time.monotonic() + 10)
        repeated = script.collect(model, episode, data, contract, 17, time.monotonic() + 10)
    assert (
        result == repeated
        and result["steps"] == 3
        and result["truncated"]
        and not result["terminated"]
    )
    assert same_restored(model_record(model), before) and model.num_timesteps == 0
    original = model.policy.predict_values
    monkeypatch.setattr(
        model.policy, "predict_values", lambda *args, **kwargs: original(*args, **kwargs) + 1
    )
    with pytest.raises(AssertionError):
        script.collect(model, episode, data, contract, 17, time.monotonic() + 10)
