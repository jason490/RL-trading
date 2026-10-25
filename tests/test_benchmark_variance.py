from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from rl_stocks.core.contracts import Bar
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.reporting.benchmark_variance import (
    discounted,
    decomposition,
    aggregate,
    feasibility,
    benchmark_rewards,
)


def test_discount_linearity_and_covariance_identity_without_fitting():
    a = np.array([0.01, 0.03, -0.02, 0.04])
    b = np.array([0.02, 0.04, -0.01, 0.05])
    np.testing.assert_allclose(
        discounted(a - b, 0.99), discounted(a, 0.99) - discounted(b, 0.99), atol=1e-15
    )
    result = decomposition(a, b)
    assert result["residual_variance"] == pytest.approx(0.0, abs=1e-18)
    assert aggregate([result, result])["ratio_of_mean_within_episode_variances"] == pytest.approx(
        0.0, abs=1e-18
    )
    assert decomposition(np.zeros(4), b)["residual_variance_ratio"] is None
    assert decomposition(-b, b)["residual_variance_ratio"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        decomposition(a, [np.nan] * 4)


def test_all_eight_feasibility_cells_are_required():
    spec = dict(
        feasibility_gate=dict(
            require_both_phases=["selected", "final"],
            require_both_groups=["stock", "etf"],
            require_both_levels=["rewards", "discounted_targets"],
            maximum_mean_within_episode_residual_variance_ratio=0.9,
        )
    )
    cells = {
        p: dict(
            groups={
                g: {
                    level: dict(ratio_of_mean_within_episode_variances=0.8)
                    for level in ["rewards", "discounted_targets"]
                }
                for g in ["stock", "etf"]
            }
        )
        for p in ["selected", "final"]
    }
    assert feasibility(cells, spec)["passed"]
    changed = deepcopy(cells)
    changed["final"]["groups"]["etf"]["rewards"]["ratio_of_mean_within_episode_variances"] = 0.9001
    verdict = feasibility(changed, spec)
    assert not verdict["passed"] and verdict["fixed_subtraction_hypothesis_closed"]
    assert not verdict["training_authorized"] and not verdict["economic_improvement_established"]
    changed["final"]["groups"]["etf"]["rewards"]["ratio_of_mean_within_episode_variances"] = None
    assert not feasibility(changed, spec)["passed"]


def test_benchmark_split_dividend_cost_accounting_and_training_bounds():
    bars = []
    for i, (price, split, dividend) in enumerate(
        [(100.0, 1.0, 0.0), (100.0, 1.0, 0.0), (50.0, 2.0, 1.0)]
    ):
        day = f"2020-12-0{i + 1}"
        bars.append(
            asdict(
                Bar(
                    symbol="A",
                    session=day,
                    open_time=day + "T14:30:00+00:00",
                    event_time=day + "T21:00:00+00:00",
                    available_time=day + "T21:15:00+00:00",
                    retrieved_time=day + "T23:00:00+00:00",
                    provider="fixture",
                    feed="fixture",
                    revision="fixture",
                    open=price,
                    high=price + 1,
                    low=price - 1,
                    close=price,
                    volume=1000.0,
                    split=split,
                    dividend=dividend,
                )
            )
        )
    episode = dict(split="train", bars=bars)
    # Fixed audit cutoff is December2; fixture moves final corporate action there.
    episode["bars"][0]["session"] = "2020-11-30"
    episode["bars"][1]["session"] = "2020-12-01"
    episode["bars"][2]["session"] = "2020-12-02"
    config = SimulationConfig(fee_bps=0.0, impact_bps=0.0)
    rewards, result = benchmark_rewards(episode, config)
    assert result["rows"][-1]["equity"] == 102000.0
    assert float(np.sum(rewards)) == pytest.approx(np.log(1.02), abs=1e-8)
    with_cost = benchmark_rewards(episode, SimulationConfig())[0]
    assert with_cost[0] < 0
    with pytest.raises(ValueError):
        benchmark_rewards(dict(episode, split="validation"), config)
    bad = deepcopy(episode)
    bad["bars"][-1]["session"] = "2024-01-01"
    with pytest.raises(ValueError):
        benchmark_rewards(bad, config)
