"""Observed variance decomposition only; no fitted control variate or reward change."""

import hashlib
import json
from pathlib import Path

import numpy as np

from rl_stocks.core.contracts import digest
from rl_stocks.environment.rewards.equity import equity_reward
from rl_stocks.evaluation.runner import baselines

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SHA256 = "509cce09ebc4c4360d6cdae83241f3d210eefd90ec6465a868cb68adf8a9c513"


def protocol():
    value = json.loads((ROOT / "configs/experiments/benchmark_variance_audit.json").read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError("fixed benchmark-variance protocol required")
    for name, sha in value["implementation_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise ValueError("bound benchmark/window implementation changed")
    return value


def benchmark_rewards(episode, config):
    if episode["split"] != "train" or any(b["session"] > "2020-12-02" for b in episode["bars"]):
        raise ValueError("fixed training-only benchmark window required")
    result = baselines(episode, config)["buy_hold"]
    previous = config.initial_equity
    rewards = []
    for row in result["rows"]:
        rewards.append(float(np.float32(equity_reward(previous, row["equity"]))))
        previous = row["equity"]
    if not rewards or not np.isfinite(rewards).all():
        raise ValueError("finite complete benchmark rewards required")
    return np.asarray(rewards), result


def discounted(rewards, gamma):
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 1 or not len(rewards) or not np.isfinite(rewards).all() or not 0 < gamma < 1:
        raise ValueError("finite rewards and fixed discount required")
    result = np.empty_like(rewards)
    tail = 0.0
    for i in reversed(range(len(rewards))):
        tail = rewards[i] + gamma * tail
        result[i] = tail
    return result


def decomposition(policy, benchmark):
    policy, benchmark = np.asarray(policy, dtype=float), np.asarray(benchmark, dtype=float)
    if (
        policy.ndim != 1
        or not len(policy)
        or benchmark.shape != policy.shape
        or not np.isfinite(policy).all()
        or not np.isfinite(benchmark).all()
    ):
        raise ValueError("finite aligned policy/benchmark series required")
    residual = policy - benchmark
    vp, vb, vr = float(np.var(policy)), float(np.var(benchmark)), float(np.var(residual))
    covariance = float(np.mean((policy - policy.mean()) * (benchmark - benchmark.mean())))
    if not np.isclose(vr, vp + vb - 2 * covariance, atol=1e-12, rtol=1e-10):
        raise ValueError("variance decomposition failed")
    return dict(
        count=len(policy),
        policy_variance=vp,
        benchmark_variance=vb,
        residual_variance=vr,
        covariance=covariance,
        policy_mean=float(policy.mean()),
        benchmark_mean=float(benchmark.mean()),
        residual_mean=float(residual.mean()),
        residual_variance_ratio=None if vp == 0 else vr / vp,
        correlation=None if vp == 0 or vb == 0 else covariance / np.sqrt(vp * vb),
    )


def aggregate(records):
    if not records:
        raise ValueError("nonempty fixed episode group required")
    vp = float(np.mean([r["policy_variance"] for r in records]))
    vb = float(np.mean([r["benchmark_variance"] for r in records]))
    vr = float(np.mean([r["residual_variance"] for r in records]))
    cov = float(np.mean([r["covariance"] for r in records]))
    return dict(
        episodes=len(records),
        mean_policy_variance=vp,
        mean_benchmark_variance=vb,
        mean_residual_variance=vr,
        mean_covariance=cov,
        ratio_of_mean_within_episode_variances=None if vp == 0 else vr / vp,
    )


def feasibility(cells, spec):
    gate = spec["feasibility_gate"]
    checks = {}
    if set(cells) != set(gate["require_both_phases"]):
        raise ValueError("both fixed phases required")
    for phase in gate["require_both_phases"]:
        for group in gate["require_both_groups"]:
            for level in gate["require_both_levels"]:
                value = cells[phase]["groups"][group][level][
                    "ratio_of_mean_within_episode_variances"
                ]
                if value is not None and (not np.isfinite(value) or value < 0):
                    raise ValueError("finite nonnegative variance ratio required")
                checks[phase + ":" + group + ":" + level] = dict(
                    ratio=value,
                    passed=value is not None
                    and value <= gate["maximum_mean_within_episode_residual_variance_ratio"],
                )
    passed = all(r["passed"] for r in checks.values())
    return dict(
        passed=passed,
        checks=checks,
        training_authorized=False,
        model_promoted=False,
        economic_improvement_established=False,
        coefficient_search_authorized=False,
        fixed_subtraction_hypothesis_closed=not passed,
    )
