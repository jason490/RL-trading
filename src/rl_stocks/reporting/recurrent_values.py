"""Censored/bootstrapped value-target diagnostics, never an accuracy guarantee."""

import json
from pathlib import Path

import numpy as np

from rl_stocks.core.contracts import digest

PROTOCOL_SHA256 = "e96a850bbf3c64cba212bf58f5ceb75a4df245fbb965aab82395be05d3a8c710"
ROOT = Path(__file__).resolve().parents[3]


def protocol():
    spec = json.loads((ROOT / "configs/experiments/recurrent_value_audit.json").read_text())
    if digest(spec) != PROTOCOL_SHA256:
        raise ValueError("fixed recurrent value audit protocol required")
    return spec


def windows(data, spec):
    if data["schema"] != "daily-indicators-v1" or len(data["names"]) != 7:
        raise ValueError("original seven-field training data required")
    result = {}
    for symbol in spec["panel"]:
        matches = [
            e
            for e in data["episodes"].values()
            if e["split"] == "train"
            and e["symbol"] == symbol
            and e["bars"][-1]["session"] == spec["window_end"]
        ]
        if len(matches) != 1:
            raise ValueError("one fixed-end training episode per symbol required")
        episode = matches[0]
        n = spec["window_transitions"] + 1
        if len(episode["bars"]) < n:
            raise ValueError("complete fixed training window required")
        selected = dict(episode)
        for name in ("bars", "features", "atr"):
            if len(episode[name]) != len(episode["bars"]):
                raise ValueError("aligned training observations required")
            selected[name] = episode[name][-n:]
        sessions = [b["session"] for b in selected["bars"]]
        if sessions != sorted(set(sessions)) or any(s > spec["window_end"] for s in sessions):
            raise ValueError("unique chronological training-only window required")
        result[symbol] = selected
    return result


def targets(rewards, values, gamma, gae_lambda, tail_value, terminated):
    rewards, values = np.asarray(rewards, dtype=float), np.asarray(values, dtype=float)
    if (
        rewards.ndim != 1
        or not len(rewards)
        or values.shape != rewards.shape
        or not np.isfinite(rewards).all()
        or not np.isfinite(values).all()
        or type(terminated) is not bool
        or not np.isfinite(tail_value)
        or not 0 < gamma < 1
        or not 0 <= gae_lambda <= 1
        or terminated
        and tail_value != 0
    ):
        raise ValueError("finite aligned rewards/values and correct terminal tail required")
    zero = np.empty_like(rewards)
    boot = np.empty_like(rewards)
    advantages = np.empty_like(rewards)
    next_zero, next_boot, next_advantage = 0.0, float(tail_value), 0.0
    for i in reversed(range(len(rewards))):
        zero[i] = rewards[i] + gamma * next_zero
        boot[i] = rewards[i] + gamma * next_boot
        next_value = values[i + 1] if i + 1 < len(values) else tail_value
        delta = rewards[i] + gamma * next_value - values[i]
        advantages[i] = delta + gamma * gae_lambda * next_advantage
        next_zero, next_boot, next_advantage = zero[i], boot[i], advantages[i]
    return dict(
        observed_zero_tail=zero,
        bootstrapped_return=boot,
        gae_advantage=advantages,
        lambda_return=values + advantages,
    )


def error_metrics(values, target):
    values, target = np.asarray(values, dtype=float), np.asarray(target, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or target.shape != values.shape
        or not np.isfinite(values).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("finite aligned nonempty value/target observations required")
    error = values - target
    variance = float(np.var(target))
    return dict(
        count=len(values),
        predicted_mean=float(values.mean()),
        target_mean=float(target.mean()),
        mean_prediction_minus_target=float(error.mean()),
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))),
        target_std=float(np.std(target)),
        explained_variance=None if variance == 0 else float(1 - np.var(error) / variance),
    )
