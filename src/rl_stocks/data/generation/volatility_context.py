"""Align optional retained VIX context without changing a baseline payload."""

from datetime import datetime
import math

from rl_stocks.features.registry.assemble import NAMES as BASE_NAMES
from rl_stocks.features.volatility_context import context


def align_context(data, rows):
    if (
        data["schema"] != "daily-indicators-v1"
        or data["names"] != list(BASE_NAMES)
        or data["normalizer"]["names"] != data["names"]
        or data["manifest"]["included_splits"] != ["train", "validation"]
    ):
        raise ValueError("original seven-feature train/development payload required")
    decisions = {}
    counts = {"train": 0, "validation": 0}
    symbols = {"train": set(), "validation": set()}
    for episode in data["episodes"].values():
        split = episode["split"]
        if split not in counts or not (
            len(episode["bars"]) == len(episode["features"]) == len(episode["atr"])
        ):
            raise ValueError("aligned train/development episodes required")
        symbols[split].add(episode["symbol"])
        for bar, features in zip(episode["bars"], episode["features"], strict=True):
            day = bar["session"]
            first, last = (
                ("2010-01-01", "2020-12-31") if split == "train" else ("2021-01-01", "2023-12-31")
            )
            if (
                bar["symbol"] != episode["symbol"]
                or not first <= day <= last
                or len(features) != 7
                or not all(math.isfinite(v) for v in features)
            ):
                raise ValueError("finite original features and causal split/symbol required")
            stamp = datetime.fromisoformat(bar["available_time"])
            if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
                raise ValueError("UTC baseline decision timestamps required")
            decisions[day] = min(decisions.get(day, stamp), stamp)
            counts[split] += 1
    if not all(counts.values()):
        raise ValueError("nonempty train and development source required")
    sessions = sorted(decisions)
    joined = context(
        rows, sessions, decision_times={d: t.isoformat() for d, t in decisions.items()}
    )
    return joined, counts, {k: sorted(v) for k, v in symbols.items()}
