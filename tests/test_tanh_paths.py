from copy import deepcopy
import math

import numpy as np
import pytest

from test_task_environment import payload
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.evaluation import tanh_paths


class Direct:
    def predict(self, observation):
        return (.5, 0., 0., 0.)


class Frozen(Direct):
    def __init__(self, data):
        self.normalizer = normalizer_from(data["normalizer"])

    def observation(self, features, names, portfolio):
        return tuple(np.asarray(self.normalizer.transform(features, names) + list(portfolio),
                                dtype=np.float32).tolist())

    def predict_features(self, *a, **kw):
        return self.predict(None)


def fixture():
    data, _ = payload()
    episode = deepcopy(data["episodes"]["SPY"])
    for key in ("bars", "features", "atr"):
        episode[key] = episode[key][:35]
    return data, episode


def test_full_paired_account_paths_are_lossless_and_repeatable(tmp_path):
    data, episode = fixture()
    a = tanh_paths.paired_episode(episode, data, Frozen(data), Direct(), tmp_path / "a.jsonl")
    b = tanh_paths.paired_episode(episode, data, Frozen(data), Direct(), tmp_path / "b.jsonl")
    assert a == b and a["decisions"] == 34
    assert a["exact_paired_path"] and a["metrics"]["costs"] > 0
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert len((tmp_path / "a.jsonl").read_text().splitlines()) == 34


def test_action_corruption_and_nontraining_rows_rejected(tmp_path):
    data, episode = fixture()
    class Wrong(Frozen):
        def predict_features(self, *a, **kw):
            return (-.5, 0., 0., 0.)
    with pytest.raises(ValueError, match="action mismatch"):
        tanh_paths.paired_episode(episode, data, Wrong(data), Direct(), tmp_path / "bad.jsonl")
    episode["split"] = "validation"
    with pytest.raises(ValueError, match="training-only"):
        tanh_paths.paired_episode(episode, data, Frozen(data), Direct(), tmp_path / "validation.jsonl")
    assert not (tmp_path / "validation.jsonl").exists()


def test_accounting_difference_fails_exactly(tmp_path, monkeypatch):
    data, episode = fixture()
    real = tanh_paths.TradingEnv
    created = []
    def factory(*args, **kwargs):
        env = real(*args, **kwargs)
        if created:
            original = env.step
            def corrupted(action):
                observation, reward, terminated, truncated, info = original(action)
                return observation, reward, terminated, truncated, dict(info, equity=info["equity"] + 1e-8)
            env.step = corrupted
        created.append(env)
        return env
    monkeypatch.setattr(tanh_paths, "TradingEnv", factory)
    with pytest.raises(ValueError, match="accounting/event/state"):
        tanh_paths.paired_episode(episode, data, Frozen(data), Direct(), tmp_path / "changed.jsonl")


def test_lossless_preserves_signed_zero_and_rejects_nonfinite():
    assert tanh_paths.lossless(0.) != tanh_paths.lossless(-0.)
    assert tanh_paths.lossless(math.nextafter(1., 2.)) != tanh_paths.lossless(1.)
    for value in (float("nan"), float("inf"), object()):
        with pytest.raises(ValueError):
            tanh_paths.lossless(value)
    assert len(tanh_paths.protocol()["cases"]) == 4
