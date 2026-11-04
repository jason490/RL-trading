from copy import deepcopy

import pytest
from rl_stocks.core.contracts import digest
from rl_stocks.training import trpo_smoke
from test_task_environment import payload


def test_colab_guard_precedes_data_and_output(tmp_path):
    with pytest.raises(RuntimeError, match="Colab"):
        trpo_smoke.run({}, tmp_path / "artifacts" / "smoke")
    assert not (tmp_path / "artifacts").exists()


def test_protocol_work_counts_are_exact_and_performance_is_unadmitted():
    spec = trpo_smoke.protocol()
    assert len(spec["cases"]) == 4 and not spec["allocation_admitted"]
    assert sum(c["steps"] for c in spec["cases"]) == 6048
    for c in spec["cases"]:
        assert c["steps"] == c["nominal_updates"] * c["rollout"]
        assert c["adam_steps"] == c["nominal_updates"] * 10 * (c["rollout"] // c["batch"])


def test_split_and_preprocessing_checks_and_environment_training_filter(monkeypatch):
    data, _ = payload()
    spec = {"dataset": digest(data), "normalizer_sha256": digest(data["normalizer"])}
    monkeypatch.setattr(trpo_smoke, "protocol", lambda: spec)
    assert trpo_smoke.validate_dataset(data) is data
    bad = deepcopy(data)
    bad["episodes"]["SPY"]["split"] = "test"
    spec["dataset"] = digest(bad)
    with pytest.raises(ValueError, match="retired"):
        trpo_smoke.validate_dataset(bad)
    bad = deepcopy(data)
    bad["normalizer"]["means"] = list(bad["normalizer"]["means"])
    bad["normalizer"]["means"][0] = 1.0
    spec["dataset"] = digest(bad)
    with pytest.raises(ValueError):
        trpo_smoke.validate_dataset(bad)
    validation = deepcopy(data["episodes"]["SPY"])
    validation["split"] = "validation"
    data["episodes"]["SPY-val"] = validation
    spec["dataset"] = digest(data)
    captured = {}

    def environment(episodes, norm, names, **kwargs):
        captured.update(episodes=episodes, normalizer=norm, names=names, settings=kwargs)
        return "only-training-env"

    monkeypatch.setattr(trpo_smoke, "TradingEnv", environment)
    assert trpo_smoke.build_environment(data, smoke=True) == "only-training-env"
    assert len(captured["episodes"]) == 5 and all(
        e["split"] == "train" for e in captured["episodes"]
    )
    assert captured["settings"]["episode_steps"] == 128
    assert captured["settings"]["observation_profile"] == "execution_full"
    assert trpo_smoke.build_environment(data, smoke=False) == "only-training-env"
    assert captured["settings"]["episode_steps"] == 512
    with pytest.raises(ValueError):
        trpo_smoke.build_environment(data, smoke=1)
