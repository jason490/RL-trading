"""No model/environment construction through generic training for Treasury data."""

import pytest

from rl_stocks.training.treasury_guard import reject_unregistered_treasury


@pytest.mark.parametrize(
    "data",
    [
        dict(schema="daily-indicators-v1-treasury-full-v1"),
        dict(manifest={"treasury_context": {}}),
        dict(names=["treasury_2y_pct"]),
    ],
)
def test_schema_manifest_or_feature_detected(data):
    with pytest.raises(ValueError, match="currently disabled"):
        reject_unregistered_treasury(data)


def test_legacy_and_other_registered_families_unchanged():
    for data in [
        dict(schema="daily-indicators-v1"),
        dict(names=["vix_log_level"]),
        dict(manifest={}),
    ]:
        reject_unregistered_treasury(data)


def test_runner_guard_precedes_construction(monkeypatch):
    from rl_stocks.training import runner, dependencies

    monkeypatch.setattr(dependencies, "verify_environment", lambda: {})
    monkeypatch.setattr(
        runner, "read_dataset", lambda p: dict(schema="daily-indicators-v1-treasury-full-v1")
    )

    def forbidden(*a, **k):
        pytest.fail("model/environment construction before Treasury gate")

    monkeypatch.setattr(runner, "create", forbidden)
    monkeypatch.setattr(runner, "TradingEnv", forbidden)
    with pytest.raises(ValueError, match="explicit"):
        runner.train("ignored", "recurrent_ppo")
