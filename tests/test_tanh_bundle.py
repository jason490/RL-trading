from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from test_canonical_tanh import fixture
from rl_stocks.core.contracts import canonical, digest
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.inference.canonical_tanh_policy import CanonicalTanhPolicy
from rl_stocks.inference.tanh_bundle import FrozenTanhPolicy, freeze
from rl_stocks.training import trpo_smoke


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    names = [f"field{i}" for i in range(7)]
    data = {"schema": "daily-indicators-v1", "names": names,
            "normalizer": {"names": names, "means": [0.] * 7, "scales": [2.] * 7},
            "manifest": {"universe": ["SPY"], "included_splits": ["train", "validation"],
                         "provenance": {"capabilities": {"feed": "fixture"}}}}
    monkeypatch.setattr(trpo_smoke, "validate_dataset", lambda value: value)
    monkeypatch.setattr(trpo_smoke, "protocol", lambda: {"dataset": digest(data),
                       "normalizer_sha256": digest(data["normalizer"])})
    actor = tmp_path / "actor.json"
    actor.write_bytes(canonical(fixture()))
    path = freeze(actor, data, tmp_path / "bundles", algorithm="trpo")
    return path, actor, data


def load(path, **kw):
    return FrozenTanhPolicy(path, feature_schema="daily-indicators-v1", feed="fixture", **kw)


def test_full_preprocessing_reference_parity_and_immutability(bundle):
    path, actor, data = bundle
    policy = load(path, expected_bundle_id=path.name)
    features, portfolio = [1.] * 7, [0.] * 14
    expected = np.asarray(normalizer_from(data["normalizer"]).transform(features, data["names"])
                          + portfolio, dtype=np.float32)
    assert tuple(expected) == policy.observation(features, data["names"], portfolio)
    assert policy.predict(expected) == CanonicalTanhPolicy.load(actor).predict(expected)
    action = policy.predict_features(features, data["names"], portfolio, symbol="SPY",
                                     available_time="2020-01-02T21:15:00+00:00",
                                     decision_time="2020-01-02T21:15:00+00:00",
                                     now="2020-01-02T21:15:00+00:00")
    assert action == policy.predict(expected)
    manifest = policy.manifest
    manifest["normalizer"]["means"][0] = 100
    assert policy.manifest["normalizer"]["means"][0] == 0
    with pytest.raises(FrozenInstanceError):
        policy.normalizer.means = ()
    with pytest.raises(FileExistsError):
        freeze(actor, data, path.parent, algorithm="trpo")


def test_bundle_contract_and_tampering_rejection(bundle):
    path, _, _ = bundle
    with pytest.raises(ValueError, match="schema/feed"):
        FrozenTanhPolicy(path, feature_schema="wrong", feed="fixture")
    with pytest.raises(ValueError, match="schema/feed"):
        FrozenTanhPolicy(path, feature_schema="daily-indicators-v1", feed="wrong")
    with pytest.raises(ValueError, match="identity"):
        load(path, expected_bundle_id="wrong")
    (path / "actor.json").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        load(path)


def test_feature_order_shape_and_availability_rejection(bundle):
    path, _, data = bundle
    p = load(path)
    with pytest.raises(ValueError, match="schema"):
        p.observation([0.] * 7, list(reversed(data["names"])), [0.] * 14)
    with pytest.raises(ValueError):
        p.observation([float("nan")] * 7, data["names"], [0.] * 14)
    with pytest.raises(ValueError):
        p.observation([0.] * 7, data["names"], [0.] * 13)
    kwargs = dict(symbol="SPY", available_time="2020-01-02T21:15:00+00:00",
                  decision_time="2020-01-02T21:15:00+00:00", now="2020-01-02T21:15:00+00:00")
    for change in ({"symbol": "UNKNOWN"}, {"max_age_seconds": -1},
                   {"now": "2020-01-05T21:15:00+00:00"},
                   {"decision_time": "2020-01-02T21:14:00+00:00"}):
        with pytest.raises(ValueError):
            p.predict_features([0.] * 7, data["names"], [0.] * 14, **(kwargs | change))


def test_bundle_extra_file_rejected(bundle):
    path, _, _ = bundle
    (path / "unexpected.json").write_text(json.dumps({}))
    with pytest.raises(ValueError, match="file set"):
        load(path)
