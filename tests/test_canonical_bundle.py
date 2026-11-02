from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

from test_task_bundle import bundle
from test_policy_architecture import environment
from rl_stocks.core.contracts import canonical, digest
from rl_stocks.evaluation.canonical_task import CanonicalEpisodePredictor, evaluate_canonical_training
from rl_stocks.evaluation.runner import _evaluate
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.inference.canonical_actor import CanonicalActor
from rl_stocks.inference.canonical_bundle import FrozenCanonicalTaskPolicy, freeze_canonical_task


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('local optimization attempted')
    for name in ('learn', 'train'):
        monkeypatch.setattr(SAC, name, forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)


def frozen_fixture(tmp_path, role='tagged_routed'):
    _, native, symbol = bundle(tmp_path, role)
    source_before = {p.name: p.read_bytes() for p in native.iterdir()}
    path = freeze_canonical_task(native, tmp_path / 'reference')
    assert {p.name: p.read_bytes() for p in native.iterdir()} == source_before
    return FrozenCanonicalTaskPolicy(path, feature_schema='task-fixture', feed='fixture'), symbol


@pytest.mark.parametrize('role', ['masked_shared', 'tagged_shared', 'tagged_routed'])
def test_frozen_raw_direct_and_filtered_parity(tmp_path, role):
    frozen, symbol = frozen_fixture(tmp_path, role)
    before = deepcopy(frozen.manifest)
    normalizer = asdict(frozen.normalizer)
    vector = np.linspace(-.5, .5, 21, dtype=np.float32)
    actor = CanonicalActor.load(frozen.path / 'actor.json')
    raw = actor.predict(tuple(vector) + (0. if role == 'masked_shared' else 1.,))
    assert frozen.raw_action(vector, symbol=symbol, session='2016-01-01') == raw
    action = frozen.predict(vector, symbol=symbol, session='2016-01-01')
    assert action[0] == np.sign(raw[0])
    assert tuple(action[1:]) == raw[1:]
    view = frozen.manifest
    view.clear()
    frozen.execution.clear()
    assert frozen.manifest == before
    assert asdict(frozen.normalizer) == normalizer
    with pytest.raises(ValueError, match='unknown'):
        frozen.predict(vector, symbol='UNKNOWN', session='2016-01-01')
    with pytest.raises(ValueError, match='unavailable'):
        frozen.predict(vector, symbol=symbol, session='2009-01-01')
    for bad in ([0.] * 22, [float('nan')] * 21):
        with pytest.raises(ValueError):
            frozen.predict(bad, symbol=symbol, session='2016-01-01')
    with pytest.raises(ValueError, match='schema/feed'):
        FrozenCanonicalTaskPolicy(frozen.path, feature_schema='wrong', feed='fixture')
    with pytest.raises(ValueError, match='identity'):
        FrozenCanonicalTaskPolicy(frozen.path, feature_schema='task-fixture', feed='fixture',
                                  expected_bundle_id='a' * 64)


def test_episode_replay_matches_independent_direct_actor(tmp_path):
    frozen, symbol = frozen_fixture(tmp_path)
    env = environment()
    episode = deepcopy(env.episodes[0])
    # Short engineered fixture, not a performance experiment or a selected window.
    for key in ('bars', 'features', 'atr'):
        episode[key] = episode[key][:9]
    data = {'normalizer': asdict(env.normalizer), 'names': env.names,
            'episodes': {'fixture': episode}}
    before = digest(data)
    adapter = CanonicalEpisodePredictor(frozen, episode)
    actual = _evaluate(adapter, data, 'train', SimulationConfig(), 'execution_full', 'unit_magnitude')
    assert adapter.cursor == 8
    actor = CanonicalActor.load(frozen.path / 'actor.json')

    class Direct:
        def predict(self, observation, deterministic=True):
            return actor.predict(tuple(observation) + (1.,)), None

    expected = _evaluate(Direct(), data, 'train', SimulationConfig(), 'execution_full', 'unit_magnitude')
    assert digest(actual) == digest(expected)
    assert digest(data) == before
    with pytest.raises(ValueError, match='exhausted'):
        adapter.predict([0.] * 21)
    for split in ('validation', 'test'):
        with pytest.raises(ValueError, match='training-only'):
            CanonicalEpisodePredictor(frozen, {**episode, 'split': split})
    future = deepcopy(episode)
    future['bars'][-1]['session'] = '2024-01-01'
    with pytest.raises(ValueError, match='training-only'):
        CanonicalEpisodePredictor(frozen, future)
    predictor = CanonicalEpisodePredictor(frozen, episode)
    with pytest.raises(ValueError, match='deterministic'):
        predictor.predict([0.] * 21, deterministic=False)
    assert predictor.cursor == 0


def test_corruption_and_file_set(tmp_path):
    frozen, _ = frozen_fixture(tmp_path)
    extra = frozen.path / 'extra.json'
    extra.write_text('{}')
    with pytest.raises(ValueError, match='file set'):
        FrozenCanonicalTaskPolicy(frozen.path, feature_schema='task-fixture', feed='fixture')
    extra.unlink()
    actor = frozen.path / 'actor.json'
    actor.write_text(actor.read_text() + ' ')
    with pytest.raises(ValueError, match='checksum'):
        FrozenCanonicalTaskPolicy(frozen.path, feature_schema='task-fixture', feed='fixture')


def test_reference_source_and_normalizer_binding(tmp_path):
    frozen, _ = frozen_fixture(tmp_path)
    source = frozen.manifest['source_manifest']
    # Construct the fixture payload exactly as the source metadata did.
    env = environment()
    data = {'schema': 'task-fixture', 'names': env.names, 'normalizer': asdict(env.normalizer),
            'episodes': {'train': deepcopy(env.episodes[0])}, 'manifest': source['data_manifest']}
    evidence = source['metadata']['task_contract']['type_manifest']
    assert digest(data) == source['metadata']['task_contract']['dataset_sha256']
    data['normalizer']['means'] = [10.] * 7
    with pytest.raises(ValueError):
        evaluate_canonical_training(frozen, data, evidence)
    # A checksum-valid altered source manifest must still bind to its original identity.
    manifest = frozen.manifest
    manifest['source_manifest']['normalizer']['means'] = [10.] * 7
    target = tmp_path / digest(manifest)
    target.mkdir()
    (target / 'actor.json').write_bytes((frozen.path / 'actor.json').read_bytes())
    (target / 'manifest.json').write_bytes(canonical(manifest))
    import hashlib
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir()}
    (target / 'checksums.json').write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match='source identity'):
        FrozenCanonicalTaskPolicy(target, feature_schema='task-fixture', feed='fixture')
