from dataclasses import FrozenInstanceError
import json

import pytest

from test_canonical_sac_policy import envelope
from rl_stocks.core.contracts import canonical, digest
from rl_stocks.inference.canonical_sac_policy import CanonicalSACPolicy
from rl_stocks.inference.standard_sac_bundle import freeze, FrozenStandardSACPolicy
from rl_stocks.risk.exposure import filter_exposure
from rl_stocks.training import trpo_smoke


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    names = [f'f{i}' for i in range(7)]
    data = {'schema': 'daily-indicators-v1', 'names': names,
            'normalizer': {'names': names, 'means': [0.] * 7, 'scales': [2.] * 7},
            'manifest': {'universe': ['SPY'], 'included_splits': ['train', 'validation'],
                         'provenance': {'capabilities': {'feed': 'fixture'}}}}
    monkeypatch.setattr(trpo_smoke, 'validate_dataset', lambda value: value)
    monkeypatch.setattr(trpo_smoke, 'protocol', lambda: {
        'dataset': digest(data), 'normalizer_sha256': digest(data['normalizer'])})
    actor = tmp_path / 'actor.json'
    actor.write_bytes(canonical(envelope()))
    checkpoint = {'phase': 'selected', 'transitions': 10000,
                  'model_sha256': 'a' * 64, 'acceptance_sha256': 'b' * 64}
    path = freeze(actor, data, tmp_path / 'bundles', checkpoint=checkpoint,
                  evaluation_execution='unit_magnitude')
    return path, actor, data, checkpoint


def load(path, mode='unit_magnitude'):
    return FrozenStandardSACPolicy(path, feature_schema='daily-indicators-v1', feed='fixture',
                                   expected_execution=mode, expected_bundle_id=path.name)


def test_shared_preprocessing_and_explicit_unit_control(bundle):
    path, actor, data, _ = bundle
    frozen = load(path)
    features, state = [2.] + [0.] * 6, [0.] * 14
    observation = frozen.observation(features, data['names'], state)
    assert observation == tuple([1.] + [0.] * 20)
    raw = CanonicalSACPolicy.load(actor).predict(observation)
    assert frozen.raw_action(observation) == raw
    expected = tuple(filter_exposure(raw, 'unit_magnitude'))
    assert frozen.predict(observation) == expected
    assert expected[0] == 1 and expected[1:] == raw[1:]
    assert frozen.predict_features(features, data['names'], state, symbol='SPY',
                                   available_time='2020-01-02T21:15:00+00:00',
                                   decision_time='2020-01-02T21:15:00+00:00',
                                   now='2020-01-02T21:15:00+00:00') == expected
    with pytest.raises(ValueError, match='execution mode'):
        load(path, 'unrestricted')


def test_unrestricted_mode_is_distinct_and_create_only(bundle, tmp_path):
    path, actor, data, checkpoint = bundle
    other = freeze(actor, data, tmp_path / 'raw', checkpoint=checkpoint,
                   evaluation_execution='unrestricted')
    assert other.name != path.name
    frozen = load(other, 'unrestricted')
    observation = [1.] + [0.] * 20
    assert frozen.predict(observation) == CanonicalSACPolicy.load(actor).predict(observation)
    with pytest.raises(FileExistsError):
        freeze(actor, data, tmp_path / 'raw', checkpoint=checkpoint,
               evaluation_execution='unrestricted')
    with pytest.raises(ValueError, match='source'):
        freeze(actor, data, tmp_path / 'wrong', checkpoint=dict(checkpoint, model_sha256='c' * 64),
               evaluation_execution='unrestricted')


def test_immutable_normalizer_and_input_rejection(bundle):
    path, _, data, _ = bundle
    frozen = load(path)
    view = frozen.manifest
    view['normalizer']['means'][0] = 999
    assert frozen.manifest['normalizer']['means'][0] == 0
    with pytest.raises(FrozenInstanceError):
        frozen.normalizer.means = ()
    with pytest.raises(ValueError, match='schema'):
        frozen.observation([0.] * 7, list(reversed(data['names'])), [0.] * 14)
    with pytest.raises(ValueError, match='stale'):
        frozen.predict_features([0.] * 7, data['names'], [0.] * 14, symbol='SPY',
                                available_time='2020-01-02T21:15:00+00:00',
                                decision_time='2020-01-02T21:15:00+00:00',
                                now='2020-01-05T21:15:00+00:00')
    (path / 'requirements.lock').write_text('tampered')
    with pytest.raises(ValueError, match='checksum'):
        load(path)


def test_checkpoint_metadata_and_feed_are_not_silently_changed(bundle, tmp_path):
    path, actor, data, checkpoint = bundle
    with pytest.raises(ValueError, match='checkpoint identity'):
        freeze(actor, data, tmp_path / 'zero', checkpoint=dict(checkpoint, transitions=0),
               evaluation_execution='unit_magnitude')
    with pytest.raises(ValueError, match='feed'):
        FrozenStandardSACPolicy(path, feature_schema='daily-indicators-v1', feed='other',
                                expected_execution='unit_magnitude')
    assert json.loads((path / 'manifest.json').read_text())['checkpoint'] == checkpoint
