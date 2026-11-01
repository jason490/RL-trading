from rl_stocks.data.universe.type_contract import verification_scope
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

from test_policy_architecture import environment
from rl_stocks.core.contracts import digest
from rl_stocks.environment.observations.task import InstrumentTypeObservation
from rl_stocks.inference.task_bundle import FrozenTaskPolicy, freeze_task
from rl_stocks.rl.sac.task_factory import create_task_sac
from rl_stocks.risk.exposure import execution_contract, filter_exposure


@pytest.fixture(autouse=True)
def no_local_optimization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('local optimization attempted')
    for name in ('learn', 'train'):
        monkeypatch.setattr(SAC, name, forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)


def bundle(root, role='tagged_routed'):
    env = environment()
    symbol = env.episodes[0]['symbol']
    data = {'schema': 'task-fixture', 'names': env.names, 'normalizer': asdict(env.normalizer),
            'episodes': {'train': deepcopy(env.episodes[0])},
            'manifest': {'included_splits': ['train', 'validation'], 'unverified_override': False,
                         'universe': [symbol], 'universe_config': {'instruments': [
                             {'symbol': symbol, 'kind': 'broad_etf'}]},
                         'provenance': {'capabilities': {'feed': 'fixture'}}}}
    evidence = {'schema': 'historical-instrument-types-v1', 'dataset': digest(data),
                'historical_evidence_verified': True, 'verification_scope': verification_scope(), 'records': [
                    {'symbol': symbol, 'kind': 'etf', 'known_from': '2010-01-01',
                     'source': 'synthetic historical fixture, not production evidence',
                     'evidence_kind': 'historical_primary_source', 'source_fact_kind': 'etf',
                     'source_sha256': 'f' * 64, 'source_available_time': '2010-01-01T00:00:00Z'}]}
    wrapped = InstrumentTypeObservation(env, evidence['records'], task_role=role)
    model = create_task_sac(wrapped, task_role=role, smoke=True)
    source = root / 'model.zip'
    model.save(source)
    path = freeze_task(source, data, task_role=role, type_manifest=evidence,
                       metadata={'algorithm': 'sac', 'observation': env.observation_spec,
                                 'action_execution': execution_contract('unrestricted', 'unit_magnitude')},
                       output=root / 'frozen')
    return model, path, symbol


@pytest.mark.parametrize('role', ['masked_shared', 'tagged_shared', 'tagged_routed'])
def test_actual_frozen_parity_and_immutable_normalizer(tmp_path, role):
    model, path, symbol = bundle(tmp_path, role)
    frozen = FrozenTaskPolicy(path, feature_schema='task-fixture', feed='fixture')
    before = deepcopy(frozen._model.policy.state_dict())
    normalizer = deepcopy(frozen.normalizer)
    x = np.linspace(-1, 1, 21, dtype=np.float32)
    original = x.copy()
    expected = model.predict(np.append(x, 0 if role == 'masked_shared' else 1),
                             deterministic=True)[0]
    for _ in range(3):
        np.testing.assert_array_equal(frozen.predict(x, symbol=symbol, session='2016-01-04'),
                                      filter_exposure(expected, 'unit_magnitude'))
    assert frozen.normalizer == normalizer
    assert np.array_equal(x, original)
    assert all(torch.equal(value, before[key]) for key, value in
               frozen._model.policy.state_dict().items())
    view = frozen.manifest
    view['metadata']['task_contract']['role'] = 'unknown'
    assert frozen.manifest['metadata']['task_contract']['role'] == role
    for bad in (np.zeros(22), np.full(21, np.nan)):
        with pytest.raises(ValueError, match='full21'):
            frozen.predict(bad, symbol=symbol, session='2016-01-04')
    with pytest.raises(ValueError, match='unknown'):
        frozen.predict(x, symbol='UNKNOWN', session='2016-01-04')
    with pytest.raises(ValueError, match='unavailable'):
        frozen.predict(x, symbol=symbol, session='2009-01-01')


def test_corrupt_bundle_and_wrong_feed_reject(tmp_path):
    _, path, _ = bundle(tmp_path)
    with pytest.raises(ValueError, match='schema/feed'):
        FrozenTaskPolicy(path, feature_schema='task-fixture', feed='wrong')
    manifest = path / 'manifest.json'
    value = json.loads(manifest.read_text())
    value['metadata']['task_contract']['role'] = 'masked_shared'
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='checksum'):
        FrozenTaskPolicy(path, feature_schema='task-fixture', feed='fixture')


def test_rehashed_metadata_cannot_relabel_policy_role(tmp_path):
    from rl_stocks.core.contracts import canonical
    from rl_stocks.inference.bundle import checksum
    _, path, _ = bundle(tmp_path)
    manifest_path = path / 'manifest.json'
    value = json.loads(manifest_path.read_text())
    value['metadata']['task_contract']['role'] = 'masked_shared'
    manifest_path.write_bytes(canonical(value))
    hashes = {p.name: checksum(p) for p in path.iterdir() if p.name != 'checksums.json'}
    (path / 'checksums.json').write_bytes(canonical(hashes))
    renamed = path.with_name(digest(value))
    path.rename(renamed)
    with pytest.raises(ValueError, match='loaded policy'):
        FrozenTaskPolicy(renamed, feature_schema='task-fixture', feed='fixture')


def test_generic_frozen_loader_rejects_task_policy_even_with_matching_vector_size(tmp_path):
    from rl_stocks.inference.bundle import FrozenPolicy, freeze
    from rl_stocks.environment.observations.state import observation_spec
    _, path, _ = bundle(tmp_path)
    names = [f'f{i}' for i in range(8)]
    generic = {'schema': 'generic-fixture', 'names': names,
               'normalizer': {'names': names, 'means': [0.] * 8, 'scales': [1.] * 8},
               'manifest': {'provenance': {'capabilities': {'feed': 'fixture'}}}}
    disguised = freeze(path / 'model.zip', generic,
                        {'algorithm': 'sac', 'observation': observation_spec('execution_full')},
                        tmp_path / 'generic')
    with pytest.raises(ValueError, match='symbol/date-aware'):
        FrozenPolicy(disguised, 'generic-fixture', 'fixture')


def test_rehashed_bundle_cannot_backdate_source_point(tmp_path):
    from rl_stocks.core.contracts import canonical
    from rl_stocks.inference.bundle import checksum
    _, path, _ = bundle(tmp_path)
    manifest_path = path / 'manifest.json'
    value = json.loads(manifest_path.read_text())
    contract = value['metadata']['task_contract']
    contract['type_manifest']['records'][0]['source_available_time'] = '2010-01-02T00:00:00Z'
    contract['type_manifest_sha256'] = digest(contract['type_manifest'])
    manifest_path.write_bytes(canonical(value))
    (path / 'checksums.json').write_bytes(canonical({
        p.name: checksum(p) for p in path.iterdir() if p.name != 'checksums.json'}))
    renamed = path.with_name(digest(value))
    path.rename(renamed)
    with pytest.raises(ValueError, match='availability'):
        FrozenTaskPolicy(renamed, feature_schema='task-fixture', feed='fixture')
