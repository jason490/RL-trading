from copy import deepcopy

import pytest

from rl_stocks.training.task_conditioning import audit_type_evidence


def payload():
    return {'manifest': {'included_splits': ['train', 'validation'],
                         'unverified_override': False,
                         'universe_config': {'instruments': [
                             {'symbol': 'A', 'kind': 'stock'},
                             {'symbol': 'B', 'kind': 'broad_etf'}]}},
            'normalizer': {'names': ['x']},
            'episodes': {s: {'symbol': s, 'split': 'train',
                              'bars': [{'session': '2010-01-04'}]} for s in ('A', 'B')}}


def test_audit_does_not_invent_historical_availability():
    data = payload()
    before = deepcopy(data)
    result = audit_type_evidence(data)
    assert data == before
    assert result['stocks'] == result['etfs'] == 1
    assert not result['historical_metadata_admitted']
    assert all('known_from' not in r for r in result['records'])


def test_audit_rejects_test_or_unknown_kind():
    data = payload()
    data['episodes']['A']['split'] = 'test'
    with pytest.raises(ValueError, match='retired'):
        audit_type_evidence(data)
    data = payload()
    data['manifest']['universe_config']['instruments'][0]['kind'] = 'unknown'
    with pytest.raises(ValueError, match='explicit'):
        audit_type_evidence(data)


def test_protocol_is_fixed_and_manifest_admission_rejects_snapshot():
    from rl_stocks.core.contracts import digest
    from rl_stocks.training.task_conditioning import protocol, validate_type_manifest
    config = protocol()
    assert config['steps_per_cell'] == 100000
    assert config['selection']['development_selection'] is False
    data = payload()
    evidence = {'schema': 'historical-instrument-types-v1', 'dataset': digest(data),
                'historical_evidence_verified': True,
                'records': [{'symbol': s, 'kind': kind, 'source': 'synthetic historical fixture',
                             'evidence_kind': 'historical_primary_source',
                             'known_from': '2009-01-01'} for s, kind in
                            [('A', 'stock'), ('B', 'etf')]]}
    assert validate_type_manifest(data, evidence) == digest(evidence)
    evidence['records'][0]['evidence_kind'] = 'current_snapshot'
    with pytest.raises(ValueError, match='historical'):
        validate_type_manifest(data, evidence)
    with pytest.raises(ValueError, match='reviewed'):
        validate_type_manifest(data, audit_type_evidence(data))
