from rl_stocks.data.universe.type_contract import verification_scope
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
                              'bars': [{'session': '2010-01-04',
                                        'available_time': '2010-01-04T21:00:00Z'}]}
                         for s in ('A', 'B')}}


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
                'historical_evidence_verified': True, 'verification_scope': verification_scope(),
                'records': [{'symbol': s, 'kind': kind, 'source': 'synthetic historical fixture',
                             'evidence_kind': 'historical_primary_source',
                             'known_from': '2009-01-01', 'source_fact_kind': kind,
                             'source_sha256': 'f' * 64, 'source_available_time': '2009-01-01T00:00:00Z'} for s, kind in
                            [('A', 'stock'), ('B', 'etf')]]}
    assert validate_type_manifest(data, evidence) == digest(evidence)
    evidence['records'][0]['evidence_kind'] = 'current_snapshot'
    with pytest.raises(ValueError, match='historical'):
        validate_type_manifest(data, evidence)
    with pytest.raises(ValueError, match='reviewed'):
        validate_type_manifest(data, audit_type_evidence(data))


def test_scope_and_availability_cannot_overclaim_or_backdate():
    from rl_stocks.data.universe.type_contract import validate_scope, validate_source_point
    evidence = {'verification_scope': verification_scope()}
    validate_scope(evidence)
    evidence['verification_scope']['historical_price_link_verified'] = True
    with pytest.raises(ValueError, match='scope'):
        validate_scope(evidence)
    record = {'kind': 'stock', 'source_fact_kind': 'stock',
              'evidence_kind': 'historical_primary_source', 'source': 'synthetic source',
              'source_sha256': 'a' * 64, 'known_from': '2010-01-04',
              'source_available_time': '2010-01-04T14:30:00Z'}
    validate_source_point(record)
    record['known_from'] = '2009-01-01'
    with pytest.raises(ValueError, match='availability'):
        validate_source_point(record)
    record['known_from'] = '2010-01-04'
    record['source_fact_kind'] = 'etf'
    with pytest.raises(ValueError, match='historical'):
        validate_source_point(record)


def test_source_after_first_decision_rejected_even_on_same_date():
    from rl_stocks.core.contracts import digest
    from rl_stocks.training.task_conditioning import validate_type_manifest
    data = payload()
    evidence = {'schema': 'historical-instrument-types-v1', 'dataset': digest(data),
                'historical_evidence_verified': True, 'verification_scope': verification_scope(),
                'records': [{'symbol': symbol, 'kind': kind, 'source_fact_kind': kind,
                             'source': 'synthetic source', 'source_sha256': 'f' * 64,
                             'evidence_kind': 'historical_primary_source',
                             'known_from': '2010-01-04',
                             'source_available_time': '2010-01-04T21:00:00Z'}
                            for symbol, kind in [('A', 'stock'), ('B', 'etf')]]}
    assert validate_type_manifest(data, evidence) == digest(evidence)
    evidence['records'][0]['source_available_time'] = '2010-01-04T21:00:01Z'
    with pytest.raises(ValueError, match='first training decision'):
        validate_type_manifest(data, evidence)
