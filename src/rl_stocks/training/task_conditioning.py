"""Causal metadata admission for the task-conditioning experiment."""
from rl_stocks.core.contracts import digest


def audit_type_evidence(dataset):
    manifest = dataset['manifest']
    if (manifest['included_splits'] != ['train', 'validation']
            or manifest['unverified_override'] is not False):
        raise ValueError('verified train/development-only payload required')
    instruments = manifest['universe_config']['instruments']
    annotations = {}
    for item in instruments:
        if item['symbol'] in annotations:
            raise ValueError('duplicate universe annotation')
        annotations[item['symbol']] = item
    training = {}
    for episode in dataset['episodes'].values():
        if episode['split'] not in ('train', 'validation'):
            raise ValueError('retired test access rejected')
        if episode['split'] != 'train':
            continue
        symbol = episode['symbol']
        if symbol in training:
            raise ValueError('one training episode per symbol required')
        dates = [bar['session'] for bar in episode['bars']]
        if not dates or dates != sorted(set(dates)) or dates[-1] > '2020-12-31':
            raise ValueError('chronological training dates required')
        kind = annotations.get(symbol, {}).get('kind')
        if kind not in ('stock', 'broad_etf'):
            raise ValueError('explicit supported instrument kind required')
        training[symbol] = {'symbol': symbol, 'kind': 'etf' if kind == 'broad_etf' else 'stock',
                            'first_training_session': dates[0],
                            'historical_evidence_verified': False}
    return {'schema': 'task-type-evidence-audit-v1',
            'dataset_identity': digest(dataset),
            'normalizer_sha256': digest(dataset['normalizer']),
            'universe_sha256': digest(manifest['universe_config']),
            'records': [training[s] for s in sorted(training)],
            'stocks': sum(r['kind'] == 'stock' for r in training.values()),
            'etfs': sum(r['kind'] == 'etf' for r in training.values()),
            'historical_metadata_admitted': False,
            'reason': 'Snapshot universe kind lacks per-symbol historical availability evidence; no known_from date inferred',
            'training_performed': False, 'development_evaluated': False,
            'test_evaluated': False, 'colab_allocated': False}


PROTOCOL_SHA256 = '4feb50d28de5a07d851573ba2bf4649cf2941d65d41db2ed3afc06ca5be163d8'


def protocol():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[3] / 'configs/experiments/task_conditioning.json'
    value = json.loads(path.read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError('task-conditioning protocol changed')
    return value


def validate_type_manifest(dataset, evidence):
    """Bind explicit reviewed historical evidence before constructing an environment."""
    from datetime import date
    audit = audit_type_evidence(dataset)
    if (evidence.get('schema') != 'historical-instrument-types-v1'
            or evidence.get('dataset') != audit['dataset_identity']
            or evidence.get('historical_evidence_verified') is not True):
        raise ValueError('reviewed historical type manifest required')
    from rl_stocks.data.universe.type_contract import validate_scope, validate_source_point
    validate_scope(evidence)
    from rl_stocks.data.validation.bars import timestamp
    first_decisions = {episode['symbol']: timestamp(episode['bars'][0]['available_time'])
                       for episode in dataset['episodes'].values()
                       if episode['split'] == 'train'}
    records = evidence['records']
    expected = {r['symbol']: r for r in audit['records']}
    if len(records) != len(expected) or {r['symbol'] for r in records} != set(expected):
        raise ValueError('exact training universe classification required')
    for record in records:
        validate_source_point(record)
        if timestamp(record['source_available_time']) > first_decisions[record['symbol']]:
            raise ValueError('type source availability follows first training decision')
        declared = expected[record['symbol']]
        if (record['kind'] != declared['kind'] or not record.get('source')
                or record.get('evidence_kind') != 'historical_primary_source'
                or date.fromisoformat(record['known_from']) >
                date.fromisoformat(declared['first_training_session'])):
            raise ValueError('causal historical instrument evidence required')
    return digest(evidence)
