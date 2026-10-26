from copy import deepcopy
import hashlib

import numpy as np
import pytest

from rl_stocks.training.crossq_archives import snapshot_sources, verify_arrays


def arrays():
    source, target, mapping = {}, {}, []
    for branch, count in [('actor', 4), ('critic', 6)]:
        for i in range(count):
            left, right = f's{i}', f't{i}'
            mapping.append(dict(branch=branch, source=left, target=right, weight_shape=[2, 2]))
            for suffix, value in [('weight', np.full((2, 2), i, dtype=np.float32)),
                                  ('bias', np.full(2, i, dtype=np.float32))]:
                source[f'{branch}.{left}.{suffix}'] = value
                target[f'{branch}.{right}.{suffix}'] = value.copy()
    for i in range(9):
        for field in ['scale', 'bias', 'ra_mean', 'ra_var', 'steps']:
            value = 1 if field in ['scale', 'ra_var'] else 0
            target[f'normalization.n{i}.{field}'] = np.array(value, dtype=np.int64) if field == 'steps' else np.full(2, value, dtype=np.float32)
    return source, target, dict(mapping=mapping)


def test_independent_archive_mapping_and_complete_native_states():
    source, target, mapping = arrays()
    result = verify_arrays(source, target, mapping)
    assert result['common_archived_linear_tensors_exact'] == 20
    assert result['native_normalization_counters'] == 9


@pytest.mark.parametrize('change', ['copy', 'dtype', 'missing_norm', 'counter', 'duplicate'])
def test_archive_corruption_or_incomplete_normalization_rejects(change):
    source, target, mapping = arrays()
    if change == 'copy':
        target['actor.t0.weight'][0, 0] = 99
    elif change == 'dtype':
        target['actor.t0.weight'] = target['actor.t0.weight'].astype(np.float64)
    elif change == 'missing_norm':
        target.pop('normalization.n0.ra_var')
    elif change == 'counter':
        target['normalization.n0.steps'] = np.array(1)
    else:
        mapping['mapping'][1] = deepcopy(mapping['mapping'][0])
    with pytest.raises(ValueError):
        verify_arrays(source, target, mapping)


def test_source_hash_validation_precedes_all_snapshot_writes(tmp_path):
    (tmp_path / 'a.py').write_bytes(b'original a')
    (tmp_path / 'b.py').write_bytes(b'original b')
    sources = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in ['a.py', 'b.py']}
    output = tmp_path / 'snapshot'
    with pytest.raises(ValueError, match='source changed'):
        snapshot_sources(dict(sources, **{'b.py': '0' * 64}), output, tmp_path)
    assert not output.exists()
    records = snapshot_sources(sources, output, tmp_path)
    assert (output / 'a.py.txt').read_bytes() == b'original a'
    assert records['a.py']['sha256'] == sources['a.py']
    with pytest.raises(ValueError, match='new contained'):
        snapshot_sources(sources, output, tmp_path)
    with pytest.raises(ValueError, match='stay in repository'):
        snapshot_sources(sources, tmp_path.parent / 'outside', tmp_path)
