"""Public retention contracts, using manual serialization records without fitting."""
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

from test_crossq_artifacts import fixture, no_optimization  # noqa: F401
from rl_stocks.reporting.crossq_public import public_evidence


def retained(tmp_path, monkeypatch, arm):
    metadata, _, files = fixture(monkeypatch, arm)
    metadata.update(status='complete', seed=11, algorithm=arm, smoke=True, actual_steps=4096, bundle_id='fixture')
    root = tmp_path / 'retained'
    run = root / 'runs/fixture'
    run.mkdir(parents=True)
    for name, content in files.items():
        if name.startswith('run/'):
            (run / name[4:]).write_bytes(content)
    (run / 'metadata.json').write_text(json.dumps(metadata))
    return root, run, metadata


@pytest.mark.parametrize('arm', ['sac', 'crossq'])
def test_required_initializer_and_nonraw_replay_normalization_evidence(tmp_path, monkeypatch, arm):
    _, run, metadata = retained(tmp_path, monkeypatch, arm)
    paths = public_evidence(run, metadata)
    assert {p.name for p in paths} == ({'crossq-initialization.npz', 'crossq-replay-state.json'} |
                                      ({'crossq-normalization-audit.json'} if arm == 'crossq' else set()))
    assert all(p.is_file() for p in paths)


@pytest.mark.parametrize('change', ['initializer', 'missing_audit', 'audit_mutation', 'replay', 'selected_norm'])
def test_invalid_public_evidence_rejected(tmp_path, monkeypatch, change):
    _, run, metadata = retained(tmp_path, monkeypatch, 'crossq')
    if change == 'initializer':
        (run / 'crossq-initialization.npz').write_bytes(b'changed')
    elif change == 'missing_audit':
        (run / 'crossq-normalization-audit.json').unlink()
    elif change == 'audit_mutation':
        (run / 'crossq-normalization-audit.json').write_text('{}')
    elif change == 'replay':
        (run / 'crossq-replay-state.json').write_text('{}')
    else:
        metadata['crossq_normalization'] = {}
    with pytest.raises((ValueError, FileNotFoundError)):
        public_evidence(run, metadata)


def test_packager_preserves_exact_bytes_and_arm_labels_without_raw_replay(tmp_path, monkeypatch):
    root, run, metadata = retained(tmp_path, monkeypatch, 'crossq')
    bundle = root / 'frozen/fixture'
    bundle.mkdir(parents=True)
    (bundle / 'model.zip').write_bytes(b'public serialization fixture')
    (bundle / 'manifest.json').write_text('{}')
    (bundle / 'checksums.json').write_text('{}')
    reports = {'fixture': dict(rows=[dict(session='2021-01-04', equity=100000)], metrics={},
                baselines={'buy_hold': {'rows': [dict(session='2021-01-04', equity=100000)]}})}
    (run / 'selected-validation.json').write_text(json.dumps(reports))
    (run / 'replay.pkl').write_bytes(b'private filled replay must stay local')
    (run / '.env').write_text('PRIVATE_FIXTURE=excluded')
    spec = importlib.util.spec_from_file_location('package_research', Path(__file__).parents[1] / 'scripts/package_research.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / 'public'
    packed = module.package([root], output)
    with tarfile.open(packed) as archive:
        names = archive.getnames()
        for path in public_evidence(run, metadata):
            assert archive.extractfile('runs/fixture/' + path.name).read() == path.read_bytes()
        assert not any(n.endswith('replay.pkl') or '.env' in n for n in names)
    assert 'crossq / crossq / dataset' in (output / 'validation.html').read_text()
