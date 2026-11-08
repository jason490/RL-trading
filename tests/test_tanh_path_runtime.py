from pathlib import Path
from types import MethodType

import pytest

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.runtime.colab import tanh_path_artifacts as artifacts
from rl_stocks.runtime.colab.tanh_paths import TanhPathJob, runtime_spec
from test_trpo_smoke_runtime import setup


def test_fixed_runtime_and_partial_local_panel_block_allocation(tmp_path, monkeypatch):
    spec = runtime_spec(Path(__file__).resolve().parents[1])
    assert spec['maximum_units'] == .25 and spec['reserve_seconds'] == 900
    assert spec['deadline_seconds'] == 7200 and not spec['training_performed']
    assert not TanhPathJob.training_performed
    job, dataset = setup(tmp_path, monkeypatch)
    job.preflight = MethodType(TanhPathJob.preflight, job)
    with pytest.raises(ValueError, match='complete local'):
        job.run_smoke(dataset)
    assert not job.calls and not job.allocated


def fixture(tmp_path, monkeypatch):
    local = tmp_path / 'local'
    local.mkdir()
    (local / 'result.json').write_text('{}')
    (local / 'provenance.json').write_text('{"fixture":"local"}')
    admission = {'local_result_sha256': checksum(local / 'result.json'),
                 'local_provenance_sha256': checksum(local / 'provenance.json'),
                 'numerical_sources': {'calculate.py': 'fixed-sha'}}
    source = {'revision': 'fixture', 'dirty': ''}
    session = {'session': 'owned-fixture', 'status': 'stopped', 'accelerator': 'CPU',
               'protocol': {'training': False, 'source': source, 'source_hash': 'fixture-source',
                            'admission': admission}}
    extracted = tmp_path / 'extracted/artifacts'
    (extracted / 'tanh-paths').mkdir(parents=True)
    (extracted / 'tanh-paths/result.json').write_text('{}')
    dataset = {'fixture': 'dataset'}
    completion = {'schema': 'tanh-path-inference-completion-v1',
                  'protocol_sha256': artifacts.PROTOCOL_SHA256, 'dataset': digest(dataset),
                  'source': source, 'source_hash': 'fixture-source', 'python': [3, 13, 18],
                  'execution': 'Colab CPU', 'durable_checkpoint_interval_seconds': 300,
                  'numerical_sources_sha256': digest(admission['numerical_sources']),
                  'training_performed': False, 'development_evaluated': False,
                  'retired_test_access': False, 'performance_promotion': False,
                  'files': {'tanh-paths/result.json': checksum(extracted / 'tanh-paths/result.json')}}
    (extracted / 'completion.json').write_bytes(canonical(completion))
    archive = tmp_path / 'transport.tar.gz'
    archive.write_text('fixture transport; extraction covered separately')
    calls = []
    def extract(*args, **kwargs):
        calls.append('extract')
        return extracted
    def compare(left, right, dataset):
        calls.append('compare')
        return {'exact_full_paths': True, 'runtime_locations_verified': False}
    monkeypatch.setattr(artifacts, 'extract_artifacts', extract)
    monkeypatch.setattr(artifacts, 'compare', compare)
    kwargs = dict(source=source, source_hash='fixture-source', dataset=dataset, local=local,
                  admission=admission, session_record=session,
                  limits={'maximum_archive_bytes': 512 * 1024 * 1024, 'maximum_archive_members': 256})
    return archive, kwargs, extracted, completion, calls


def test_remote_acceptance_requires_stopped_inference_and_provenance(tmp_path, monkeypatch):
    archive, kwargs, _, _, calls = fixture(tmp_path, monkeypatch)
    kwargs['session_record']['status'] = 'active'
    with pytest.raises(ValueError, match='stopped inference'):
        artifacts.accept_remote(archive, tmp_path / 'unused', **kwargs)
    assert not calls
    kwargs['session_record']['status'] = 'stopped'
    proof = artifacts.accept_remote(archive, tmp_path / 'unused', **kwargs)
    assert proof['runtime_locations_verified'] and proof['cleanup_verified']
    assert calls == ['extract', 'compare']


@pytest.mark.parametrize('mutation', ['python', 'retired', 'sources', 'manifest', 'local'])
def test_remote_identity_corruption_rejected_before_comparison(tmp_path, monkeypatch, mutation):
    archive, kwargs, extracted, complete, calls = fixture(tmp_path, monkeypatch)
    if mutation == 'python':
        complete['python'] = [3, 12, 0]
    elif mutation == 'retired':
        complete['retired_test_access'] = True
    elif mutation == 'sources':
        complete['numerical_sources_sha256'] = 'changed'
    elif mutation == 'manifest':
        complete['files'] = {}
    else:
        (kwargs['local'] / 'provenance.json').write_text('changed')
    (extracted / 'completion.json').write_bytes(canonical(complete))
    with pytest.raises(ValueError):
        artifacts.accept_remote(archive, tmp_path / 'unused', **kwargs)
    assert 'compare' not in calls
