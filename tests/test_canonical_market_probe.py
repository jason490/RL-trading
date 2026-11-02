from dataclasses import asdict
from copy import deepcopy
import gzip
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from test_policy_architecture import environment
from test_canonical_bundle import frozen_fixture
from rl_stocks.core.contracts import canonical, digest
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.canonical_task import CanonicalEpisodePredictor
from rl_stocks.evaluation.canonical_window import replay_window
from rl_stocks.evaluation.runner import _evaluate
from rl_stocks.runtime.colab.canonical_market_probe import CanonicalMarketJob, PROTOCOL_SHA256, protocol


def test_minimal_engine_matches_existing_complete_report_contract(tmp_path):
    from stable_baselines3 import SAC
    import torch
    # Model construction/export permitted; optimization and native prediction aren't.
    from unittest.mock import patch
    def forbidden(*a, **kw):
        pytest.fail('native model execution attempted')
    with patch.object(SAC, 'learn', forbidden), patch.object(SAC, 'train', forbidden), \
            patch.object(torch.optim.Adam, 'step', forbidden), patch.object(torch.Tensor, 'backward', forbidden):
        frozen, _ = frozen_fixture(tmp_path)
    env = environment()
    episode = deepcopy(env.episodes[0])
    for key in ('bars', 'features', 'atr'):
        episode[key] = episode[key][:9]
    view = {'normalizer': asdict(env.normalizer), 'names': env.names,
            'episodes': {'fixture': episode}}
    actual = replay_window(frozen, view)
    reference = _evaluate(CanonicalEpisodePredictor(frozen, episode), view, 'train',
                          SimulationConfig(), 'execution_full', 'unit_magnitude')
    assert digest(actual) == digest(reference)


def test_minimal_engine_imports_without_torch_or_sb3():
    code = '''
import sys
from rl_stocks.evaluation.canonical_window import replay_window
assert 'torch' not in sys.modules and 'stable_baselines3' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30)


def test_inference_lock_is_exact_existing_project_subset():
    root = Path(__file__).resolve().parents[1]
    original = (root / 'requirements.lock').read_text()
    subset = (root / 'configs/reference-inference.lock').read_text()
    names = []
    for line in subset.splitlines():
        if '==' in line and not line.startswith('#'):
            names.append(line.split('==')[0])
        if not line.startswith('#'):
            assert line in original
    assert len(names) == 5
    assert hashlib.sha256(subset.encode()).hexdigest() == protocol(root)['inference_lock_sha256']


def valid_record(spec, raw):
    return {'schema': spec['schema'], 'protocol_sha256': PROTOCOL_SHA256,
        'python_minor': [3, 13], 'versions': spec['dependencies'],
        'reference_bundle_id': spec['reference_bundle_id'],
        'window_episodes_sha256': spec['window_episodes_sha256'],
        'episodes': 26, 'predictions': 19136, 'results_sha256': hashlib.sha256(raw).hexdigest(),
        'forbidden_imports': [], 'elapsed_seconds': 1., 'protected_inputs_unchanged': True,
        **{name: False for name in ('training_performed', 'development_evaluated',
            'retired_test_access', 'performance_promotion', 'native_acceptance_changed')}}


def test_full_paths_verifier_rejects_corruption_and_different_outcome(tmp_path, monkeypatch):
    from rl_stocks.runtime.colab import canonical_market_probe
    root = Path(__file__).resolve().parents[1]
    spec = protocol(root)
    raw = canonical({'fixture': {'rows': [{'equity': 100000.}]}})
    record = valid_record(spec, raw)
    spec = {**spec, 'reference_results_sha256': record['results_sha256']}
    monkeypatch.setattr(canonical_market_probe, 'protocol', lambda r: spec)
    job = CanonicalMarketJob(tmp_path, sys.executable, 'CPU')
    job.extra_path = tmp_path / 'paths.gz'
    job.extra_path.write_bytes(gzip.compress(raw, mtime=0))
    record['paths_sha256'] = hashlib.sha256(job.extra_path.read_bytes()).hexdigest()
    assert job.verify_result(record, {})
    for key, value in [('predictions', 19135), ('forbidden_imports', ['torch']),
                       ('results_sha256', 'a'*64), ('retired_test_access', True)]:
        changed = {**record, key: value}
        with pytest.raises(ValueError):
            job.verify_result(changed, {})
    job.extra_path.write_bytes(gzip.compress(b'changed', mtime=0))
    with pytest.raises(ValueError, match='checksum'):
        job.verify_result(record, {})


def test_worker_setup_checks_default_python_before_install_and_safe_archive(tmp_path):
    root = Path(__file__).resolve().parents[1]
    job = CanonicalMarketJob(root, sys.executable, 'CPU')
    script = job.build_worker({'archive_sha256': 'a'*64}, '/content/result.json')
    compile(script, '<worker>', 'exec')
    assert script.index('sys.version_info') < script.index('create_environment(')
    assert 'python=create_environment(sys.executable' in script
    assert 'diagnostic receipt exported' in script
    assert '".." not in path.parts' in script


def test_setup_failure_is_retrievable_without_attempting_missing_paths(tmp_path):
    import json
    import zipfile
    root = Path(__file__).resolve().parents[1]
    job = CanonicalMarketJob(root, sys.executable, 'CPU')
    archive = tmp_path / 'fixture.zip'
    source = '''
class Failure(RuntimeError):
    record={'stage':'kernel-pip-preflight','returncode':73,'stderr':'fixture child failure'}
def create_environment(*a,**kw):
    raise Failure('fixture')
def checked(*a,**kw):
    raise AssertionError('unreachable replay')
'''
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('configs/reference-inference.lock',
                        (root / 'configs/reference-inference.lock').read_bytes())
        for package in ('rl_stocks', 'rl_stocks/runtime', 'rl_stocks/runtime/colab'):
            stream.writestr('src/' + package + '/__init__.py', '')
        stream.writestr('src/rl_stocks/runtime/colab/inference_setup.py', source)
    output = tmp_path / 'reference-probe.json'
    script = job.build_worker({'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()},
                              str(output))
    script = script.replace('/content/' + job.session + '-market.zip', str(archive))
    script = script.replace('/content/' + job.session + '-market', str(tmp_path / 'work'))
    worker = tmp_path / 'worker.py'
    worker.write_text(script)
    result = subprocess.run([sys.executable, '-I', str(worker)], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode != 0
    metadata = json.loads(output.read_text())
    assert metadata['status'] == 'failed'
    assert metadata['stage'] == 'kernel-pip-preflight' and metadata['returncode'] == 73
    assert metadata['stderr'] == 'fixture child failure'
    def forbidden(*a, **kw):
        pytest.fail('attempted download of paths that setup never produced')
    job.retrieve = forbidden
    job.retrieve_extra(tmp_path, '/unused')
    assert job.extra_path is None
    with pytest.raises(ValueError, match='kernel-pip-preflight'):
        job.verify_result(metadata, {})
