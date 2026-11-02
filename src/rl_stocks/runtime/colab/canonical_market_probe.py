"""Full fixed market replay under bounded CPU-only Colab lifecycle control."""
import argparse
import fcntl
import gzip
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys
import time
import zipfile

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.provenance import repository_state
from rl_stocks.runtime.colab.reference_probe import ReferenceProbeJob

PROTOCOL_SHA256 = 'b9bcb419386de24f18201e52cb84533232ba20970727eac975b8b48f59b7d249'


def protocol(root):
    spec = json.loads((Path(root) / 'configs/experiments/canonical_market_probe.json').read_text())
    if digest(spec) != PROTOCOL_SHA256:
        raise ValueError('fixed market cross-machine protocol required')
    return spec


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run_worker(root, output):
    from rl_stocks.evaluation.canonical_diagnostic import protocol as prior_protocol, validate_window
    from rl_stocks.evaluation.canonical_window import replay_window
    from rl_stocks.inference.canonical_bundle import FrozenCanonicalTaskPolicy
    root, output = Path(root), Path(output)
    spec = protocol(root)
    if list(sys.version_info[:2]) != spec['expected_python_minor']:
        raise ValueError('inspect default Colab Python before supporting its new version')
    versions = {name: version(name) for name in spec['dependencies']}
    if versions != spec['dependencies']:
        raise ValueError('hash-locked inference dependency versions required')
    view = json.loads((root / 'view.json').read_text())
    validate_window(view, prior_protocol(root))
    frozen = FrozenCanonicalTaskPolicy(root / 'frozen' / spec['reference_bundle_id'],
        feature_schema=view['schema'], feed=view['manifest']['provenance']['capabilities']['feed'],
        expected_bundle_id=spec['reference_bundle_id'])
    source = frozen.manifest['source_manifest']
    if digest(view['normalizer']) != digest(source['normalizer']):
        raise ValueError('frozen preprocessing mismatch')
    protected = digest((view, frozen.manifest))
    started = time.monotonic()
    results = replay_window(frozen, view)
    elapsed = time.monotonic() - started
    if digest((view, frozen.manifest)) != protected:
        raise ValueError('market probe inputs mutated')
    raw = canonical(results)
    paths = Path(str(output) + '.paths.gz')
    with paths.open('xb') as stream:
        stream.write(gzip.compress(raw, mtime=0))
    record = {'schema': spec['schema'], 'protocol_sha256': PROTOCOL_SHA256,
        'python': platform.python_version(), 'python_minor': list(sys.version_info[:2]),
        'versions': versions, 'reference_bundle_id': frozen.path.name,
        'results_sha256': digest(results), 'window_episodes_sha256': digest(view['episodes']),
        'episodes': len(results), 'predictions': sum(len(r['rows']) for r in results.values()),
        'forbidden_imports': sorted(set(sys.modules) & {'torch', 'stable_baselines3'}),
        'elapsed_seconds': elapsed, 'paths_sha256': file_sha(paths),
        'protected_inputs_unchanged': True, 'training_performed': False,
        'development_evaluated': False, 'retired_test_access': False,
        'performance_promotion': False, 'native_acceptance_changed': False}
    with output.open('xb') as stream:
        stream.write(canonical(record))
    print('Fixed reference market replay exported', flush=True)


def prepare(root, output):
    from rl_stocks.data.generation.datasets import read_dataset
    from rl_stocks.evaluation.canonical_diagnostic import protocol as prior_protocol, validate_window
    from rl_stocks.evaluation.training_fit import training_window
    root, output = Path(root).resolve(), Path(output).resolve()
    spec = protocol(root)
    git = repository_state(root)
    if git['dirty']:
        raise ValueError('clean committed market payload source required')
    old = prior_protocol(root)
    if digest(old) != spec['prior_market_protocol_sha256']:
        raise ValueError('fixed prior engineering protocol required')
    receipt = json.loads((root / 'artifacts/canonical-market-replay-v1/result.json').read_text())
    if (receipt['results_sha256'] != spec['reference_results_sha256']
            or receipt['reference_bundle'] != spec['reference_bundle_id']):
        raise ValueError('fixed completed host reference required')
    data = read_dataset(root / 'data/datasets' / old['dataset'])
    view = training_window(data, original_panel_only=True)
    validate_window(view, old)
    lock = root / 'configs/reference-inference.lock'
    if file_sha(lock) != spec['inference_lock_sha256']:
        raise ValueError('fixed hash-locked inference requirements required')
    output.mkdir(parents=True, exist_ok=False)
    source = root / 'artifacts/canonical-market-replay-v1/frozen' / spec['reference_bundle_id']
    archive = output / 'payload.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
        for path in sorted((root / 'src').rglob('*.py')):
            if path.is_symlink():
                raise ValueError('source symlinks unsupported')
            stream.write(path, str(path.relative_to(root)))
        for name in ('canonical_task_replay.json', 'canonical_market_probe.json'):
            path = root / 'configs/experiments' / name
            stream.write(path, str(path.relative_to(root)))
        stream.write(lock, 'configs/reference-inference.lock')
        for name in ('actor.json', 'manifest.json', 'checksums.json'):
            stream.write(source / name, f'frozen/{source.name}/{name}')
        stream.writestr('view.json', canonical(view))
    inputs = {'protocol_sha256': PROTOCOL_SHA256, 'source_revision': git['revision'], 'archive': str(archive.relative_to(root)),
              'archive_sha256': file_sha(archive)}
    (output / 'input.json').write_bytes(canonical(inputs))
    return output / 'input.json'


class CanonicalMarketJob(ReferenceProbeJob):
    limit_units = .1
    deadline_seconds = 1800
    execution_seconds = 900
    kind = 'canonical-market-cross-machine-probe'

    def validate_input(self, inputs, expected):
        spec = protocol(self.root)
        if (self.limit_units != spec['maximum_incremental_units']
                or self.deadline_seconds != spec['deadline_seconds']
                or self.execution_seconds != spec['execution_seconds']
                or self.reserve_seconds != spec['teardown_reserve_seconds']):
            raise ValueError('fixed market-probe operation limits required')
        if (inputs['protocol_sha256'] != PROTOCOL_SHA256
                or inputs['source_revision'] != repository_state(self.root)['revision']
                or file_sha(self.root / inputs['archive']) != inputs['archive_sha256']
                or expected['results_sha256'] != spec['reference_results_sha256']):
            raise ValueError('bound market payload and original host reference required')

    def prepare_remote(self, inputs):
        self.operation(['upload', '-s', self.session, str(self.root / inputs['archive']),
                        '/content/' + self.session + '-market.zip'], 120)

    def build_worker(self, inputs, remote):
        spec = protocol(self.root)
        # Install only the five existing pinned numerical inference dependencies,
        # into an isolated venv made with Colab's current default interpreter.
        return f'''import hashlib, json, os, pathlib, subprocess, sys, zipfile
output=pathlib.Path({remote!r})
stage="inspect-default-python"
try:
    assert sys.version_info[:2] == (3,13), "Inspect actual default Colab Python before updating project"
    archive=pathlib.Path({('/content/' + self.session + '-market.zip')!r})
    stage="verify-payload"
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=={inputs['archive_sha256']!r}
    root=pathlib.Path({('/content/' + self.session + '-market')!r})
    root.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as stream:
        assert len(stream.infolist()) < 1500
        for entry in stream.infolist():
            path=pathlib.PurePosixPath(entry.filename)
            assert not path.is_absolute() and ".." not in path.parts
        stream.extractall(root)
    lock=root/"configs/reference-inference.lock"
    assert hashlib.sha256(lock.read_bytes()).hexdigest()=={spec['inference_lock_sha256']!r}
    sys.path.insert(0,str(root/"src"))
    from rl_stocks.runtime.colab.inference_setup import create_environment, checked
    stage="create-inference-environment"
    python=create_environment(sys.executable,root/"venv",lock)
    program="import sys; sys.path.insert(0,sys.argv[1]+'/src'); from rl_stocks.runtime.colab.canonical_market_probe import run_worker; run_worker(sys.argv[1],sys.argv[2])"
    stage="market-replay"
    checked(stage,[python,"-I","-c",program,str(root),str(output)],420)
    print("Fixed market replay receipt exported")
except BaseException as error:
    failure={{"status":"failed","stage":stage,"error_type":type(error).__name__,
              "python":sys.version,"training_performed":False}}
    failure.update(getattr(error,"record",{{"message":str(error)[-2000:]}}))
    if not output.exists():
        output.write_text(json.dumps(failure,sort_keys=True,allow_nan=False))
    print("Market probe failed; diagnostic receipt exported")
    raise
'''

    def retrieve_extra(self, output, remote):
        metadata = json.loads((output / 'reference-probe.json').read_text())
        if metadata.get('status') == 'failed':
            self.extra_path = None
            return
        path = output / 'market-paths.json.gz'
        if not path.exists() and not self.retrieve(remote + '.paths.gz', path):
            raise RuntimeError('bounded full market path retrieval exhausted')
        self.extra_path = path

    def verify_result(self, actual, expected):
        if actual.get('status') == 'failed':
            raise ValueError('market probe failed at ' + actual.get('stage', 'unknown stage'))
        spec = protocol(self.root)
        if (actual['protocol_sha256'] != PROTOCOL_SHA256
                or actual['schema'] != spec['schema'] or actual['python_minor'] != [3, 13]
                or actual['versions'] != spec['dependencies']
                or actual['reference_bundle_id'] != spec['reference_bundle_id']
                or actual['window_episodes_sha256'] != spec['window_episodes_sha256']
                or actual['episodes'] != 26 or actual['predictions'] != 19136
                or actual['results_sha256'] != spec['reference_results_sha256']
                or actual['forbidden_imports'] != []
                or not math.isfinite(actual.get('elapsed_seconds', float('nan')))
                or not 0 < actual['elapsed_seconds'] < 420
                or actual['protected_inputs_unchanged'] is not True
                or any(actual[name] is not False for name in ('training_performed',
                    'development_evaluated', 'retired_test_access', 'performance_promotion',
                    'native_acceptance_changed'))):
            raise ValueError('complete market replay differs from fixed host reference')
        if file_sha(self.extra_path) != actual['paths_sha256']:
            raise ValueError('retrieved market paths checksum mismatch')
        with gzip.open(self.extra_path, 'rb') as stream:
            raw = stream.read(128*1024*1024+1)
        if len(raw) > 128*1024*1024 or hashlib.sha256(raw).hexdigest() != actual['results_sha256']:
            raise ValueError('retrieved full market paths differ from fixed reference')
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--input', default='.runtime/canonical-market-probe-input-v1/input.json')
    parser.add_argument('--cli-python')
    args = parser.parse_args()
    root = Path.cwd()
    if args.prepare:
        print(prepare(root, Path(args.input).parent))
        return
    if not args.cli_python:
        parser.error('--cli-python required for allocation')
    inputs = json.loads(Path(args.input).read_text())
    expected = json.loads((root / 'artifacts/canonical-market-replay-v1/result.json').read_text())
    with (root / '.runtime/allocation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        CanonicalMarketJob(root, args.cli_python, 'CPU').run_probe(
            inputs, expected, root / '.runtime/canonical-market-probe-v1-claim.json')


if __name__ == '__main__':
    main()
