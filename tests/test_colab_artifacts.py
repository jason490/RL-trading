import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.runtime.colab.jobs import job_plan


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = job_plan('ppo')
        self.artifacts = self.root / 'artifacts'
        self.artifacts.mkdir()
        self.runs = []
        for index, job in enumerate(self.plan['jobs']):
            # These fixture bytes test integrity checking, not neural inference.
            manifest = {'fixture': index}
            bundle_id = digest(manifest)
            bundle = self.artifacts / 'frozen' / bundle_id
            bundle.mkdir(parents=True)
            (bundle / 'manifest.json').write_bytes(canonical(manifest))
            (bundle / 'model.zip').write_bytes(b'fixture weights')
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}
            (bundle / 'checksums.json').write_bytes(canonical(hashes))
            run = {**job, 'seed': 11, 'dataset': 'dataset-id', 'bundle_id': bundle_id,
                   'requested_steps': job['steps'], 'actual_steps': job['steps'], 'status': 'complete'}
            folder = self.artifacts / 'runs' / str(index)
            folder.mkdir(parents=True)
            (folder / 'metadata.json').write_text(json.dumps(run))
            self.runs.append(run)
        self.completion = {'plan': self.plan, 'dataset': 'dataset-id', 'python': [3,13,16],
                           'gpu': 'NVIDIA A100', 'completed': self.runs}
        self.write_completion()

    def write_completion(self):
        (self.artifacts / 'completion.json').write_text(json.dumps(self.completion))

    def check(self):
        path = self.root / 'result.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            archive.add(self.artifacts, arcname='artifacts')
        return validate_archive(path, self.plan, 'dataset-id')

    def test_complete_plan_and_checksums(self):
        self.assertEqual(len(self.check()), 5)

    def test_remote_exception_rejected_despite_cli_success(self):
        (self.artifacts / 'failure.txt').write_text('remote exception')
        with self.assertRaisesRegex(RuntimeError, 'remote training failed'):
            self.check()

    def test_wrong_python_rejected(self):
        self.completion['python'] = [3,12,14]
        self.write_completion()
        with self.assertRaisesRegex(RuntimeError, 'manifest mismatch'):
            self.check()

    def test_missing_job_rejected(self):
        (self.artifacts / 'runs/4/metadata.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'missing or unexpected'):
            self.check()

    def test_corrupted_weights_rejected(self):
        next(self.artifacts.glob('frozen/*/model.zip')).write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
            self.check()
