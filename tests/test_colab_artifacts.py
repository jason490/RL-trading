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

    def test_per_job_seeds_are_verified(self):
        seeds=(11,23,37,11,23)
        self.plan={'version':3,'accelerator':'A100','jobs':[
            {**job,'seed':seed} for job,seed in zip(self.plan['jobs'],seeds)]}
        for index,seed in enumerate(seeds):
            self.runs[index]['seed']=seed
            (self.artifacts/'runs'/str(index)/'metadata.json').write_text(json.dumps(self.runs[index]))
        self.completion.update(plan=self.plan,completed=self.runs)
        self.write_completion()
        self.assertEqual(len(self.check()),5)
        self.runs[-1]['seed']=42
        (self.artifacts/'runs/4/metadata.json').write_text(json.dumps(self.runs[-1]))
        with self.assertRaisesRegex(RuntimeError,'missing or unexpected'):
            self.check()

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

    def test_cpu_preflight_hardware_and_jobs(self):
        self.plan=job_plan(accelerator='CPU')
        for index,job in enumerate(self.plan['jobs']):
            self.runs[index].update(job,requested_steps=job['steps'],actual_steps=job['steps'])
            (self.artifacts/'runs'/str(index)/'metadata.json').write_text(json.dumps(self.runs[index]))
        for index in (3,4):
            (self.artifacts/'runs'/str(index)/'metadata.json').unlink()
        self.completion.update(plan=self.plan,gpu=None,completed=self.runs[:3])
        self.write_completion()
        self.assertEqual(len(self.check()),3)
        self.completion['gpu']='NVIDIA A100'
        self.write_completion()
        with self.assertRaisesRegex(RuntimeError,'manifest mismatch'):self.check()

    def test_completed_artifacts_survive_cli_transport_failure_and_cleanup(self):
        import contextlib
        import io
        import shutil
        from rl_stocks.runtime.colab.budget import Usage
        from rl_stocks.runtime.colab.runner import ColabJob
        self.check()  # Produce integrity fixtures before output directories exist.
        for folder in ('src','scripts','configs','dataset-id'):
            (self.root/folder).mkdir()
        for name in ('pyproject.toml','requirements.txt','requirements.lock'):
            (self.root/name).write_text('mock lifecycle fixture')
        (self.root/'dataset-id/dataset.json').write_text('{}')
        source=self.root/'result.tar.gz'
        class TransportFailureJob(ColabJob):
            remote_active=False
            def meter(self):return Usage(150,5.3,1) if self.remote_active else Usage(150,0,0)
            def short(self,args,timeout=25):
                if args[0]=='stop':self.remote_active=False
                return 'No active sessions found' if args[0]=='sessions' else 'A100'
            def operation(self,args,timeout):
                if args[0]=='new':
                    self.remote_active=True
                    (self.state/'sessions.json').write_text(json.dumps({self.session:{'name':self.session,'endpoint':'owned'}}))
                if args[0]=='exec':raise RuntimeError('output stream failed after completion')
            def retrieve(self,remote,local):
                shutil.copyfile(source,local)
                self.ledger.active['retrieved'].append(str(local));self.ledger.save()
                return True
        job=TransportFailureJob(self.root,'python','A100')
        with contextlib.redirect_stdout(io.StringIO()):
            job.run(self.root/'dataset-id','ppo')
        self.assertTrue(job.ledger.active['exec_cli_failed'])
        self.assertTrue(job.ledger.active['training_verified'])
        self.assertEqual(job.ledger.active['status'],'stopped')
