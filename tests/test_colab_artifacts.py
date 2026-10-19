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

    def test_learning_curve_config_intervals_and_csv_are_verified(self):
        from rl_stocks.runtime.colab.jobs import learning_plan
        self.plan=learning_plan(11,50000)
        job=self.plan['jobs'][0];run=self.runs[0]
        curve=[{'steps':step,'phase':'post_update' if step==50000 else 'on_step','score':.01}
               for step in range(10000,50001,10000)]
        run.update(job,requested_steps=50000,actual_steps=50000,
                   reward_config={'kind':'net_log_equity','scale':1.},validation_checkpoints=curve)
        folder=self.artifacts/'runs/0'
        (folder/'metadata.json').write_text(json.dumps(run))
        for index in (1,2,3,4):(self.artifacts/'runs'/str(index)/'metadata.json').unlink()
        journal=folder/'learning-curve.jsonl'
        journal.write_text(''.join(json.dumps(r)+'\n' for r in curve))
        csv=folder/'learning-curve.csv'
        csv.write_text('steps,return_pct\n'+''.join(f"{r['steps']},1.0\n" for r in curve))
        (folder/'learning-curve.html').write_text('curve')
        self.completion.update(plan=self.plan,gpu=None,completed=[run]);self.write_completion()
        self.assertEqual(len(self.check()),1)
        journal.write_text(''.join(json.dumps(r)+'\n' for r in curve[:-1]))
        with self.assertRaisesRegex(RuntimeError,'learning curve'):self.check()
        journal.write_text(''.join(json.dumps(r)+'\n' for r in curve))
        csv.write_text(csv.read_text().replace('1.0','999.0'))
        with self.assertRaisesRegex(RuntimeError,'CSV mismatch'):self.check()
        run['reward_config']['scale']=10.
        (folder/'metadata.json').write_text(json.dumps(run))
        with self.assertRaisesRegex(RuntimeError,'configuration mismatch'):self.check()

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

    def test_initialization_and_smoke_curve_contract_are_verified(self):
        from rl_stocks.runtime.colab.jobs import initialization_plan
        from rl_stocks.training.initialization import initialization_spec
        self.plan=initialization_plan('wide_low_noise',smoke=True)
        job=self.plan['jobs'][0];run=self.runs[0]
        curve=[{'steps':256,'phase':'on_step','score':.01},
               {'steps':512,'phase':'post_update','score':.01}]
        run.update(job,requested_steps=512,actual_steps=512,
                   initialization=initialization_spec('wide_low_noise'),
                   reward_config={'kind':'net_log_equity','scale':1.},validation_checkpoints=curve)
        folder=self.artifacts/'runs/0'
        for index in (1,2,3,4):(self.artifacts/'runs'/str(index)/'metadata.json').unlink()
        (folder/'learning-curve.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in curve))
        (folder/'learning-curve.csv').write_text('steps,return_pct\n256,1.0\n512,1.0\n')
        (folder/'learning-curve.html').write_text('curve')
        (folder/'metadata.json').write_text(json.dumps(run))
        self.completion.update(plan=self.plan,gpu=None,completed=[run]);self.write_completion()
        self.assertEqual(len(self.check()),1)
        run['initialization']=initialization_spec('standard')
        (folder/'metadata.json').write_text(json.dumps(run))
        with self.assertRaisesRegex(RuntimeError,'initialization-study'):self.check()

    def test_stability_export_requires_post_update_endpoint(self):
        from rl_stocks.runtime.colab.jobs import stability_plan
        self.plan=stability_plan('recurrent_ppo',11)
        job=self.plan['jobs'][0];run=self.runs[0]
        curve=[{'steps':step,'phase':'post_update' if step==100000 else 'on_step','score':.01}
               for step in range(10000,100001,10000)]
        run.update(job,requested_steps=100000,actual_steps=100000,
                   reward_config={'kind':'net_log_equity','scale':1.},validation_checkpoints=curve)
        folder=self.artifacts/'runs/0'
        (folder/'metadata.json').write_text(json.dumps(run))
        for index in (1,2,3,4):(self.artifacts/'runs'/str(index)/'metadata.json').unlink()
        journal=folder/'learning-curve.jsonl'
        journal.write_text(''.join(json.dumps(r)+'\n' for r in curve))
        (folder/'learning-curve.csv').write_text('steps,return_pct\n'+''.join(f"{r['steps']},1.0\n" for r in curve))
        (folder/'learning-curve.html').write_text('curve')
        self.completion.update(plan=self.plan,gpu=None,completed=[run]);self.write_completion()
        self.assertEqual(len(self.check()),1)
        curve[-1]['phase']='on_step';run['validation_checkpoints']=curve
        (folder/'metadata.json').write_text(json.dumps(run))
        journal.write_text(''.join(json.dumps(r)+'\n' for r in curve))
        with self.assertRaisesRegex(RuntimeError,'learning curve'):self.check()

    def test_causal_sampling_export_requires_actual_training_counts(self):
        from rl_stocks.runtime.colab.jobs import sampling_plan
        from rl_stocks.training.initialization import initialization_spec
        self.plan=sampling_plan('small_balanced',smoke=True);job=self.plan['jobs'][0];run=self.runs[0]
        curve=[{'steps':256,'phase':'on_step','score':.01},{'steps':512,'phase':'post_update','score':.01}]
        counts={'mode':'downtrend_balanced','starts':4,'downtrend_starts':2,'transitions':512,'downtrend_transitions':100}
        run.update(job,requested_steps=512,actual_steps=512,initialization=initialization_spec('wide'),
            start_sampling={'mode':'downtrend_balanced'},sampling_diagnostics=counts,
            reward_config={'kind':'net_log_equity','scale':1.},validation_checkpoints=curve)
        folder=self.artifacts/'runs/0'
        for index in (1,2,3,4):(self.artifacts/'runs'/str(index)/'metadata.json').unlink()
        (folder/'metadata.json').write_text(json.dumps(run))
        (folder/'training-sampling.json').write_text(json.dumps(counts))
        (folder/'learning-curve.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in curve))
        (folder/'learning-curve.csv').write_text('steps,return_pct\n256,1.0\n512,1.0\n')
        (folder/'learning-curve.html').write_text('curve')
        self.completion.update(plan=self.plan,gpu=None,completed=[run]);self.write_completion()
        assert len(self.check())==1
        counts['transitions']=511
        (folder/'metadata.json').write_text(json.dumps(run));(folder/'training-sampling.json').write_text(json.dumps(counts))
        with self.assertRaisesRegex(RuntimeError,'sampling diagnostic'):self.check()

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
