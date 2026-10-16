"""Lifecycle fixtures use fake CLI processes; no network or allocations."""
import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from rl_stocks.runtime.colab.fleet import Fleet,validate_tasks
from rl_stocks.runtime.colab.jobs import job_plan
from rl_stocks.runtime.colab.payloads import prepare_payload


class FakeCommand:
    def __init__(self,path,delay=0,code=0,callback=None,expired=False):
        self.path=path;self.remaining=delay;self.code=code;self.callback=callback
        self.timed_out=expired
    def poll(self):
        if self.callback:self.callback();self.callback=None
        if self.remaining:self.remaining-=1;return None
        return self.code
    def expired(self):return self.timed_out
    def close(self):pass
    def terminate(self):self.remaining=0


class FakeCLI:
    command=['fake-colab']
    def __init__(self,root,fail_meter=False,fail_stop=False,interrupt=False,ready_failures=0,download_timeouts=0):
        self.root=root;self.active={};self.high_water=0;self.calls=[];self.meters=0
        self.fail_meter=fail_meter;self.fail_stop=fail_stop;self.interrupt=interrupt;self.fleet=None
        self.ready_failures=ready_failures
        self.download_timeouts=download_timeouts
    def short(self,args,timeout=25):
        if args==['usage']:
            self.meters+=1
            if self.fail_meter and self.meters>=4:raise RuntimeError('private authenticated transport failure')
            return f'Current balance: 150 compute units\nUsage rate: {len(self.active)*.08}/hr\nActive assignments: {len(self.active)}\n'
        if args==['sessions']:
            return '\n'.join(f'[{n}] endpoint-{n} | Hardware: {h}' for n,h in self.active.items()) if self.active else 'No active sessions found'
        return 'Version: 0.7.4'
    def launch(self,args,path,timeout):
        self.calls.append(args);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('')
        name=args[args.index('-s')+1] if '-s' in args else None
        if args[0]=='new':
            hardware=args[args.index('--gpu')+1] if '--gpu' in args else 'CPU'
            self.active[name]=hardware;self.high_water=max(self.high_water,len(self.active))
            (self.root/'.runtime/sessions.json').write_text(json.dumps({n:{'endpoint':'endpoint-'+n} for n in self.active}))
        elif args[0]=='status':path.write_text('Hardware: '+self.active[name])
        elif args[0]=='exec':
            callback=(lambda:setattr(self.fleet,'interrupted',True)) if self.interrupt else None
            return FakeCommand(path,delay=300,callback=callback)
        elif args[0]=='download':
            destination=Path(args[-1]);destination.parent.mkdir(parents=True,exist_ok=True)
            if args[-2].endswith('research-artifacts.tar.gz') and self.download_timeouts:
                self.download_timeouts-=1
                return FakeCommand(path,delay=999,expired=True)
            if args[-2].endswith('.ready'):
                if self.ready_failures:
                    self.ready_failures-=1;path.write_text('Download failed: binary download failed: ReadTimeout')
                    return FakeCommand(path,code=1)
                destination.touch()
            else:
                with tarfile.open(destination,'w:gz') as archive:
                    info=tarfile.TarInfo('artifacts/fixture.json');data=b'{}';info.size=len(data);archive.addfile(info,io.BytesIO(data))
        elif args[0]=='stop':
            if not self.fail_stop:self.active.pop(name,None)
        elif args[0]=='sessions':
            path.write_text('\n'.join(f'[{n}] endpoint-{n} | Hardware: {h}' for n,h in self.active.items()) if self.active else 'No active sessions found')
        return FakeCommand(path)


class FleetTests(unittest.TestCase):
    def test_final_download_command_timeouts_use_bounded_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root,cli,fleet=self.fixture(directory,count=1,download_timeouts=2)
            self.assertEqual(self.run_fixture(fleet)['status'],'complete')
            self.assertEqual(sum(c[0]=='download' and c[-2].endswith('research-artifacts.tar.gz') for c in cli.calls),3)
    def test_marker_timeouts_retry_but_remain_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root,cli,fleet=self.fixture(directory,count=1,ready_failures=2)
            self.assertEqual(self.run_fixture(fleet)['status'],'complete')
            self.assertEqual(sum(c[0]=='download' and c[-2].endswith('.ready') for c in cli.calls),3)
        with tempfile.TemporaryDirectory() as directory:
            root,cli,fleet=self.fixture(directory,count=1,ready_failures=3)
            with self.assertRaises(RuntimeError):self.run_fixture(fleet)
            self.assertEqual(cli.active,{})
            self.assertEqual(sum(c[0]=='download' and c[-2].endswith('.ready') for c in cli.calls),3)
    def fixture(self,directory,count=4,**options):
        root=Path(directory)
        for name in ('src','scripts','configs','.runtime','dataset'): (root/name).mkdir()
        for name in ('pyproject.toml','requirements.txt','requirements.lock'):(root/name).write_text('fixture')
        (root/'dataset/dataset.json').write_text('{}')
        tasks=[{'id':'cell-'+str(i),'dataset':'dataset','plan':job_plan(accelerator='T4' if i%2 else 'CPU')} for i in range(count)]
        cli=FakeCLI(root,**options);fleet=Fleet(root,cli,tasks,count,root/'.runtime/state.json');cli.fleet=fleet
        return root,cli,fleet

    def run_fixture(self,fleet):
        timer=[0.]
        def monotonic():timer[0]+=.25;return timer[0]
        def validate(path,plan,dataset):
            return [{**j,'actual_steps':j['steps'],'seed':j.get('seed',plan.get('seed')),'bundle_id':'fixture'} for j in plan['jobs']]
        with patch('rl_stocks.runtime.colab.fleet.time.sleep'),patch('rl_stocks.runtime.colab.fleet.time.monotonic',side_effect=monotonic),patch('rl_stocks.runtime.colab.fleet.validate_archive',side_effect=validate),contextlib.redirect_stdout(io.StringIO()):
            return fleet.run()

    def test_four_concurrent_mixed_hardware_cells_and_verified_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root,cli,fleet=self.fixture(directory)
            result=self.run_fixture(fleet)
            self.assertEqual(result['status'],'complete');self.assertEqual(len(result['completed']),4)
            self.assertEqual(cli.high_water,4);self.assertEqual(cli.active,{})
            self.assertGreater(cli.meters,5)
            ledger=json.loads((root/'.runtime/usage-ledger.json').read_text())
            self.assertEqual(len(ledger['attempts']),1)
            self.assertEqual(ledger['attempts'][0]['kind'],'fleet')
            self.assertTrue(all(c['cleanup_verified'] for c in ledger['attempts'][0]['children'].values()))
            self.assertEqual(len(set(a.payload for a in fleet.actors)),4)

    def test_metering_failure_and_interruption_stop_all_owned_sessions(self):
        for options,budget_stopped in [({'fail_meter':True},True),({'interrupt':True},False)]:
            with self.subTest(options=options),tempfile.TemporaryDirectory() as directory:
                root,cli,fleet=self.fixture(directory,**options)
                with self.assertRaises(RuntimeError):self.run_fixture(fleet)
                self.assertEqual(cli.active,{})
                ledger=json.loads((root/'.runtime/usage-ledger.json').read_text())
                self.assertEqual(ledger['budget_stopped'],budget_stopped)
                self.assertEqual(ledger['attempts'][-1]['status'],'stopped')

    def test_cleanup_failure_preserves_ids_and_rejects_another_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            root,cli,fleet=self.fixture(directory,count=1,fail_stop=True)
            with self.assertRaises(RuntimeError):self.run_fixture(fleet)
            ledger=json.loads((root/'.runtime/usage-ledger.json').read_text())
            self.assertEqual(ledger['attempts'][-1]['status'],'cleanup_failed')
            child=next(iter(ledger['attempts'][-1]['children'].values()))
            self.assertIn(child['session'],cli.active)
            self.assertFalse(child['cleanup_verified'])
            from rl_stocks.runtime.colab.fleet import recover
            cli.fail_stop=False
            with contextlib.redirect_stdout(io.StringIO()):recover(root,cli,fleet.name,root/'.runtime/state.json')
            self.assertEqual(cli.active,{})
            self.assertEqual(json.loads((root/'.runtime/usage-ledger.json').read_text())['attempts'][-1]['status'],'stopped')

    def test_existing_unowned_session_is_never_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root,cli,fleet=self.fixture(directory,count=1);cli.active['unrelated']='CPU'
            with self.assertRaises(Exception):self.run_fixture(fleet)
            self.assertEqual(cli.active,{'unrelated':'CPU'})
            self.assertFalse(any(c[0]=='stop' for c in cli.calls))

    def test_payloads_and_task_validation_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root,cli,fleet=self.fixture(directory,count=1)
            left=job_plan(accelerator='CPU');right=job_plan('sac','T4')
            a=prepare_payload(root,root/'dataset',left,'left');b=prepare_payload(root,root/'dataset',right,'right')
            for path,expected in [(a,left),(b,right)]:
                with tarfile.open(path) as archive:self.assertEqual(json.load(archive.extractfile('job-plan.json')),expected)
            with self.assertRaises(ValueError):validate_tasks([{'id':'../escape','dataset':'dataset','plan':left}],root)
            with self.assertRaises(ValueError):Fleet(root,cli,fleet.tasks,0,root/'.runtime/new.json')
