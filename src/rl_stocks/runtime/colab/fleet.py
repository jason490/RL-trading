"""Concurrent named runtimes with one aggregate meter and bounded owned cleanup."""
import argparse
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tarfile
import time
import uuid

from rl_stocks.core.provenance import repository_state
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.runtime.colab.budget import BudgetStop,execution_window,parse_usage
from rl_stocks.runtime.colab.fleet_budget import FleetBudget
from rl_stocks.runtime.colab.jobs import validate_plan
from rl_stocks.runtime.colab.payloads import prepare_payload
from rl_stocks.runtime.colab.transport import connection_lost,worker_not_started


class Command:
    def __init__(self,command,args,root,path,timeout):
        self.path=path;path.parent.mkdir(parents=True,exist_ok=True)
        self.output=path.open('w');os.chmod(path,0o600)
        self.process=subprocess.Popen(command+args,cwd=root,stdout=self.output,stderr=self.output,
            stdin=subprocess.DEVNULL,start_new_session=True)
        self.started=time.monotonic();self.timeout=timeout

    def poll(self):return self.process.poll()
    def expired(self):return time.monotonic()-self.started>self.timeout
    def close(self):self.output.close()
    def terminate(self):
        if self.poll() is None:
            os.killpg(self.process.pid,signal.SIGTERM)
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGKILL);self.process.wait(timeout=5)
        self.close()


class CLI:
    def __init__(self,root,python):
        self.root=root;self.command=[str(python),str(root/'scripts/colab_safe.py')]
    def short(self,args,timeout=25):
        p=subprocess.run(self.command+args,cwd=self.root,capture_output=True,text=True,timeout=timeout,stdin=subprocess.DEVNULL)
        if p.returncode:raise RuntimeError('Colab command failed: '+args[0])
        return p.stdout
    def launch(self,args,path,timeout):return Command(self.command,args,self.root,path,timeout)


@dataclass
class Runtime:
    task:dict
    session:str
    payload:Path|None=None
    phase:str='pending'
    command:object=None
    snapshot:object=None
    sequence:int=0
    install_retries:int=0
    exec_retries:int=0
    probe_failures:int=0
    download_attempts:int=0
    stop_attempts:int=0
    exec_deadline:float=0.
    probe_due:float=0.
    snapshot_due:float=0.
    cancelled:bool=False


def validate_tasks(tasks,root):
    if not isinstance(tasks,list) or not tasks:raise ValueError('nonempty task list required')
    seen=set();result=[]
    for task in tasks:
        if set(task)!={'id','dataset','plan'} or not re.fullmatch(r'[A-Za-z0-9_-]+',task['id']) or task['id'] in seen:
            raise ValueError('unique safe task IDs and exact task fields required')
        validate_plan(task['plan']);plan=task['plan']
        if plan['accelerator']=='A100' and not any(not j['smoke'] for j in plan['jobs']):
            raise ValueError('A100 requires research, not smoke-only allocation')
        dataset=(root/task['dataset']).resolve()
        if not (dataset/'dataset.json').is_file():raise ValueError('immutable dataset missing')
        if plan.get('version')==15:
            from rl_stocks.data.generation.datasets import read_dataset
            from rl_stocks.training.history.spec import validate_payload
            if dataset.name!=plan['dataset_contract']['identity']:raise ValueError('historical task dataset identity mismatch')
            validate_payload(read_dataset(dataset),plan['history_study']['arm'])
        if plan.get('version')==12:
            from rl_stocks.data.generation.datasets import read_dataset
            from rl_stocks.runtime.colab.jobs import validate_context_payload
            if dataset.name!=plan['dataset_contract']['identity']:raise ValueError('trend-context task dataset identity mismatch')
            validate_context_payload(read_dataset(dataset),plan)
        if plan.get('version') in (13,14):
            from rl_stocks.data.generation.datasets import read_dataset
            from rl_stocks.training.supervised.spec import validate_payload
            if dataset.name!=plan['dataset_contract']['identity']:raise ValueError('signal task dataset identity mismatch')
            validate_payload(read_dataset(dataset))
        result.append({**task,'dataset':str(dataset)});seen.add(task['id'])
    return result


class Fleet:
    def __init__(self,root,cli,tasks,concurrency,state_path,study_limit=None,study_start=None):
        if type(concurrency) is not int or concurrency<1:raise ValueError('positive configurable concurrency required')
        self.root=Path(root).resolve();self.cli=cli;self.tasks=validate_tasks(tasks,self.root)
        self.concurrency=concurrency;self.state_path=Path(state_path)
        self.budget=FleetBudget(self.root/'.runtime/usage-ledger.json')
        self.name='rl-stocks-fleet-'+uuid.uuid4().hex[:12]
        self.actors=[Runtime(t,'rl-stocks-'+t['plan']['accelerator'].lower()+'-'+uuid.uuid4().hex[:12]) for t in self.tasks]
        self.state={'fleet':self.name,'status':'preflight','completed':{},'initial_ledger_units':self.budget.ledger.value['consumed']}
        self.study_budget=None
        if study_start is not None and study_limit is None:raise ValueError('study start requires an allowance')
        if study_limit is not None:
            baseline=self.budget.ledger.value['consumed'] if study_start is None else study_start
            if not math.isfinite(study_limit) or study_limit<=0 or not math.isfinite(baseline) or not 0<=baseline<=self.budget.ledger.value['consumed']:
                raise ValueError('positive finite study allowance and valid existing baseline required')
            self.study_budget={'start_units':baseline,'limit_units':study_limit}
            self.state['study_budget']=self.study_budget
        self.last_meter=0.;self.fatal=None;self.interrupted=False

    def guard_study(self,seconds=60):
        if self.study_budget is None:return
        ledger=self.budget.ledger;active=self.budget.active
        projected=ledger.value['consumed']-self.study_budget['start_units']+(
            ledger.clock()-active['last_time']+max(360,seconds))*(active['rate']+.01)/3600
        if projected>=self.study_budget['limit_units']:
            self.state['study_budget_stopped']=True;self.save()
            ledger.value['budget_stopped']=True
            active['stop_reason']='study allowance exhausted; stop owned fleet and require review'
            ledger.save();raise BudgetStop(active['stop_reason'])

    def save(self):
        self.state['sessions']={a.session:{'task':a.task['id'],'phase':a.phase} for a in self.actors if a.phase!='pending'}
        self.state_path.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.state_path.with_suffix('.tmp')
        with temporary.open('w') as stream:
            os.chmod(temporary,0o600);json.dump(self.state,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        temporary.replace(self.state_path)

    def meter(self):
        self.last_meter=time.monotonic()
        try:
            usage=parse_usage(self.cli.short(['usage']))
            listing=self.cli.short(['sessions'])
        except Exception:
            if self.interrupted:raise KeyboardInterrupt()
            self.budget.halt('aggregate usage unavailable; stop all owned sessions')
        self.budget.observe(usage)
        rows=re.findall(r'^\[([^\]]+)\]\s+(\S+)\s+\|\s+Hardware:',listing,re.M)
        if not rows and 'No active sessions found' not in listing:
            self.budget.halt('assignment ownership listing unavailable')
        children=self.budget.active['children']
        endpoints={c.get('endpoint') for c in children.values() if c.get('endpoint')}
        if any(name not in children and endpoint not in endpoints for name,endpoint in rows):
            self.budget.halt('unowned account assignment detected')
        return usage

    def launch(self,a,phase,args,timeout,teardown=False):
        if not teardown:
            self.guard_study(timeout);self.budget.guard(a.session,timeout)
        a.phase=phase;a.sequence+=1
        path=self.root/'.runtime'/a.session/f'{phase}-{a.sequence}-output.txt'
        a.command=self.cli.launch(args,path,timeout)
        self.budget.active['children'][a.session]['phase']=phase
        self.budget.ledger.save();self.save()

    def download(self,a,phase,remote,name,timeout=120,teardown=False):
        output=self.root/'artifacts/colab'/a.session;output.mkdir(parents=True,exist_ok=True)
        self.launch(a,phase,['download','-s',a.session,remote,str(output/name)],timeout,teardown)

    def execute(self,a):
        child=self.budget.active['children'][a.session]
        long=any(j['steps']>=50000 for j in a.task['plan']['jobs'])
        preferred=6600 if any(j['steps']==250000 for j in a.task['plan']['jobs']) else 5400 if long else 1800
        timeout=execution_window(child['deadline']-time.time(),preferred,600 if long else 300)
        a.exec_deadline=time.monotonic()+timeout;a.snapshot_due=time.monotonic()+60
        self.launch(a,'exec',['exec','-s',a.session,'-f',str(self.root/'scripts/remote_smoke.py')],timeout)

    def stop(self,a):
        if a.command:a.command.terminate();a.command=None
        if a.snapshot:a.snapshot.terminate();a.snapshot=None
        a.stop_attempts+=1
        self.budget.active['children'][a.session]['status']='stopping'
        self.launch(a,'stop',['stop','-s',a.session],30,teardown=True)

    def cancel(self,a):
        if a.phase in ('pending','done','stop','verify','salvage'):return
        was_running=a.exec_deadline>0;a.cancelled=True
        if a.command:a.command.terminate();a.command=None
        if a.snapshot:a.snapshot.terminate();a.snapshot=None
        if was_running and not self.budget.ledger.value['budget_stopped']:
            try:
                self.download(a,'salvage','/content/research-checkpoint.tar.gz','checkpoint.tar.gz',45)
                return
            except Exception:pass
        self.stop(a)

    def fail(self,error):
        if self.fatal is None:
            self.fatal=type(error).__name__;self.state.update(status='stopping',error_type=self.fatal);self.save()
        for a in self.actors:self.cancel(a)

    def snapshot(self,a):
        if a.snapshot:
            code=a.snapshot.poll()
            if code is not None or a.snapshot.expired():
                if code is None:a.snapshot.terminate()
                else:a.snapshot.close()
                if code==0:
                    path=str(self.root/'artifacts/colab'/a.session/'checkpoint.tar.gz')
                    retrieved=self.budget.active['children'][a.session]['retrieved']
                    if path not in retrieved:retrieved.append(path);self.budget.ledger.save()
                a.snapshot=None
        if a.phase in ('exec','ready','await_export') and not a.snapshot and time.monotonic()>=a.snapshot_due:
            self.budget.guard(a.session,50)
            folder=self.root/'artifacts/colab'/a.session;folder.mkdir(parents=True,exist_ok=True)
            a.snapshot=self.cli.launch(['download','-s',a.session,'/content/research-checkpoint.tar.gz',str(folder/'checkpoint.tar.gz')],
                self.root/'.runtime'/a.session/'snapshot-output.txt',45)
            a.snapshot_due=time.monotonic()+60

    def completed_command(self,a,code,path):
        phase=a.phase;child=self.budget.active['children'][a.session]
        if phase=='new':
            if code:raise RuntimeError('runtime allocation failed')
            raw=json.loads((self.root/'.runtime/sessions.json').read_text())
            child['endpoint']=raw.get(a.session,{}).get('endpoint');child['status']='active'
            usage=self.meter()
            if child['accelerator']!='CPU' and usage.rate<=0:self.budget.halt('GPU consumption rate unavailable')
            self.launch(a,'status',['status','-s',a.session],25)
        elif phase=='status':
            if code or child['accelerator'] not in path.read_text(errors='replace').upper():raise RuntimeError('allocated hardware mismatch')
            self.launch(a,'upload',['upload','-s',a.session,str(a.payload),'/content/research-upload.tar.gz'],120)
        elif phase=='upload':
            if code:raise RuntimeError('runtime upload failed')
            self.launch(a,'install',['install','-s',a.session,'uv==0.13.1'],180)
        elif phase=='install':
            if code:
                if connection_lost(path) and not a.install_retries:
                    a.install_retries+=1;self.launch(a,'install',['install','-s',a.session,'uv==0.13.1'],180);return
                raise RuntimeError('runtime bootstrap install failed')
            self.execute(a)
        elif phase=='exec':
            if code:
                child['exec_cli_failed']=True
                if not a.exec_retries and worker_not_started(self.root,a.session):
                    a.exec_retries+=1;self.execute(a);return
            a.phase='await_export';a.probe_due=0.;child['phase']=a.phase;self.budget.ledger.save();self.save()
        elif phase=='ready':
            if code:
                message=path.read_text(errors='replace')
                if 'HTTP 404' in message:a.probe_failures=0
                else:
                    a.probe_failures+=1
                    if any(status in message for status in ('HTTP 401','HTTP 403')) or a.probe_failures>=3:
                        raise RuntimeError('remote export status unavailable after bounded retries')
                a.phase='await_export';a.probe_due=time.monotonic()+10
            else:
                a.download_attempts=1;self.download(a,'download','/content/research-artifacts.tar.gz','artifacts.tar.gz')
        elif phase=='download':
            if code:
                if a.download_attempts>=3:raise RuntimeError('final download retries exhausted')
                a.download_attempts+=1;self.download(a,'download','/content/research-artifacts.tar.gz','artifacts.tar.gz');return
            output=self.root/'artifacts/colab'/a.session;archive=output/'artifacts.tar.gz'
            runs=validate_archive(archive,a.task['plan'],Path(a.task['dataset']).name)
            with tarfile.open(archive) as tar:tar.extractall(output/'retrieved',filter='data')
            child['training_verified']=True;child['retrieved'].append(str(archive))
            self.state['completed'][a.task['id']]={'session':a.session,'root':str(output/'retrieved/artifacts'),
                'runs':[{k:r[k] for k in ('algorithm','seed','actual_steps','bundle_id')} for r in runs]}
            self.save();self.stop(a)
        elif phase=='salvage':
            if code==0:child['retrieved'].append(str(self.root/'artifacts/colab'/a.session/'checkpoint.tar.gz'))
            self.stop(a)
        elif phase=='stop':self.launch(a,'verify',['sessions'],25,teardown=True)
        elif phase=='verify':
            listing=path.read_text(errors='replace');endpoint=child.get('endpoint')
            valid=not code and ('No active sessions found' in listing or 'Hardware:' in listing)
            absent=valid and f'[{a.session}]' not in listing and (not endpoint or endpoint not in listing)
            # Without an endpoint, a listing containing other assignments cannot prove ownership absence.
            absent=absent and (bool(endpoint) or 'No active sessions found' in listing)
            if absent:
                self.budget.finish_child(a.session,True);a.phase='done';self.save()
            elif a.stop_attempts<3:self.stop(a)
            else:
                self.budget.finish_child(a.session,False);a.phase='done';self.save()
                raise RuntimeError('owned runtime cleanup not verified')

    def step(self,a):
        if a.phase in ('pending','done'):return
        if not self.fatal:self.snapshot(a)
        if a.phase=='await_export':
            if time.monotonic()+20>=a.exec_deadline:raise TimeoutError('remote export deadline')
            if time.monotonic()>=a.probe_due:
                self.download(a,'ready','/content/research-artifacts.ready','export.ready',20)
            return
        if a.command is None:return
        command=a.command;code=command.poll()
        lost=a.phase in ('exec','install') and connection_lost(command.path)
        if code is None and not lost and not command.expired():return
        if code is None:
            command.terminate()
            if not lost:
                if a.phase in ('stop','verify','salvage','ready','download'):code=1
                else:raise TimeoutError('runtime operation deadline: '+a.phase)
            else:code=1
        else:command.close()
        a.command=None;self.completed_command(a,code,command.path)

    def run(self):
        state_dir=self.root/'.runtime';state_dir.mkdir(exist_ok=True)
        with (state_dir/'allocation.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if self.state_path.exists():raise RuntimeError('existing fleet state requires recovery review')
            if self.budget.ledger.value['budget_stopped'] or any(a['status']!='stopped' for a in self.budget.ledger.value['attempts']):
                raise BudgetStop('project stop or unresolved ownership requires manual review')
            if not (self.root/'requirements.lock').is_file():raise RuntimeError('dependency lock missing')
            if any(not j['smoke'] for t in self.tasks for j in t['plan']['jobs']):
                git=repository_state(self.root)
                if git.get('dirty')!='' or not git.get('revision'):raise RuntimeError('clean committed research source required')
            self.cli.short(['version'])
            for cmd in ('usage','new','status','install','exec','upload','download','stop','sessions'):self.cli.short([cmd,'--help'])
            for a in self.actors:a.payload=prepare_payload(self.root,a.task['dataset'],a.task['plan'],a.session)
            usage=parse_usage(self.cli.short(['usage']));self.budget.start(self.name,usage,len(self.tasks))
            self.last_meter=time.monotonic();self.state['status']='running';self.save()
            previous={s:signal.signal(s,lambda *_:setattr(self,'interrupted',True)) for s in (signal.SIGINT,signal.SIGTERM)}
            cleanup_deadline=None
            try:
                while True:
                    if self.interrupted and not self.fatal:self.fail(KeyboardInterrupt())
                    if time.monotonic()-self.last_meter>=60:
                        try:self.meter()
                        except BaseException as error:self.fail(error)
                    active=[a for a in self.actors if a.phase not in ('pending','done')]
                    if self.fatal and cleanup_deadline is None:cleanup_deadline=time.monotonic()+360
                    if cleanup_deadline and time.monotonic()>cleanup_deadline:break
                    if not self.fatal:
                        try:
                            self.guard_study(60)
                            for a in active:
                                if a.phase not in ('stop','verify'):self.budget.guard(a.session,60)
                            pending=next((a for a in self.actors if a.phase=='pending'),None)
                            if pending and len(active)<self.concurrency and not any(a.phase in ('new','status') for a in active):
                                self.budget.admit(pending.session,pending.task['plan']['accelerator'])
                                pending.phase='admitted'
                                args=['new','-s',pending.session]
                                if pending.task['plan']['accelerator']!='CPU':args+=['--gpu',pending.task['plan']['accelerator']]
                                self.launch(pending,'new',args,180)
                        except BaseException as error:self.fail(error)
                    for a in self.actors:
                        if time.monotonic()-self.last_meter>=60:
                            try:self.meter()
                            except BaseException as error:self.fail(error)
                        try:self.step(a)
                        except BaseException as error:
                            if not self.fatal:self.state.update(failed_task=a.task['id'],failure_phase=a.phase)
                            self.fail(error)
                    if not any(a.phase not in ('pending','done') for a in self.actors):
                        if self.fatal or all(a.phase=='done' for a in self.actors):break
                    time.sleep(.5)
            except BaseException as error:
                self.fail(error)
                # Handled code-path failures still receive a bounded cleanup loop.
                deadline=time.monotonic()+360
                while any(a.phase not in ('pending','done') for a in self.actors) and time.monotonic()<deadline:
                    for a in self.actors:
                        try:self.step(a)
                        except BaseException:pass
                    time.sleep(.5)
            finally:
                for a in self.actors:
                    if a.command:a.command.terminate();a.command=None
                    if a.snapshot:a.snapshot.terminate();a.snapshot=None
                    if a.session in self.budget.active['children'] and a.phase!='done':self.budget.finish_child(a.session,False)
                try:self.meter()
                except BaseException:pass
                clean=self.budget.finish()
                self.state.update(status='complete' if clean and not self.fatal and len(self.state['completed'])==len(self.tasks) else 'failed',cleanup_verified=clean)
                self.save()
                for s,handler in previous.items():signal.signal(s,handler)
                summary={'fleet':self.name,'completed_cells':len(self.state['completed']),
                    'conservative_project_units':self.budget.ledger.value['consumed'],
                    'remaining_project_units':150-self.budget.ledger.value['consumed'],'cleanup_verified':clean,
                    'budget_stopped':self.budget.ledger.value['budget_stopped'],
                    'metering_unreliable':self.budget.active.get('metering_unreliable',False)}
                print(json.dumps(summary),flush=True)
                for child in self.budget.active['children'].values():
                    if child['status']!='stopped':print('Cleanup failed. Session: '+child['session']+'. Recovery: '+
                        ' '.join(self.cli.command+['stop','-s',child['session']]),flush=True)
            if self.state['status']!='complete':raise RuntimeError('fleet failed; inspect private state and owned recovery IDs')
            return self.state


def recover(root,cli,fleet_name,state_path):
    """Concurrently stop recorded children; never allocate or clear budget stops."""
    root=Path(root);state_path=Path(state_path)
    with (root/'.runtime/allocation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget=FleetBudget(root/'.runtime/usage-ledger.json')
        state=json.loads(state_path.read_text())
        if (budget.active.get('kind')!='fleet' or budget.active['session']!=fleet_name or
                state.get('fleet')!=fleet_name):
            raise ValueError('exact recorded fleet and matching state file required')
        fleet=object.__new__(Fleet)
        fleet.root=root;fleet.cli=cli;fleet.budget=budget;fleet.state=state
        fleet.state_path=state_path;fleet.name=fleet_name;fleet.last_meter=0.
        fleet.fatal='Recovery';fleet.interrupted=False;fleet.actors=[]
        for name,child in budget.active['children'].items():
            task={'id':state.get('sessions',{}).get(name,{}).get('task',name)}
            fleet.actors.append(Runtime(task,name,phase='done' if child['status']=='stopped' else 'recover'))
        previous={s:signal.signal(s,signal.SIG_IGN) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            for actor in fleet.actors:
                if actor.phase!='done':fleet.stop(actor)
            deadline=time.monotonic()+360
            while any(a.phase!='done' for a in fleet.actors) and time.monotonic()<deadline:
                if time.monotonic()-fleet.last_meter>=60:
                    try:fleet.meter()
                    except Exception:pass
                for actor in fleet.actors:
                    try:fleet.step(actor)
                    except Exception:pass
                time.sleep(.5)
        finally:
            for actor in fleet.actors:
                if actor.command:actor.command.terminate()
                if actor.phase!='done':budget.finish_child(actor.session,False)
            try:fleet.meter()
            except BaseException:pass
            clean=budget.finish()
            state.update(cleanup_verified=clean,status='failed');fleet.save()
            for s,handler in previous.items():signal.signal(s,handler)
        print(json.dumps({'fleet':fleet_name,'cleanup_verified':clean}),flush=True)
        if not clean:
            for child in budget.active['children'].values():
                if child['status']!='stopped':print('Cleanup failed. Session: '+child['session']+'. Recovery: '+
                    ' '.join(cli.command+['stop','-s',child['session']]),flush=True)
            raise RuntimeError('fleet recovery cleanup incomplete; recorded owned IDs remain unresolved')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=Path)
    parser.add_argument('--recover',help='Cleanup an exact recorded fleet ID; never starts training')
    parser.add_argument('--concurrency',type=int,default=1)
    parser.add_argument('--cli-python',required=True)
    parser.add_argument('--state-output',type=Path,required=True)
    parser.add_argument('--study-unit-limit',type=float)
    parser.add_argument('--study-start-units',type=float)
    args=parser.parse_args();root=Path.cwd().resolve()
    cli=CLI(root,args.cli_python)
    if args.recover:
        if args.jobs:parser.error('--recover cannot combine --jobs')
        if args.study_unit_limit is not None or args.study_start_units is not None:parser.error('recovery does not accept study budget overrides')
        recover(root,cli,args.recover,args.state_output)
    else:
        if not args.jobs:parser.error('--jobs is required unless recovering')
        Fleet(root,cli,json.loads(args.jobs.read_text()),args.concurrency,args.state_output,
              study_limit=args.study_unit_limit,study_start=args.study_start_units).run()


if __name__=='__main__':main()
