"""One owned Colab runtime at a time. Metering failures always enter cleanup."""
import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from rl_stocks.runtime.colab.budget import BudgetStop,Ledger,parse_usage
from rl_stocks.runtime.colab.jobs import job_plan
from rl_stocks.core.provenance import repository_state
from rl_stocks.runtime.colab.artifacts import validate_archive

class ColabJob:
    def __init__(self,root,cli_python,accelerator="A100"):
        self.root=Path(root).resolve()
        self.command=[str(cli_python),str(self.root/"scripts/colab_safe.py")]
        self.state=self.root/".runtime";self.state.mkdir(exist_ok=True)
        self.ledger=Ledger(self.state/"usage-ledger.json")
        if accelerator not in ("CPU","T4","A100"):raise ValueError("unsupported runtime")
        self.accelerator=accelerator
        self.session="rl-stocks-"+accelerator.lower()+"-"+uuid.uuid4().hex[:12]
        self.last_meter=0.;self.allocated=False
    def short(self,args,timeout=25):
        p=subprocess.run(self.command+args,cwd=self.root,capture_output=True,text=True,timeout=timeout,stdin=subprocess.DEVNULL)
        if p.returncode:raise RuntimeError("Colab command failed: "+args[0])
        return p.stdout
    def meter(self):return parse_usage(self.short(["usage"]))
    def tick(self):
        self.ledger.guard(60)
        if time.monotonic()-self.last_meter>=60:
            poll_started=time.monotonic()
            try:self.ledger.observe(self.meter())
            except Exception as exc:
                self.ledger.value["budget_stopped"]=True
                self.ledger.active["metering_unreliable"]=True
                self.ledger.save()
                raise BudgetStop("metering unavailable; teardown required") from exc
            self.last_meter=poll_started
            if self.ledger.active["meter_readings"][-1]["assignments"]>1:
                raise BudgetStop("another account assignment prevents isolated metering")
            if self.allocated and getattr(self,"accelerator","A100")!="CPU" and self.ledger.active["rate"]<=0:
                raise BudgetStop("allocated runtime has no measurable consumption rate")
            self.ledger.guard(60)
    def operation(self,args,timeout):
        self.ledger.guard(timeout)
        logs=self.state/self.session;logs.mkdir(exist_ok=True)
        log=logs/(args[0]+"-output.txt")
        self.ledger.active["operation"]=args[0];self.ledger.save()
        # CLI output may include session identifiers; files stay private, not printed.
        with log.open("w") as output:
            os.chmod(log,0o600)
            process=subprocess.Popen(self.command+args,cwd=self.root,stdout=output,stderr=output,stdin=subprocess.DEVNULL,start_new_session=True)
            start=time.monotonic()
            try:
                while process.poll() is None:
                    self.tick()
                    if args[0]=="exec" and self.last_meter!=getattr(self,"last_snapshot_meter",None):
                        self.last_snapshot_meter=self.last_meter
                        self.ledger.guard(25)
                        destination=self.root/"artifacts"/"colab"/self.session
                        destination.mkdir(parents=True,exist_ok=True)
                        checkpoint=destination/"checkpoint.tar.gz"
                        try:
                            self.short(["download","-s",self.session,"/content/research-checkpoint.tar.gz",str(checkpoint)],20)
                            if str(checkpoint) not in self.ledger.active["retrieved"]:
                                self.ledger.active["retrieved"].append(str(checkpoint));self.ledger.save()
                        except Exception:pass  # Remote first checkpoint may not exist yet.
                    if time.monotonic()-start>timeout:raise TimeoutError(args[0])
                    time.sleep(.5)
                if process.returncode:raise RuntimeError("Colab operation failed: "+args[0])
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
    def retrieve(self,remote,local):
        for attempt in range(3):
            try:
                self.operation(["download","-s",self.session,remote,str(local)],30)
                self.ledger.active["retrieved"].append(str(local));self.ledger.save()
                return True
            except BudgetStop:raise
            except Exception:continue
        return False

    def execute_worker(self):
        try:
            self.operation(["exec","-s",self.session,"-f",str(self.root/"scripts/remote_smoke.py")],1800)
        except BudgetStop:
            raise
        except RuntimeError:
            # A completed remote job may outlive a failed CLI output stream.
            # The downloaded manifest and checksums still decide acceptance.
            self.ledger.active["exec_cli_failed"]=True;self.ledger.save()
    def cleanup(self):
        verified=False
        try:
            raw=json.loads((self.state/"sessions.json").read_text())
            self.ledger.active["endpoint"]=raw.get(self.session,{}).get("endpoint") or self.ledger.active.get("endpoint")
            self.ledger.save()
        except (AttributeError,OSError,ValueError):pass
        for _ in range(3):
            try:
                self.short(["stop","-s",self.session],30)
                listing=self.short(["sessions"],25)
                endpoint=self.ledger.active.get("endpoint")
                valid_listing="No active sessions found" in listing or "Hardware:" in listing
                unknown_assignment=not endpoint and "Hardware:" in listing
                if valid_listing and not unknown_assignment and f'[{self.session}]' not in listing and (not endpoint or endpoint not in listing):
                    verified=True;break
            except Exception:continue
        try:self.ledger.observe(self.meter())
        except Exception:pass
        self.ledger.finish(verified)
        if not verified:
            recovery=" ".join(self.command+["stop","-s",self.session])
            print(f"Cleanup failed. Session: {self.session}. Recovery: {recovery}",file=sys.stderr)
        return verified
    def run(self,dataset,research_algorithm=None):
        if self.accelerator=="A100" and research_algorithm is None:
            raise ValueError("A100 requires a full --research job; use CPU/T4 for preflight")
        plan=job_plan(research_algorithm,self.accelerator)
        # Help and version are checked before any metered allocation.
        version=self.short(["version"])
        for cmd in ("usage","new","install","exec","upload","download","stop","sessions"):
            self.short([cmd,"--help"])
        if not (self.root/"requirements.lock").exists():
            raise RuntimeError("transitive requirements.lock missing; resolve and verify dependencies before provisioning")
        usage=self.meter()
        self.ledger.begin(self.session,usage)
        self.ledger.active["cli_version"]=version.strip();self.ledger.save()
        self.ledger.active["accelerator"]=self.accelerator;self.ledger.save()
        self.last_meter=time.monotonic()
        # Persist the intended unique name before provisioning so interruption has
        # a recovery command even if new succeeds but its response is lost.
        def interrupted(signum,frame):raise KeyboardInterrupt()
        previous={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
        ok=False
        try:
            allocation=["new","-s",self.session]
            if self.accelerator!="CPU":allocation.extend(["--gpu",self.accelerator])
            self.operation(allocation,180)
            self.allocated=True
            raw=json.loads((self.state/"sessions.json").read_text())
            # CLI versions differ in state shape; endpoint is optional, never token.
            def find(v):
                if isinstance(v,dict):
                    if v.get("name")==self.session:return v.get("endpoint")
                    for x in v.values():
                        r=find(x)
                        if r:return r
                if isinstance(v,list):
                    for x in v:
                        r=find(x)
                        if r:return r
            self.ledger.active["endpoint"]=find(raw);self.ledger.active["status"]="active";self.ledger.save()
            try:self.ledger.observe(self.meter())
            except Exception as exc:
                self.ledger.active["metering_unreliable"]=True
                raise BudgetStop("post-allocation metering unavailable") from exc
            self.last_meter=time.monotonic()
            if self.accelerator!="CPU" and self.ledger.active["rate"]<=0:raise BudgetStop("consumption rate unavailable")
            status=self.short(["status","-s",self.session])
            if self.accelerator not in status.upper():raise RuntimeError("runtime allocation not verified")
            archive=self.state/"research-upload.tar.gz"
            job_file=self.state/"job-plan.json"
            job_file.write_text(json.dumps(plan,indent=2))
            git_file=self.state/"source-git.json"
            git_file.write_text(json.dumps(repository_state(self.root),indent=2))
            self.ledger.active["job_plan"]=plan;self.ledger.save()
            with tarfile.open(archive,"w:gz") as tar:
                for item in ("src","scripts","pyproject.toml","requirements.txt","requirements.lock","configs"):
                    tar.add(self.root/item,arcname=item)
                tar.add(Path(dataset)/"dataset.json",arcname="dataset.json")
                tar.add(job_file,arcname="job-plan.json")
                tar.add(git_file,arcname="source-git.json")
            self.operation(["upload","-s",self.session,str(archive),"/content/research-upload.tar.gz"],120)
            # Use Colab's Python; isolate dependencies without changing interpreter.
            self.operation(["install","-s",self.session,"uv==0.13.1"],180)
            self.execute_worker()
            output=self.root/"artifacts"/"colab"/self.session;output.mkdir(parents=True,exist_ok=True)
            ok=self.retrieve("/content/research-artifacts.tar.gz",output/"artifacts.tar.gz")
            if not ok:raise RuntimeError("artifact download exhausted retries")
            runs=validate_archive(output/"artifacts.tar.gz",plan,Path(dataset).name)
            self.ledger.active["training_verified"]=True
            self.ledger.active["runs"]=[{k:r[k] for k in ("algorithm","device","smoke","actual_steps","bundle_id")} for r in runs]
            self.ledger.save()
        except BudgetStop:
            self.ledger.value["budget_stopped"]=True;self.ledger.save()
            raise
        finally:
            # A second handled signal must not skip bounded retrieval/teardown.
            for s in previous:signal.signal(s,signal.SIG_IGN)
            # On remote failure the bootstrap's finally block still exports artifacts.
            # Retrieval is attempted only while metering and working budget allow it.
            if not ok and self.allocated:
                try:
                    output=self.root/"artifacts"/"colab"/self.session;output.mkdir(parents=True,exist_ok=True)
                    self.retrieve("/content/research-artifacts.tar.gz",output/"partial-artifacts.tar.gz")
                except Exception:pass
            clean=self.cleanup()
            for s,handler in previous.items():signal.signal(s,handler)
            print(json.dumps({"session":self.session,"estimated_units":self.ledger.value["consumed"],
                "remaining_project_budget":150-self.ledger.value["consumed"],
                "measured_attempt":self.ledger.active["measured"],"retrieved":self.ledger.active["retrieved"],
                "metering_unreliable":self.ledger.active.get("metering_unreliable",False),
                "training_verified":self.ledger.active.get("training_verified",False),
                "cleanup_verified":clean}))
        if not clean:raise RuntimeError("cleanup not verified")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset");parser.add_argument("--cli-python",required=True)
    parser.add_argument("--research",choices=("ppo","sac","td3"),
                        help="After smoke acceptance, train one 50,000-transition seed-11 model")
    parser.add_argument("--accelerator",choices=("CPU","T4","A100"),
                        help="Default CPU for preflight; A100 for --research")
    args=parser.parse_args()
    root=Path.cwd();state=root/".runtime";state.mkdir(exist_ok=True)
    with (state/"allocation.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        accelerator=args.accelerator or ("A100" if args.research else "CPU")
        ColabJob(root,args.cli_python,accelerator).run(args.dataset,args.research)

if __name__=="__main__":main()
