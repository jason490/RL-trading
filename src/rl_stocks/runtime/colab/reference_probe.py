"""Single owned CPU probe with original shared ledger and mandatory cleanup.

This is inference-only, never a retry of the stopped native training study.
"""
import argparse
import base64
import fcntl
import io
import json
from pathlib import Path
import signal
import time
import zipfile

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.provenance import repository_state
from rl_stocks.inference.canonical_actor import CanonicalActor, ROLES
from rl_stocks.inference.canonical_probe import verify
from rl_stocks.runtime.colab.budget import BudgetStop
from rl_stocks.runtime.colab.runner import ColabJob

LIMIT_UNITS = .05
DEADLINE = 1200
RESERVE = 360
FILES = ('rl_stocks/__init__.py', 'rl_stocks/core/__init__.py',
         'rl_stocks/core/contracts.py', 'rl_stocks/inference/__init__.py',
         'rl_stocks/inference/canonical_math.py', 'rl_stocks/inference/canonical_actor.py',
         'rl_stocks/inference/canonical_probe.py')


def worker_script(root, actors, remote_output):
    root = Path(root)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
        for name in FILES:
            stream.writestr(name, (root / 'src' / name).read_bytes())
    encoded = base64.b64encode(archive.getvalue()).decode()
    # Execute a fresh isolated process so existing notebook imports cannot select
    # kernels or cause native-package import checks to misreport this worker.
    code = f'''import base64, json, os, pathlib, subprocess, sys, tempfile
with tempfile.TemporaryDirectory(prefix="reference-probe-") as folder:
    folder = pathlib.Path(folder)
    archive = folder / "source.zip"
    archive.write_bytes(base64.b64decode({encoded!r}))
    actors = {actors!r}
    (folder / "actors.json").write_text(json.dumps(actors))
    program = "import json,sys; sys.path.insert(0,sys.argv[1]); from rl_stocks.inference.canonical_probe import execute; assert sys.version_info[:2]==(3,13), 'Inspect default Colab Python before updating project'; result=execute(json.load(open(sys.argv[2]))); open(sys.argv[3],'w').write(json.dumps(result,sort_keys=True,allow_nan=False))"
    destination = pathlib.Path({remote_output!r})
    assert not destination.exists(), "fresh probe output required"
    subprocess.run([sys.executable, "-I", "-c", program, str(archive), str(folder / "actors.json"), str(destination)], check=True, timeout=180)
print("Reference inference probe exported")
'''
    return code


class ReferenceProbeJob(ColabJob):
    def tick(self):
        super().tick()
        self.last_snapshot_meter = self.last_meter  # No training checkpoints exist in this probe.
        active = self.ledger.active
        projected = self.ledger.value['consumed'] - active['base_consumed'] + (
            self.ledger.clock() - active['last_time'] + RESERVE) * (active['rate'] + .01)/3600
        if projected >= LIMIT_UNITS:
            self.ledger.value['budget_stopped'] = True
            self.ledger.save()
            raise BudgetStop('reference probe allowance exhausted; no automatic restart')
        self.ledger.guard(RESERVE)

    def run_probe(self, actors, expected, claim):
        if set(actors) != set(ROLES):
            raise ValueError('all three role actors required')
        verify(expected, expected)
        for role in ROLES:
            CanonicalActor(actors[role], expected_role=role)
            if expected['roles'][role]['actor_sha256'] != actors[role]['sha256']:
                raise ValueError('host reference actor identity mismatch')
        git = repository_state(self.root)
        if git['dirty']:
            raise ValueError('clean committed probe source required')
        if self.ledger.value.get('minimum_account_balance', 0) < 150:
            raise BudgetStop('persisted account floor150 required')
        for command in ('usage', 'new', 'exec', 'download', 'stop', 'sessions', 'status'):
            self.short([command, '--help'])
        version = self.short(['version'])
        usage = self.meter()
        # One immutable claim across failures and successful calls; never retry an
        # allocation by restarting the controller. The shared allocation lock is
        # held by main(), and begin() rejects any existing account assignments.
        claim = Path(claim)
        payload = {'session': self.session, 'actors_sha256': digest(actors),
                   'source': git, 'limit_units': LIMIT_UNITS, 'deadline_seconds': DEADLINE,
                   'training': False, 'automatic_retry': False}
        with claim.open('x') as stream:
            json.dump(payload, stream, sort_keys=True)
        output = self.root / 'artifacts' / 'colab' / self.session
        output.mkdir(parents=True, exist_ok=False)
        remote = '/content/' + self.session + '-reference.json'
        script = output / 'probe.py'
        script.write_text(worker_script(self.root, actors, remote))
        retrieved = output / 'reference-probe.json'
        self.ledger.begin(self.session, usage)
        active = self.ledger.active
        active.update(accelerator='CPU', kind='canonical-inference-probe',
                      deadline=min(active['deadline'], self.ledger.clock()+DEADLINE),
                      cli_version=version.strip(), protocol=payload)
        self.ledger.save()
        self.last_meter = time.monotonic()
        result = None

        def interrupted(signum, frame):
            raise KeyboardInterrupt()

        previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            self.operation(['new', '-s', self.session], 180)
            self.allocated = True
            active['status'] = 'active'
            self.ledger.save()
            try:
                self.ledger.observe(self.meter())
            except Exception as error:
                active['metering_unreliable'] = True
                self.ledger.value['budget_stopped'] = True
                self.ledger.save()
                raise BudgetStop('post-allocation metering unavailable; teardown required') from error
            self.last_meter = time.monotonic()
            self.tick()
            if 'CPU' not in self.short(['status', '-s', self.session]).upper():
                raise RuntimeError('owned CPU allocation not verified')
            self.operation(['exec', '-s', self.session, '-f', str(script)], 240)
            if not self.retrieve(remote, retrieved):
                raise RuntimeError('bounded reference artifact retrieval exhausted')
            result = json.loads(retrieved.read_text())
            verify(result, expected)
            active['reference_verified'] = True
            self.ledger.save()
        except BudgetStop:
            self.ledger.value['budget_stopped'] = True
            self.ledger.save()
            raise
        finally:
            for s in previous:
                signal.signal(s, signal.SIG_IGN)
            try:
                if self.allocated and result is None:
                    try:
                        self.retrieve(remote, retrieved)
                    except Exception:
                        pass  # Budget/metering failure must proceed directly to owned cleanup.
                clean = self.cleanup()
                summary = {'session': self.session, 'cleanup_verified': clean,
                           'reference_verified': active.get('reference_verified', False),
                           'project_units': self.ledger.value['consumed'],
                           'measured_units': active['measured'], 'estimated_units': active['estimated'],
                           'remaining_project_units': 150-self.ledger.value['consumed'],
                           'retrieved': active['retrieved'], 'training': False}
                (output / 'summary.json').write_bytes(canonical(summary))
                print(json.dumps(summary))
            finally:
                for s, handler in previous.items():
                    signal.signal(s, handler)
        if not clean:
            raise RuntimeError('owned reference probe cleanup not verified')
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='JSON containing actors and host expected probe')
    parser.add_argument('--cli-python', required=True)
    args = parser.parse_args()
    root = Path.cwd()
    payload = json.loads(Path(args.input).read_text())
    with (root / '.runtime/allocation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ReferenceProbeJob(root, args.cli_python, 'CPU').run_probe(
            payload['actors'], payload['expected'], root / '.runtime/canonical-reference-probe-v1-claim.json')


if __name__ == '__main__':
    main()
