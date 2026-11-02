import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_canonical_actor import fixture
from rl_stocks.inference.canonical_actor import ROLES
from rl_stocks.inference.canonical_probe import execute, verify
from rl_stocks.runtime.colab.budget import BudgetStop, Usage
from rl_stocks.runtime.colab.reference_probe import ReferenceProbeJob, worker_script


@pytest.fixture(scope='module')
def probe():
    actors = {role: fixture(role) for role in ROLES}
    expected = execute(actors)
    expected['native_packages_imported'] = []
    return actors, expected


def test_worker_uses_isolated_stdlib_process(tmp_path, probe):
    actors, expected = probe
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'probe.json'
    script = tmp_path / 'worker.py'
    script.write_text(worker_script(root, actors, str(output)))
    subprocess.run([sys.executable, str(script)], check=True, timeout=30, capture_output=True)
    actual = json.loads(output.read_text())
    assert verify(actual, expected)
    assert actual['native_packages_imported'] == []
    for role in ROLES:
        assert actual['roles'][role]['count'] == (128 if role == 'masked_shared' else 256)


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(python_major_minor=[3, 14]),
    lambda r: r.update(native_packages_imported=['torch']),
    lambda r: r.update(training_performed=True),
    lambda r: r['roles']['tagged_shared']['output_bits'][0].__setitem__(0, 1234),
    lambda r: r['roles']['masked_shared'].update(elapsed_seconds=float('nan')),
])
def test_probe_rejects_different_bits_runtime_or_latency(probe, mutation):
    _, expected = probe
    actual = json.loads(json.dumps(expected))
    mutation(actual)
    with pytest.raises(ValueError):
        verify(actual, expected)


class FakeProbe(ReferenceProbeJob):
    def __init__(self, root, expected, fail=None):
        super().__init__(root, sys.executable, 'CPU')
        self.expected, self.fail, self.calls = expected, fail, []
        self.remote_active = False
        self.stops = 0
        self.ledger.value['minimum_account_balance'] = 150
        self.ledger.save()

    def short(self, args, timeout=25):
        if args == ['version']:
            return 'Version 0.7.4'
        if args[-1:] == ['--help']:
            return 'help'
        if args[0] == 'status':
            return 'Hardware: CPU'
        if args[0] == 'stop':
            if self.fail == 'cleanup':
                raise RuntimeError('stop failure')
            self.stops += 1
            self.remote_active = False
            return ''
        if args[0] == 'sessions':
            return 'No active sessions found'
        raise AssertionError(args)

    def meter(self):
        if self.fail == 'meter' and self.remote_active:
            raise RuntimeError('meter unavailable')
        return Usage(548., .08 if self.remote_active else 0., int(self.remote_active))

    def operation(self, args, timeout):
        self.calls.append(args[0])
        self.tick()
        if args[0] == 'new':
            self.remote_active = True
        elif args[0] == 'exec':
            self.exec_arguments = tuple(args)
            if self.fail == 'interrupt':
                raise KeyboardInterrupt()
            if self.fail == 'exec':
                raise RuntimeError('execution failure')
            if self.fail == 'budget':
                self.ledger.active['base_consumed'] = self.ledger.value['consumed'] - .1
                self.tick()
        elif args[0] == 'download':
            if self.fail == 'download':
                raise RuntimeError('download failure')
            Path(args[-1]).write_text(json.dumps(self.expected))
        else:
            raise AssertionError(args)


@pytest.mark.parametrize('failure', [None, 'exec', 'meter', 'download', 'budget', 'interrupt'])
def test_owned_cleanup_success_failure_and_no_retry(tmp_path, probe, monkeypatch, failure):
    from rl_stocks.runtime.colab import reference_probe
    actors, expected = probe
    monkeypatch.setattr(reference_probe, 'repository_state', lambda p: {'dirty': '', 'revision': 'a'*40})
    monkeypatch.setattr(reference_probe, 'worker_script', lambda *a: '# no remote code in lifecycle fixture')
    job = FakeProbe(tmp_path, expected, failure)
    claim = tmp_path / '.runtime/claim.json'
    if failure:
        with pytest.raises((RuntimeError, BudgetStop, KeyboardInterrupt)):
            job.run_probe(actors, expected, claim)
    else:
        assert verify(job.run_probe(actors, expected, claim), expected)
    assert not job.remote_active
    assert job.stops == 1
    assert job.ledger.active['status'] == 'stopped'
    assert json.loads(claim.read_text())['session'] == job.session
    assert job.calls.count('new') == 1
    if 'exec' in job.calls:
        arguments = job.exec_arguments
        assert float(arguments[arguments.index('--timeout')+1]) == job.execution_seconds
    with pytest.raises(FileExistsError):
        FakeProbe(tmp_path, expected).run_probe(actors, expected, claim)


def test_low_balance_prevents_allocation(tmp_path, probe, monkeypatch):
    from rl_stocks.runtime.colab import reference_probe
    actors, expected = probe
    monkeypatch.setattr(reference_probe, 'repository_state', lambda p: {'dirty': '', 'revision': 'a'*40})
    monkeypatch.setattr(reference_probe, 'worker_script', lambda *a: '# no remote code')
    job = FakeProbe(tmp_path, expected)
    monkeypatch.setattr(job, 'meter', lambda: Usage(170., 0., 0))
    with pytest.raises(BudgetStop):
        job.run_probe(actors, expected, tmp_path / '.runtime/claim.json')
    assert 'new' not in job.calls


def test_cleanup_failure_retains_session_and_recovery(tmp_path, probe, monkeypatch, capsys):
    from rl_stocks.runtime.colab import reference_probe
    actors, expected = probe
    monkeypatch.setattr(reference_probe, 'repository_state', lambda p: {'dirty': '', 'revision': 'a'*40})
    monkeypatch.setattr(reference_probe, 'worker_script', lambda *a: '# no remote code')
    job = FakeProbe(tmp_path, expected, 'cleanup')
    with pytest.raises(RuntimeError, match='cleanup not verified'):
        job.run_probe(actors, expected, tmp_path / '.runtime/claim.json')
    assert job.ledger.active['status'] == 'cleanup_failed'
    assert job.ledger.active['session'] == job.session
    assert job.remote_active
    error = capsys.readouterr().err
    assert job.session in error and 'stop -s' in error


def test_projected_operation_reserves_teardown_before_launch(tmp_path, probe):
    _, expected = probe
    job = FakeProbe(tmp_path, expected)
    job.ledger.begin(job.session, Usage(548., 0., 0))
    job.ledger.observe(Usage(548., .3, 1))
    # Current six-minute reserve fits, but the next long operation plus teardown
    # does not fit the fixed allowance; no remote operation may start.
    job.guard_allowance(360)
    with pytest.raises(BudgetStop, match='allowance'):
        job.guard_allowance(1260)
    assert job.calls == []
    assert job.ledger.value['budget_stopped'] is True
