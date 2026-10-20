"""Authenticated absence, not a timeout alone, permits one existing-session retry."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from rl_stocks.runtime.colab import fleet


def controller():
    c=fleet.Fleet.__new__(fleet.Fleet);c.root=Path('/tmp');c.fatal=None
    c.budget=SimpleNamespace(active={'children':{'owned':dict(exec_cli_failed=True,deadline=5000)}})
    c.snapshot=Mock();c.execute=Mock();c.download=Mock()
    a=fleet.Runtime(dict(plan=dict(jobs=[dict(steps=1024)])),'owned',phase='await_export',exec_deadline=3000,probe_due=0.)
    return c,a


def test_unknown_then_verified_absence_retries_once_and_preserves_deadline(monkeypatch):
    c,a=controller();monkeypatch.setattr(fleet.time,'monotonic',lambda:1000.)
    probe=Mock(side_effect=[False,True]);monkeypatch.setattr(fleet,'worker_not_started',probe)
    c.step(a);c.execute.assert_not_called();assert a.exec_retries==0
    c.step(a);c.execute.assert_called_once_with(a,preserve_deadline=True)
    assert a.exec_retries==1 and a.exec_deadline==3000
    c.step(a);assert c.execute.call_count==1 and probe.call_count==2
    assert c.budget.active['children']['owned']['deadline']==5000


def test_successful_exec_or_expired_deadline_never_triggers_retry(monkeypatch):
    c,a=controller();monkeypatch.setattr(fleet.time,'monotonic',lambda:1000.)
    probe=Mock(return_value=True);monkeypatch.setattr(fleet,'worker_not_started',probe)
    c.budget.active['children']['owned']['exec_cli_failed']=False
    c.step(a);probe.assert_not_called();c.execute.assert_not_called()
    a.exec_deadline=1010
    with pytest.raises(TimeoutError):c.step(a)
    probe.assert_not_called()


def test_retry_command_cannot_extend_original_execution_deadline(monkeypatch):
    c,a=controller();c.budget.active['children']['owned']['deadline']=10000
    monkeypatch.setattr(fleet.time,'time',lambda:1000.)
    monkeypatch.setattr(fleet.time,'monotonic',lambda:1000.)
    c.launch=Mock();c.execute=fleet.Fleet.execute.__get__(c)
    c.execute(a,preserve_deadline=True)
    assert a.exec_deadline==2800  # Earlier of existing3000 and smoke1800s window.
    assert c.launch.call_args.args[-1]==1800
    a.exec_deadline=1010
    with pytest.raises(TimeoutError):c.execute(a,preserve_deadline=True)
