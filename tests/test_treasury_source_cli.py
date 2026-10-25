"""Handled source failure persists evidence, releases lock and forbids restart."""

import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import signal

import pytest


def test_failed_collector_cleanup_and_no_automatic_restart(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "treasury_source_cli", root / "scripts/audit_treasury_source.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = tmp_path / "configs/experiments/treasury_source_audit.json"
    protocol.parent.mkdir(parents=True)
    shutil.copyfile(root / "configs/experiments/treasury_source_audit.json", protocol)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "JOURNAL", tmp_path / ".runtime/treasury-source-progress.json")

    def fail(spec, started):
        module.save(dict(status="source_request", year=2009, requests_started=1))
        raise ValueError("failed fixture")

    monkeypatch.setattr(module, "run", fail)
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 1 and signal.getsignal(signal.SIGALRM) == previous
    failure = json.loads((tmp_path / "results/treasury-source-2026-10-25/failure.json").read_text())
    assert failure["status"] == "failed" and failure["requests_started"] == 1
    assert not failure["automatic_retry"] and not failure["colab_allocated"]
    with (tmp_path / ".runtime/treasury-source.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(ValueError, match="no automatic restart"):
        module.main()
    assert not list(tmp_path.rglob("*.tmp"))
