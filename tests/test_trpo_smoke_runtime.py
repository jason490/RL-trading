import json
from pathlib import Path
import tarfile

import pytest

from rl_stocks.runtime.colab import trpo_smoke as runtime
from rl_stocks.runtime.colab.budget import BudgetStop, Ledger, Usage


def fixed():
    root = Path(__file__).resolve().parents[1]
    return runtime.runtime_spec(root)


class FixtureJob(runtime.TrpoSmokeJob):
    def __init__(self, root, *, fail=None):
        self.root = root
        self.state = root / ".runtime"
        self.state.mkdir()
        self.ledger = Ledger(self.state / "usage-ledger.json")
        self.ledger.value["minimum_account_balance"] = 150
        self.ledger.save()
        self.session, self.accelerator = "owned-fixture", "CPU"
        self.allocated = False
        self.fail = fail
        self.calls = []
        self.cleaned = False
        self.meters = 0

    def short(self, args, timeout=25):
        return "CPU" if args[0] == "status" else "fixture"

    def meter(self):
        self.meters += 1
        if self.fail == "meter" and self.meters > 1:
            raise RuntimeError("fixture metering failure")
        return Usage(500, 0, 0 if self.meters == 1 else 1)

    def tick(self):
        self.guard_allowance(900)

    def operation(self, args, timeout):
        self.calls.append(args)
        if self.fail == args[0]:
            raise RuntimeError("fixture operation failure")
        if self.fail == "budget" and args[0] == "upload":
            self.ledger.value["budget_stopped"] = True
            raise BudgetStop("fixture budget stop")

    def retrieve(self, remote, local):
        if "kernel" in remote:
            local.write_text(json.dumps({"python": [3, 13, 18]}))
        else:
            local.write_text("fixture transport, not a real training artifact")
        self.ledger.active["retrieved"].append(str(local))
        return True

    def cleanup(self):
        self.cleaned = True
        self.ledger.finish(True)
        return True


def setup(tmp_path, monkeypatch, fail=None):
    spec = fixed()
    monkeypatch.setattr(runtime, "runtime_spec", lambda root: spec)
    monkeypatch.setattr(runtime, "validate_dataset", lambda data: data)
    monkeypatch.setattr(runtime, "repository_state", lambda root: {"revision": "fixture", "dirty": ""})
    monkeypatch.setattr(runtime, "source_identity", lambda: "fixture-source")
    def prepare(root, dataset, output, source):
        output.write_text("fixture upload")
        return output
    monkeypatch.setattr(runtime, "prepare_upload", prepare)
    job = FixtureJob(tmp_path, fail=fail)
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"fixture":true}')
    return job, dataset


@pytest.mark.parametrize("failure", ["new", "exec", "meter", "budget"])
def test_failures_always_stop_owned_session_and_never_retry(tmp_path, monkeypatch, failure):
    job, dataset = setup(tmp_path, monkeypatch, failure)
    with pytest.raises((RuntimeError, BudgetStop)):
        job.run_smoke(dataset)
    assert job.cleaned and job.ledger.active["status"] == "stopped"
    assert sum(args[0] == "new" for args in job.calls) == 1
    before = list(job.calls)
    with pytest.raises(ValueError, match="already claimed"):
        job.run_smoke(dataset)
    assert job.calls == before


def test_acceptance_after_verified_cleanup(tmp_path, monkeypatch):
    job, dataset = setup(tmp_path, monkeypatch)
    def accept(*a, **kw):
        assert job.cleaned and job.ledger.active["status"] == "stopped"
        return {"fixture_accepted": True}
    monkeypatch.setattr(runtime, "accept", accept)
    assert job.run_smoke(dataset)["fixture_accepted"]
    assert job.ledger.active["training_verified"]
    summary = json.loads((job.output / "summary.json").read_text())
    assert summary["cleanup_verified"] and summary["training_verified"]


def test_offline_rejection_retains_transport_and_stopped_session(tmp_path, monkeypatch):
    job, dataset = setup(tmp_path, monkeypatch)
    def reject(*a, **kw):
        assert job.cleaned
        raise ValueError("fixture physical rejection")
    monkeypatch.setattr(runtime, "accept", reject)
    with pytest.raises(ValueError, match="physical rejection"):
        job.run_smoke(dataset)
    assert (job.output / "artifacts.tar.gz").exists()
    assert not job.ledger.active.get("training_verified", False)
    assert job.ledger.active["status"] == "stopped"


def test_shared_and_incremental_budgets_preserved(tmp_path):
    job = FixtureJob(tmp_path)
    job.spec = fixed()
    job.ledger.value["consumed"] = 40
    job.ledger.begin(job.session, Usage(500, 0, 0))
    job.guard_allowance(100)
    job.ledger.value["consumed"] = 40.25
    with pytest.raises(BudgetStop, match="allowance"):
        job.guard_allowance(100)
    assert job.ledger.value["budget_stopped"]
    assert job.ledger.active["base_consumed"] == 40
    assert job.ledger.value["minimum_account_balance"] == 150


def test_upload_excludes_credentials_and_python_caches(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for directory in ("src/pkg", "src/pkg/__pycache__", "scripts", "configs"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for name in ("src/pkg/a.py", "src/pkg/__pycache__/a.pyc", "scripts/worker.py", "configs/spec.json",
                 "pyproject.toml", "requirements.txt", "requirements.lock", ".env"):
        (root / name).write_text("fixture")
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}")
    out = tmp_path / "upload.tar.gz"
    runtime.prepare_upload(root, dataset, out, {"revision": "fixture", "dirty": ""})
    with tarfile.open(out) as archive:
        names = archive.getnames()
    assert ".env" not in names and not any("__pycache__" in name for name in names)
    assert "src/pkg/a.py" in names and "source-git.json" in names


def test_reviewed_attempt_counts_prior_charges_in_original_allowance(tmp_path):
    job = FixtureJob(tmp_path)
    job.spec = fixed()
    job.study_base = 40
    job.ledger.value["consumed"] = 40.249
    job.ledger.begin(job.session, Usage(500, 0, 0))
    with pytest.raises(BudgetStop, match="allowance"):
        job.guard_allowance(900)
    assert job.ledger.active["base_consumed"] == 40.249
    assert job.study_base == 40
