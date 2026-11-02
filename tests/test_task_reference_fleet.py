import json
from pathlib import Path
from unittest.mock import patch

import pytest

from test_fleet import FakeCLI, FakeCommand
from rl_stocks.runtime.colab.fleet import Fleet
from rl_stocks.runtime.colab.jobs import task_reference_candidate_plan


class CandidateCLI(FakeCLI):
    def __init__(self, root, kernel=(3, 13, 18), worker_error=False, **kwargs):
        super().__init__(root, **kwargs)
        self.kernel = kernel
        self.worker_error = worker_error

    def short(self, args, timeout=25):
        if args == ["usage"]:
            self.meters += 1
            if self.fail_meter and self.meters >= 4:
                raise RuntimeError("meter unavailable")
            return f"Current balance: 548 compute units\nUsage rate: {len(self.active) * 0.08}/hr\nActive assignments: {len(self.active)}\n"
        return super().short(args, timeout)

    def launch(self, args, path, timeout):
        if args[0] == "exec" and Path(args[-1]).name == "check_colab_kernel.py":
            self.calls.append(args)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("kernel inspected")
            return FakeCommand(path)
        if args[0] == "download" and args[-2].endswith("task-reference-kernel.json"):
            self.calls.append(args)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
            target = Path(args[-1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps({"python": list(self.kernel), "supported": self.kernel[:2] == (3, 13)})
            )
            return FakeCommand(path)
        result = super().launch(args, path, timeout)
        if args[0] == "exec" and self.worker_error:
            result.code = 1
        return result


def setup(tmp_path, monkeypatch, **options):
    root = tmp_path
    for name in (".runtime", "scripts", "configs", "src", "dataset"):
        (root / name).mkdir()
    (root / "requirements.lock").write_text("fixture")
    (root / "dataset/dataset.json").write_text("{}")
    from rl_stocks.training import task_reference_runtime, task_runtime

    monkeypatch.setattr(task_reference_runtime, "admit", lambda *a: {"execution_admitted": True})
    monkeypatch.setattr(task_runtime, "shared_budget", lambda *a: (2.0, 0.0))
    monkeypatch.setattr(
        "rl_stocks.runtime.colab.fleet.validate_tasks",
        lambda tasks, root: [{**t, "dataset": str(root / "dataset")} for t in tasks],
    )
    monkeypatch.setattr(
        "rl_stocks.runtime.colab.fleet.repository_state",
        lambda root: {"dirty": "", "revision": "a" * 40},
    )

    def payload(root, dataset, plan, session):
        path = root / ".runtime" / f"{session}-payload.tar.gz"
        path.write_bytes(b"fixture")
        return path

    monkeypatch.setattr("rl_stocks.runtime.colab.fleet.prepare_payload", payload)

    def extract(archive, plan, dataset, destination):
        destination.mkdir(parents=True)
        return {
            "python": [3, 13, 18],
            "source_git": {"dirty": "", "revision": "a" * 40},
            "source_hash": "b" * 64,
            "plan": plan,
        }

    monkeypatch.setattr(
        "rl_stocks.runtime.colab.task_screen_artifacts.extract_screen_transport", extract
    )
    monkeypatch.setattr("rl_stocks.runtime.colab.fleet.worker_not_started", lambda *a: False)
    tasks = [{"id": "candidate", "dataset": "dataset", "plan": task_reference_candidate_plan()}]
    cli = CandidateCLI(root, **options)
    runner = Fleet(root, cli, tasks, 1, root / ".runtime/state.json")
    cli.fleet = runner
    runner.budget.ledger.value["minimum_account_balance"] = 150
    runner.budget.ledger.save()
    return runner, cli


def run(runner):
    timer = [0.0]

    def clock():
        timer[0] += 0.25
        return timer[0]

    with (
        patch("rl_stocks.runtime.colab.fleet.time.monotonic", side_effect=clock),
        patch("rl_stocks.runtime.colab.fleet.time.sleep"),
    ):
        return runner.run()


def test_transport_complete_stops_before_any_model_acceptance(tmp_path, monkeypatch):
    runner, cli = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "rl_stocks.runtime.colab.fleet.validate_archive",
        lambda *a: pytest.fail("model acceptance occurred before shutdown"),
    )
    state = run(runner)
    assert state["status"] == "complete" and state["cleanup_verified"]
    assert cli.active == {}
    child = next(iter(runner.budget.active["children"].values()))
    assert child["actual_default_python"] == [3, 13, 18]
    assert child["transport_verified"] is True and child["training_verified"] is False
    assert "post-shutdown" in state["completed"]["candidate"]["acceptance_stage"]
    kernel_index = next(
        i
        for i, c in enumerate(cli.calls)
        if c[0] == "exec" and Path(c[-1]).name == "check_colab_kernel.py"
    )
    install_index = next(i for i, c in enumerate(cli.calls) if c[0] == "install")
    assert kernel_index < install_index
    native = [c for c in cli.calls if c[0] == "exec" and Path(c[-1]).name == "remote_smoke.py"]
    assert len(native) == 1 and "--timeout" in native[0]
    snapshots = child.get("snapshots", [])
    assert len({s["path"] for s in snapshots}) == len(snapshots)
    assert all(Path(s["path"]).exists() for s in snapshots)


def test_new_default_kernel_stops_before_install_or_learning(tmp_path, monkeypatch):
    runner, cli = setup(tmp_path, monkeypatch, kernel=(3, 14, 0))
    with pytest.raises(RuntimeError):
        run(runner)
    assert cli.active == {}
    assert not any(c[0] == "install" for c in cli.calls)
    assert not any(c[0] == "exec" and Path(c[-1]).name == "remote_smoke.py" for c in cli.calls)
    assert runner.state["cleanup_verified"]


@pytest.mark.parametrize("failure", ["meter", "interrupt", "worker_reply"])
def test_failure_or_lost_reply_never_retrains_and_cleans_owned_runtime(
    tmp_path, monkeypatch, failure
):
    options = {
        "fail_meter": failure == "meter",
        "interrupt": failure == "interrupt",
        "worker_error": failure == "worker_reply",
    }
    runner, cli = setup(tmp_path, monkeypatch, **options)
    if failure == "worker_reply":
        assert run(runner)["status"] == "complete"
    else:
        with pytest.raises(RuntimeError):
            run(runner)
    assert cli.active == {} and runner.state["cleanup_verified"]
    assert sum(c[0] == "new" for c in cli.calls) == 1
    assert sum(c[0] == "exec" and Path(c[-1]).name == "remote_smoke.py" for c in cli.calls) <= 1
