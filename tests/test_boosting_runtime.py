from copy import deepcopy
from io import BytesIO
import tarfile

import pytest

from rl_stocks.core.contracts import canonical
from rl_stocks.runtime.colab.boosting_runtime import admission, claim, extract_transport
from rl_stocks.runtime.colab.budget import Ledger
from rl_stocks.runtime.colab.jobs import (
    boosting_plan,
    reviewed_boosting_plan,
    component_boosting_plan,
    validate_plan,
)
from test_task_reference_fleet import setup, CandidateCLI, run as run_fleet
from rl_stocks.runtime.colab.fleet import Fleet


def test_fixed_plan_claim_and_persistent_baseline(tmp_path):
    plan = boosting_plan()
    assert validate_plan(plan) == plan
    for field, value in [("version", 48), ("accelerator", "A100"), ("protocol_sha256", "wrong")]:
        bad = deepcopy(plan)
        bad[field] = value
        with pytest.raises(ValueError):
            validate_plan(bad)
    ledger = Ledger(tmp_path / "usage.json")
    assert admission(tmp_path, ledger) == (0.25, 0.0)
    ledger.value["consumed"] = 0.1
    assert admission(tmp_path, ledger) == (0.25, 0.0)
    claim(ledger, "rl-stocks-cpu-123456789abc")
    with pytest.raises(RuntimeError):
        admission(tmp_path, ledger)
    with pytest.raises(RuntimeError):
        claim(ledger, "rl-stocks-cpu-123456789abc")


def archive(path, plan, extra=False):
    identity = "a" * 64
    prefix = "artifacts/boosting/"
    keys = {f"{a}-{s}.npz" for a in ("masked", "full") for s in (11, 23, 37)}
    files = {
        prefix + n: b"x"
        for n in ("prefix.npz", "timing.json", "progress.json", "probe-predictions.npz")
    }
    files.update({prefix + "models/" + k: b"x" for k in keys})
    files.update(
        {
            prefix + "frozen/" + identity + "/" + k: b"x"
            for k in keys | {"manifest.json", "checksums.json"}
        }
    )
    files[prefix + "result.json"] = canonical({"bundle_id": identity})
    completion = {
        "plan": plan,
        "dataset": plan["dataset_contract"]["identity"],
        "gpu": None,
        "python": [3, 13, 18],
        "source_git": {"revision": "b" * 40, "dirty": ""},
        "source_hash": "c" * 64,
    }
    files["artifacts/completion.json"] = canonical(completion)
    if extra:
        files["artifacts/boosting/unexpected.txt"] = b"bad"
    with tarfile.open(path, "w:gz") as t:
        for name, data in files.items():
            m = tarfile.TarInfo(name)
            m.size = len(data)
            t.addfile(m, BytesIO(data))
    return completion


@pytest.mark.parametrize("reviewed", [False, True, "component"])
def test_complete_transport_before_offline_model_acceptance(tmp_path, reviewed):
    plan = (
        component_boosting_plan()
        if reviewed == "component"
        else reviewed_boosting_plan()
        if reviewed
        else boosting_plan()
    )
    p = tmp_path / "ok.tar.gz"
    completion = archive(p, plan)
    assert (
        extract_transport(p, plan, plan["dataset_contract"]["identity"], tmp_path / "out")
        == completion
    )
    archive(tmp_path / "bad.tar.gz", plan, extra=True)
    with pytest.raises(ValueError):
        extract_transport(
            tmp_path / "bad.tar.gz", plan, plan["dataset_contract"]["identity"], tmp_path / "badout"
        )
    assert not (tmp_path / "badout").exists()


@pytest.mark.parametrize("reviewed", [False, True, "component"])
def test_boosting_lifecycle_stops_before_offline_acceptance(tmp_path, monkeypatch, reviewed):
    original, _ = setup(tmp_path, monkeypatch)
    root = original.root
    from rl_stocks.runtime.colab import boosting_runtime

    monkeypatch.setattr(boosting_runtime, "admission", lambda *a: (0.25, 0.0))
    from rl_stocks.runtime.colab import boosting_review

    monkeypatch.setattr(boosting_review, "admission", lambda *a: (0.25, 0.0))

    def transport(path, plan, dataset_id, destination):
        destination.mkdir()
        return {"python": [3, 13, 18], "source_git": {"revision": "a" * 40, "dirty": ""}}

    monkeypatch.setattr(boosting_runtime, "extract_transport", transport)
    cli = CandidateCLI(root)
    runner = Fleet(
        root,
        cli,
        [
            {
                "id": "boost",
                "dataset": "dataset",
                "plan": component_boosting_plan()
                if reviewed == "component"
                else reviewed_boosting_plan()
                if reviewed
                else boosting_plan(),
            }
        ],
        1,
        root / ".runtime/boost.json",
    )
    run_fleet(runner)
    assert runner.state["cleanup_verified"] and not cli.active
    bootstrap = next(
        a for a in cli.calls if a[0] == "exec" and any("install_colab_bootstrap.py" in x for x in a)
    )
    assert bootstrap[bootstrap.index("--timeout") + 1] == "180"
    assert not any(a[0] == "install" for a in cli.calls)
    completed = runner.state["completed"]["boost"]
    assert completed["runs"] == []
    args = next(a for a in cli.calls if a[0] == "exec" and "--env" in a)
    assert any(s.startswith("RL_ALLOCATION_DEADLINE=") for s in args)
    assert runner.budget.active["children"][completed["session"]]["training_verified"] is False


@pytest.mark.parametrize(
    "options", [{"kernel": (3, 14, 0)}, {"fail_meter": True}, {"worker_error": True}]
)
def test_boosting_failure_cleans_owned_runtime(tmp_path, monkeypatch, options):
    original, _ = setup(tmp_path, monkeypatch)
    root = original.root
    from rl_stocks.runtime.colab import boosting_runtime

    monkeypatch.setattr(boosting_runtime, "admission", lambda *a: (0.25, 0.0))
    cli = CandidateCLI(root, **options)
    runner = Fleet(
        root,
        cli,
        [{"id": "boost", "dataset": "dataset", "plan": boosting_plan()}],
        1,
        root / ".runtime/boost.json",
    )
    with pytest.raises(RuntimeError):
        run_fleet(runner)
    assert runner.state["cleanup_verified"] and not cli.active
    if "kernel" in options:
        assert not any(a[0] == "install" for a in cli.calls)

    if options.get("worker_error"):
        assert (
            sum(
                a[0] == "exec" and any("install_colab_bootstrap.py" in x for x in a)
                for a in cli.calls
            )
            == 1
        )
        assert not any("--env" in a for a in cli.calls)


@pytest.mark.parametrize("component", [False, True])
def test_review_admission_preserves_failure_charges_and_science(tmp_path, monkeypatch, component):
    from rl_stocks.runtime.colab import boosting_review as review

    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "results/residual-boosting-2026-11-04").mkdir(parents=True)
    originals = {
        ".runtime/residual-boosting-fleet-v1.json": b"original failed state",
        ".runtime/residual-boosting-allocation-v1.json": b"original claim",
    }
    for name, data in originals.items():
        (tmp_path / name).write_bytes(data)
    failure = {
        "worker_started": False,
        "model_fitting_performed": False,
        "cleanup_verified": True,
        "protected": {name: review.sha(tmp_path / name) for name in originals},
    }
    receipt = tmp_path / "results/residual-boosting-2026-11-04/bootstrap-failure.json"
    receipt.write_bytes(canonical(failure))
    (tmp_path / "science.py").write_text("unchanged")
    (tmp_path / "scripts/install_colab_bootstrap.py").write_text("bounded bootstrap")
    config = {
        "failure_receipt_sha256": review.sha(receipt),
        "original_study_start_units": 40.0,
        "scientific_protocol_sha256": "fixed",
        "original_failed_session": "old",
        "original_failure_project_units": 40.01,
        "unchanged_scientific_source_sha256": {"science.py": review.sha(tmp_path / "science.py")},
        "bootstrap_script_sha256": review.sha(tmp_path / "scripts/install_colab_bootstrap.py"),
    }
    monkeypatch.setattr(review, "spec", lambda *a: config)
    baseline = {"start_units": 40.0, "limit_units": 0.25, "protocol_sha256": "fixed"}
    base = runtime / "residual-boosting-budget-v1.json"
    base.write_bytes(canonical(baseline))
    ledger = Ledger(runtime / "usage-ledger.json")
    ledger.value["consumed"] = 40.01
    ledger.value["attempts"] = [
        {
            "status": "stopped",
            "children": {"old": {"session": "old", "status": "stopped", "cleanup_verified": True}},
        }
    ]
    assert review.admission(tmp_path, ledger, component) == (0.25, 40.0)
    baseline["start_units"] = 40.01
    base.write_bytes(canonical(baseline))
    with pytest.raises(ValueError, match="baseline"):
        review.admission(tmp_path, ledger, component)
    baseline["start_units"] = 40.0
    base.write_bytes(canonical(baseline))
    (tmp_path / "science.py").write_text("changed")
    with pytest.raises(ValueError, match="scientific"):
        review.admission(tmp_path, ledger, component)
    (tmp_path / "science.py").write_text("unchanged")
    review.claim(ledger, "rl-stocks-cpu-123456789abc", component)
    with pytest.raises(RuntimeError):
        review.admission(tmp_path, ledger, component)
    with pytest.raises(RuntimeError):
        review.claim(ledger, "rl-stocks-cpu-123456789abc", component)
