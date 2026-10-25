"""Static archive/math fixtures; physical learned restoration remains separate."""

from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pytest

from test_guard_artifacts import archive as guard_archive
from test_gae_initialization import model
from test_gae_jobs import accepted_profile
from test_gae_rollouts import fixture as buffer_fixture
from rl_stocks.core.contracts import canonical, digest
from rl_stocks.environment.execution.simulator import SimulationConfig
from dataclasses import asdict
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.runtime.colab.distributional_artifacts import DEVELOPMENT_SYMBOLS
from rl_stocks.runtime.colab.jobs import gae_plan
from rl_stocks.rl.registry.factory import resolved_parameters
from rl_stocks.training import gae_initialization as study
from rl_stocks.training.gae_rollouts import validate_arrays, GaeRolloutAudit
from rl_stocks.training.gae_spec import parameters, PROTOCOL_SHA256

ROOT = Path(__file__).parents[1]


def archive(root, arm="mc_trace", smoke=True, mutate=None, missing=None):
    guard_archive(root, "control", smoke)
    artifacts = root / "artifacts"
    run = artifacts / "runs/fixture"
    metadata = json.loads((run / "metadata.json").read_text())
    old = artifacts / "frozen" / metadata.pop("bundle_id")
    frozen = json.loads((old / "manifest.json").read_text())
    plan = gae_plan(arm, smoke=smoke, profile=None if smoke else accepted_profile(root))
    job = plan["jobs"][0]
    params = parameters(arm, smoke)
    m = model(arm, smoke=smoke)
    proof = study.apply_initialization(m, arm, smoke=smoke)
    initial = study.model_record(m)
    final = deepcopy(initial)
    target = job["steps"]
    epochs = target // params["n_steps"] * params["n_epochs"]
    adam = epochs * params["n_steps"] // params["batch_size"]
    final.update(num_timesteps=target, policy_tensor_sha256="b" * 64, optimizer_sha256="e" * 64)
    final["optimizer"].update(
        num_timesteps=target,
        sb3_nominal_epoch_counter=epochs,
        optimizer_state_count=21,
        adam_step_min=adam,
        adam_step_max=adam,
        per_parameter_adam_steps={n: adam for n, _ in m.policy.named_parameters()},
    )
    for key in list(metadata):
        if key.startswith("guard_"):
            del metadata[key]
    metadata.update(
        {k: v for k, v in job.items() if k not in ("initialization", "start_sampling")},
        simulation=asdict(SimulationConfig()),
        gae_arm=arm,
        gae_shared_initialization=proof,
        gae_initialization=initial,
        gae_final=deepcopy(final),
        gae_selected=deepcopy(final),
        resolved_parameters=resolved_parameters("recurrent_ppo", smoke, params),
        duration_seconds=10.0,
    )
    counts = metadata["sampling_diagnostics"]
    year = "2010"
    per = {
        key.split(":")[0]: dict(
            **row,
            decision_years={year: row["transitions"]},
            execution_years={year: row["transitions"]},
        )
        for key, row in counts["history_groups"].items()
    }
    counts["year_counts"] = dict(
        version=1,
        measured=True,
        start_years={year: counts["starts"]},
        decision_years={year: target},
        execution_years={year: target},
        decision_execution_year_pairs={year + "->" + year: target},
        per_symbol=per,
    )
    source_report = next(iter(json.loads((run / "validation.json").read_text()).values()))
    reports = {s: dict(deepcopy(source_report), symbol=s) for s in DEVELOPMENT_SYMBOLS}
    for name in ["validation.json", "selected-validation.json"]:
        (run / name).write_text(json.dumps(reports))
    from rl_stocks.training.checkpoint_state import write_checkpoint_state

    write_checkpoint_state(
        run, reports, "latest.zip", target, "post_update", target, "post_update", True
    )
    frozen["model_sha256"] = hashlib.sha256((run / "best.zip").read_bytes()).hexdigest()
    (old / "model.zip").write_bytes((run / "best.zip").read_bytes())
    shutil.copyfile(ROOT / proof["artifact_file"], run / "gae-initialization.npz")
    arrays = buffer_fixture(arm, smoke)
    memory = BytesIO()
    np.savez_compressed(memory, **arrays)
    raw = memory.getvalue()
    sidecar = []
    for i, step in enumerate(range(params["n_steps"], target + 1, params["n_steps"])):
        filename = f"gae-rollouts/rollout-{step:06d}.npz"
        path = run / filename
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
        record = validate_arrays(arrays, arm, smoke)
        record.update(
            protocol_sha256=PROTOCOL_SHA256,
            arm=arm,
            steps=step,
            rollout_index=i + 1,
            arrays_file=filename,
            arrays_sha256=hashlib.sha256(raw).hexdigest(),
            before_optimizer_update=True,
            sb3_nominal_epoch_counter=i * params["n_epochs"],
        )
        sidecar.append(record)
    (run / "gae-rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sidecar))
    audit = GaeRolloutAudit(run, arm, smoke)
    audit.records = sidecar
    metadata["gae_rollouts"] = audit.summary()
    if mutate:
        mutate(metadata, frozen, run)
    frozen["metadata"] = deepcopy(metadata)
    identity = digest(frozen)
    folder = old.with_name(identity)
    old.rename(folder)
    (folder / "manifest.json").write_bytes(canonical(frozen))
    (folder / "checksums.json").unlink()
    (folder / "checksums.json").write_bytes(
        canonical({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()})
    )
    metadata["bundle_id"] = identity
    (run / "metadata.json").write_text(json.dumps(metadata))
    (run / "training-sampling.json").write_text(json.dumps(counts))
    (artifacts / "completion.json").write_text(
        json.dumps(
            dict(
                plan=plan,
                dataset=metadata["dataset"],
                python=[3, 13, 16],
                gpu=None,
                completed=[dict(bundle_id=identity)],
            )
        )
    )
    if missing:
        (run / missing).unlink()
    packed = root / "gae.tar"
    with tarfile.open(packed, "w") as stream:
        stream.add(artifacts, arcname="artifacts")
    return packed, plan, metadata["dataset"]


@pytest.mark.parametrize("arm", ["control", "mc_trace"])
@pytest.mark.parametrize("smoke", [True, False])
def test_both_registered_phases_validate_complete_actual_array_fixtures(tmp_path, arm, smoke):
    assert len(validate_archive(*archive(tmp_path, arm, smoke))) == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda m, f, r: m.update(gae_arm="control"),
        lambda m, f, r: m["gae_final"].update(gae_lambda=0.95),
        lambda m, f, r: m["gae_final"].update(gae_lambda=True),
        lambda m, f, r: m["gae_final"]["optimizer"].update(adam_step_max=0),
        lambda m, f, r: m["gae_rollouts"].update(all_checked_before_optimizer=False),
        lambda m, f, r: m["gae_rollouts"].update(maximum_return_error=1.0),
        lambda m, f, r: m["action_execution"].update(evaluation_exposure_mode="unrestricted"),
        lambda m, f, r: m["sampling_diagnostics"].pop("year_counts"),
        lambda m, f, r: f["normalizer"]["means"].__setitem__(0, 100.0),
        lambda m, f, r: (r / "gae-rollout-failure.json").write_text("{}"),
        lambda m, f, r: (r / "gae-rollouts/extra.npz").write_bytes(b"extra"),
    ],
)
def test_summary_checksums_cannot_hide_gae_schema_math_or_failure_drift(tmp_path, change):
    with pytest.raises(RuntimeError):
        validate_archive(*archive(tmp_path, mutate=change))


@pytest.mark.parametrize(
    "missing",
    [
        "gae-initialization.npz",
        "checkpoint-state.json",
        "latest.zip",
        "gae-rollouts.jsonl",
        "gae-rollouts/rollout-000128.npz",
    ],
)
def test_required_input_and_actual_array_evidence_cannot_be_omitted(tmp_path, missing):
    with pytest.raises(RuntimeError):
        validate_archive(*archive(tmp_path, missing=missing))


def test_tampered_npz_with_updated_checksum_still_rejects_target_math(tmp_path):
    def mutate(m, f, run):
        path = run / "gae-rollouts/rollout-000128.npz"
        from rl_stocks.training.gae_rollouts import read_arrays

        values = read_arrays(path.read_bytes())
        values["returns"][0, 0] += 1.0
        with path.open("wb") as stream:
            np.savez_compressed(stream, **values)
        records = [json.loads(r) for r in (run / "gae-rollouts.jsonl").read_text().splitlines()]
        records[0]["arrays_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (run / "gae-rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    with pytest.raises(RuntimeError):
        validate_archive(*archive(tmp_path, mutate=mutate))
