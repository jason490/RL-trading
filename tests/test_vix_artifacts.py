"""Static VIX archive fixtures; no optimization or learned acceptance claim."""

from copy import deepcopy
from dataclasses import asdict
from rl_stocks.environment.execution.simulator import SimulationConfig
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from test_guard_artifacts import archive as guard_archive
from test_vix_context import construct
from rl_stocks.core.contracts import canonical, digest
from rl_stocks.data.generation.volatility_payload import AVAILABILITY
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.runtime.colab.distributional_artifacts import DEVELOPMENT_SYMBOLS
from rl_stocks.runtime.colab.jobs import vix_plan
from rl_stocks.training import vix_context as study
from rl_stocks.training.vix_spec import parameters
from rl_stocks.rl.registry.factory import resolved_parameters

ROOT = Path(__file__).parents[1]


def archive(root, arm="full", mutate=None, missing=None):
    guard_archive(root, "control", True)
    artifacts = root / "artifacts"
    run = artifacts / "runs/fixture"
    metadata = json.loads((run / "metadata.json").read_text())
    old = artifacts / "frozen" / metadata.pop("bundle_id")
    frozen = json.loads((old / "manifest.json").read_text())
    plan = vix_plan(arm, smoke=True)
    job = plan["jobs"][0]
    model = construct(arm)
    proof = study.apply_initialization(model, arm, smoke=True)
    initial = study.model_record(model)
    final = deepcopy(initial)
    final.update(num_timesteps=512, policy_tensor_sha256="b" * 64)
    final["optimizer"].update(
        num_timesteps=512,
        sb3_nominal_epoch_counter=40,
        optimizer_state_count=21,
        adam_step_min=80,
        adam_step_max=80,
        per_parameter_adam_steps={n: 80 for n, _ in model.policy.named_parameters()},
    )
    for key in list(metadata):
        if key.startswith("guard_"):
            del metadata[key]
    metadata.update(
        {k: v for k, v in job.items() if k not in ("initialization", "start_sampling")},
        simulation=asdict(SimulationConfig()),
        vix_arm=arm,
        vix_binding_sha256=study.BINDING_SHA256,
        vix_initialization=initial,
        vix_final=deepcopy(final),
        vix_selected=deepcopy(final),
        vix_shared_initialization=proof,
        duration_seconds=10.0,
        resolved_parameters=resolved_parameters("recurrent_ppo", True, parameters(True)),
    )
    metadata["dataset"] = study.binding()["datasets"][arm]
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
        decision_years={year: 512},
        execution_years={year: 512},
        decision_execution_year_pairs={year + "->" + year: 512},
        per_symbol=per,
    )
    contract = study.binding()["dataset_contracts"][arm]
    frozen.update(names=contract["names"], feature_schema=contract["schema"])
    payloads = json.loads((ROOT / "results/vix-context-2026-10-25/payloads.json").read_text())
    if arm != "legacy":
        frozen["normalizer"] = payloads["records"][arm]["normalizer"]
        frozen["data_manifest"]["vix_context"] = dict(
            version=1,
            mode=arm,
            symbol="^VIX",
            source_provider="yahoo",
            source_feed="yahoo_index_close",
            names=contract["names"][7:],
            normalization_training_only=True,
            historical_publication_vintages_verified=False,
            executable=False,
            availability=AVAILABILITY,
            source_lag_sessions=1,
            change_sessions=20,
            source_dataset=study.binding()["datasets"]["legacy"],
            source_context=study.binding()["source_context"],
            source_normalizer_sha256=study.binding()["dataset_contracts"]["legacy"][
                "normalizer_sha256"
            ],
            source_episodes_sha256="a" * 64,
            source_request=dict(start="2009-01-01", end_exclusive="2024-01-01"),
            retrieved_time="2026-10-25T00:00:00+00:00",
            extra_normalizer_fit_rows=contract["extra_normalizer_fit_rows"],
        )
    source_report = json.loads((run / "validation.json").read_text())["fixture"]
    reports = {
        symbol: dict(deepcopy(source_report), symbol=symbol) for symbol in DEVELOPMENT_SYMBOLS
    }
    for name in ("validation.json", "selected-validation.json"):
        (run / name).write_text(json.dumps(reports))
    # Checkpoint state binds the updated report. Archive-only weights remain synthetic.
    from rl_stocks.training.checkpoint_state import write_checkpoint_state

    write_checkpoint_state(run, reports, "latest.zip", 512, "post_update", 512, "post_update", True)
    frozen["model_sha256"] = hashlib.sha256((run / "best.zip").read_bytes()).hexdigest()
    (old / "model.zip").write_bytes((run / "best.zip").read_bytes())
    shutil.copyfile(ROOT / proof["artifact_file"], run / "vix-initialization.npz")
    if mutate:
        mutate(metadata, frozen)
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
    packed = root / "vix.tar"
    with tarfile.open(packed, "w") as target:
        target.add(artifacts, arcname="artifacts")
    return packed, plan, metadata["dataset"]


@pytest.mark.parametrize("arm", ["legacy", "masked", "full"])
def test_registered_archives_accept(tmp_path, arm):
    assert len(validate_archive(*archive(tmp_path, arm))) == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda m, f: m.update(vix_arm="legacy"),
        lambda m, f: m.update(vix_binding_sha256="0" * 64),
        lambda m, f: m["vix_final"]["optimizer"].update(adam_step_max=79),
        lambda m, f: m["vix_initialization"]["rng"].update(torch="0" * 64),
        lambda m, f: m["action_execution"].update(evaluation_exposure_mode="unrestricted"),
        lambda m, f: m["sampling_diagnostics"].pop("year_counts"),
        lambda m, f: m["simulation"].update(fee_bps=0),
        lambda m, f: f["normalizer"]["means"].__setitem__(7, 100.0),
        lambda m, f: f["data_manifest"]["vix_context"].update(source_context="0" * 64),
        lambda m, f: f["data_manifest"]["vix_context"].update(
            historical_publication_vintages_verified=True
        ),
        lambda m, f: f.update(model_sha256="0" * 64),
    ],
)
def test_checksums_do_not_excuse_contract_drift(tmp_path, change):
    with pytest.raises(RuntimeError):
        validate_archive(*archive(tmp_path, mutate=change))


@pytest.mark.parametrize(
    "missing", ["vix-initialization.npz", "checkpoint-state.json", "latest.zip"]
)
def test_missing_required_artifacts_reject(tmp_path, missing):
    with pytest.raises(RuntimeError):
        validate_archive(*archive(tmp_path, missing=missing))
