"""Archive semantic fixtures; no training or learned-restoration claim."""

from copy import deepcopy
import hashlib
import json
import tarfile

import pytest

from test_vix_artifacts import archive as vix_archive
from test_treasury_initialization import model
from test_treasury_payload import synthetic_sources
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.treasury_payload import build_payload
from rl_stocks.runtime.colab import treasury_artifacts as checks
from rl_stocks.runtime.colab.jobs import treasury_plan
from rl_stocks.training import treasury_initialization as starts


@pytest.fixture(scope="module")
def sources():
    return synthetic_sources()


def fixture(root, arm, monkeypatch, sources):
    packed, _, _ = vix_archive(root, "legacy")
    with tarfile.open(packed) as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    prefix = "artifacts/runs/fixture/"
    run = json.loads(files[prefix + "metadata.json"])
    folder = "artifacts/frozen/" + run["bundle_id"] + "/"
    frozen = json.loads(files[folder + "manifest.json"])
    plan = treasury_plan(arm, smoke=True)
    m = model(arm)
    proof = starts.apply_initialization(m, arm, smoke=True)
    initial = starts.model_record(m)
    final = deepcopy(initial)
    final.update(num_timesteps=512, policy_tensor_sha256="b" * 64)
    final["optimizer"].update(
        num_timesteps=512, sb3_nominal_epoch_counter=40,
        optimizer_state_count=21, adam_step_min=80, adam_step_max=80,
        per_parameter_adam_steps={n: 80 for n, _ in m.policy.named_parameters()},
    )
    run.update(treasury_arm=arm, treasury_binding_sha256=checks.BINDING_SHA256,
               study_protocol_sha256=checks.PROTOCOL_SHA256,
               treasury_shared_initialization=proof, treasury_initialization=initial,
               treasury_selected=deepcopy(final), treasury_final=final)
    spec = deepcopy(checks.binding())
    run["dataset"] = spec["datasets"][arm]
    if arm != "legacy":
        data = build_payload(*sources, arm)
        frozen.update(names=data["names"], feature_schema=data["schema"],
                      normalizer=data["normalizer"], data_manifest=data["manifest"])
        lineage = frozen["data_manifest"]["treasury_context"]
        lineage.update(source_dataset=spec["datasets"]["legacy"], source_context=spec["source_context"])
        # Bind this small synthetic lineage; production binding stays immutable.
        spec["dataset_contracts"][arm].update(
            normalizer_sha256=digest(data["normalizer"]),
            eligibility_sha256=lineage["eligibility_sha256"],
            extra_normalizer_fit_rows=lineage["extra_normalizer_fit_rows"],
        )
        monkeypatch.setattr(checks, "binding", lambda: spec)
    frozen["metadata"] = {k: v for k, v in run.items() if k != "bundle_id"}
    frozen["model_sha256"] = hashlib.sha256(files[prefix + "best.zip"]).hexdigest()
    files[prefix + "treasury-initialization.npz"] = starts.portable_path(proof["artifact_file"]).read_bytes()
    return files, prefix, folder, run, plan, frozen


@pytest.mark.parametrize("arm", ["legacy", "masked", "full"])
def test_semantic_archive_contract(tmp_path, monkeypatch, sources, arm):
    files, prefix, folder, run, plan, frozen = fixture(tmp_path, arm, monkeypatch, sources)
    files[folder + "manifest.json"] = json.dumps(frozen).encode()
    checks.validate_treasury_artifacts(files.__getitem__, prefix, folder, run, plan)


@pytest.mark.parametrize("change", ["arm", "optimizer", "eligibility", "years", "duration", "weights"])
def test_semantic_drift_rejects_even_with_consistent_metadata(tmp_path, monkeypatch, sources, change):
    files, prefix, folder, run, plan, frozen = fixture(tmp_path, "full", monkeypatch, sources)
    if change == "arm":
        run["treasury_arm"] = "masked"
    elif change == "optimizer":
        run["treasury_final"]["optimizer"]["adam_step_max"] = 79
    elif change == "eligibility":
        frozen["data_manifest"]["treasury_context"]["eligibility_sha256"] = "0" * 64
    elif change == "years":
        run["sampling_diagnostics"].pop("year_counts")
    elif change == "duration":
        run["duration_seconds"] = float("nan")
    else:
        frozen["model_sha256"] = "0" * 64
    frozen["metadata"] = {k: v for k, v in run.items() if k != "bundle_id"}
    files[folder + "manifest.json"] = json.dumps(frozen).encode()
    with pytest.raises((RuntimeError, ValueError)):
        checks.validate_treasury_artifacts(files.__getitem__, prefix, folder, run, plan)


@pytest.mark.parametrize("arm", ["legacy", "masked", "full"])
def test_generic_archive_dispatch(tmp_path, monkeypatch, sources, arm):
    from io import BytesIO
    from rl_stocks.core.contracts import canonical
    from rl_stocks.runtime.colab.artifacts import validate_archive
    from rl_stocks.rl.registry.factory import resolved_parameters

    files, prefix, folder, run, plan, frozen = fixture(tmp_path, arm, monkeypatch, sources)
    run["resolved_parameters"] = resolved_parameters(
        "recurrent_ppo", True, plan["jobs"][0]["overrides"], "native",
        plan["dataset_contract"]["market_features"],
    )
    frozen["metadata"] = {k: v for k, v in run.items() if k != "bundle_id"}
    identity = digest(frozen)
    old_folder, folder = folder, "artifacts/frozen/" + identity + "/"
    files = {folder + k[len(old_folder):] if k.startswith(old_folder) else k: v for k, v in files.items()}
    files[folder + "manifest.json"] = canonical(frozen)
    files[folder + "checksums.json"] = canonical({
        k[len(folder):]: hashlib.sha256(v).hexdigest() for k, v in files.items()
        if k.startswith(folder) and not k.endswith("checksums.json")
    })
    run["bundle_id"] = identity
    files[prefix + "metadata.json"] = canonical(run)
    files["artifacts/completion.json"] = canonical(dict(
        plan=plan, dataset=run["dataset"], python=[3, 13, 16], gpu=None,
        completed=[dict(bundle_id=identity)],
    ))
    path = tmp_path / "treasury.tar"
    with tarfile.open(path, "w") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, BytesIO(content))
    assert len(validate_archive(path, plan, run["dataset"])) == 1
