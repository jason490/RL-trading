"""Admission and one-time ownership for the fixed reference-execution candidate."""

import hashlib
import json
from pathlib import Path
import re
import subprocess

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once

REVIEW_SHA256 = "a0f433f565f937489809f01abb9d828e8bbe5a53e4d4f0d0fc114df157b33500"
REFERENCE_SHA256 = "8c8f7bb00b1ba4c38e01d6dbbee401033d2b4b4d895ad9dbab2cbaf0b37e79a8"
ROOT = Path(__file__).resolve().parents[3]
CLAIM = "task-reference-candidate-tagged_routed-11-allocation-v1.json"
ANNOTATION = b"            from rl_stocks.training.task_backend import task_execution_backend\n            metadata['execution_backend'] = task_execution_backend(model.device)\n"


def execution_spec(root=ROOT):
    value = json.loads(
        (Path(root) / "configs/experiments/task_reference_candidate_execution.json").read_text()
    )
    if digest(value) != REVIEW_SHA256 or value["reference_protocol_sha256"] != REFERENCE_SHA256:
        raise ValueError("fixed routed candidate execution review required")
    return value


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_sources(root, profile, audit):
    from rl_stocks.training.task_profile_sources import SCOPES

    root = Path(root)
    original = profile["compute_sources"]
    current = set()
    for name in SCOPES:
        path = root / name
        if path.is_dir():
            current.update(str(p.relative_to(root)) for p in path.rglob("*.py"))
        elif path.exists():
            current.add(name)
    additions = set(audit["approved_reference_only_additions"])
    if current - set(original) != additions or set(original) - current:
        raise ValueError("reviewed native compute inventory changed")
    for name, expected in original.items():
        if (root / name).is_symlink() or checksum(root / name) != expected:
            raise ValueError("measured native learning source changed: " + name)
    worker = "src/rl_stocks/training/task_screen_worker.py"
    source = (root / worker).read_bytes()
    previous = subprocess.run(
        ["git", "show", audit["native_worker_origin_revision"] + ":" + worker],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    if source.count(ANNOTATION) != 1 or source.replace(ANNOTATION, b"") != previous:
        raise ValueError("native worker changed beyond reviewed read-only annotation")
    if checksum(root / "src/rl_stocks/training/task_backend.py") != audit["backend_helper_sha256"]:
        raise ValueError("reviewed read-only backend helper changed")
    return {
        "original_profiled_files": len(original),
        "reference_only_additions": sorted(additions),
        "native_compute_inventory_sha256": digest(original),
        "native_worker_sha256": checksum(root / worker),
    }


def admit(root, ledger):
    from rl_stocks.runtime.colab.budget import BudgetStop
    from rl_stocks.runtime.colab.jobs import task_profile_plan
    from rl_stocks.runtime.colab.task_profile_artifacts import validate_task_profile_archive
    from rl_stocks.training.task_runtime import shared_budget, verify_all_smokes
    from rl_stocks.evaluation.task_reference_controls import load_assets, validate_matrix

    root = Path(root)
    spec = execution_spec(root)
    runtime = ledger.path.resolve().parent
    if (runtime / CLAIM).exists() or (
        runtime / "task-screen-tagged_routed-allocation-v1.json"
    ).exists():
        raise RuntimeError("routed full candidate already claimed; no automatic retry")
    if (runtime / "task-reference-family-closure-v1.json").exists():
        raise RuntimeError("reference comparison closed; no automatic extension")
    limit, start = shared_budget(ledger)
    if start != spec["shared_original_study_baseline"] or any(
        a["status"] != "stopped" for a in ledger.value["attempts"]
    ):
        raise ValueError("original baseline and verified stopped owned state required")
    files = {
        "reference-controls.json": "control_results_sha256",
        "reference-control-assets.json": "physical_assets_sha256",
        "reference-native-source-audit.json": "native_source_audit_sha256",
        "reference-candidate-projection.json": "projection_sha256",
        "research-profile-acceptance.json": "profile_receipt_sha256",
    }
    records = {}
    for name, key in files.items():
        path = root / "results/task-conditioning-2026-10-31" / name
        if checksum(path) != spec[key]:
            raise ValueError("bound candidate prerequisite changed: " + name)
        records[name] = json.loads(path.read_text())
    controls = records["reference-controls.json"]
    assets = load_assets(root)
    validate_matrix(controls["controls"], assets)
    if checksum(root / controls["artifact_path"]) != controls["artifact_sha256"]:
        raise ValueError("complete control producer artifact changed")
    for phases in controls["controls"].values():
        for point in phases.values():
            if checksum(root / point["paths_path"]) != point["paths_sha256"]:
                raise ValueError("reference control paths changed")
    profile = records["research-profile-acceptance.json"]
    archive = root / profile["archive"]
    if (
        checksum(archive) != profile["archive_sha256"]
        or profile["study_start_units"] != start
        or digest(profile) != spec["profile_acceptance_sha256"]
    ):
        raise ValueError("original measured profile archive/baseline changed")
    plan = task_profile_plan()
    from rl_stocks.runtime.colab.verify_task_screen_export import no_optimization

    with no_optimization():
        proof = validate_task_profile_archive(archive, plan, plan["dataset_contract"]["identity"])
        if digest(proof) != digest(profile["physical_acceptance"]):
            raise ValueError("physical measured profile acceptance changed")
        verify_all_smokes(root, ledger)
    sources = verify_sources(root, profile, records["reference-native-source-audit.json"])
    # Reference results must retain the exact arithmetic implementation that
    # generated the admitted controls, as well as unchanged native learning.
    reference_files = [
        "src/rl_stocks/inference/" + name
        for name in (
            "canonical_math.py",
            "canonical_actor.py",
            "canonical_export.py",
            "canonical_bundle.py",
        )
    ]
    reference_files += records["reference-native-source-audit.json"][
        "approved_reference_only_additions"
    ]
    for relative in reference_files:
        previous = subprocess.run(
            ["git", "show", controls["source"]["revision"] + ":" + relative],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        if (root / relative).read_bytes() != previous:
            raise ValueError("admitted reference arithmetic changed: " + relative)
    projection = records["reference-candidate-projection.json"]
    remaining = limit - (ledger.value["consumed"] - start)
    planned = spec["allocation_seconds"] * (profile["observed_rate"] + 0.01) / 3600
    if (
        projection["projected_allocation_seconds"] > spec["allocation_seconds"]
        or projection["post_stop_verification_seconds"] > spec["post_stop_verification_seconds"]
        or planned >= remaining
    ):
        raise BudgetStop("complete candidate operation exceeds original allowance/deadline")
    return {
        "schema": "task-reference-candidate-admission-v1",
        "review_sha256": REVIEW_SHA256,
        "reference_protocol_sha256": REFERENCE_SHA256,
        "study_start_units": start,
        "remaining_study_units": remaining,
        "planned_allocation_units": planned,
        "source_equivalence": sources,
        "execution_admitted": True,
        "performance_promotion": False,
    }


def claim_candidate(ledger, session):
    from rl_stocks.training.task_runtime import shared_budget

    shared_budget(ledger)
    if not re.fullmatch(r"rl-stocks-cpu-[a-f0-9]{12}", session):
        raise ValueError("owned CPU candidate session required")
    path = ledger.path.resolve().parent / CLAIM
    if (path.parent / "task-reference-family-closure-v1.json").exists() or (
        path.parent / "task-screen-tagged_routed-allocation-v1.json"
    ).exists():
        raise RuntimeError("reference comparison closed or native candidate claimed")
    if path.exists():
        raise RuntimeError("candidate already claimed; no automatic retry")
    spec = execution_spec()
    write_once(
        path,
        canonical(
            {
                "review_sha256": REVIEW_SHA256,
                "reference_protocol_sha256": REFERENCE_SHA256,
                "session": session,
                "role": "tagged_routed",
                "seed": 11,
                "transitions": 100000,
                "study_baseline": spec["shared_original_study_baseline"],
                "allocation_seconds": 7200,
                "teardown_reserve_seconds": 600,
                "automatic_retry_allowed": False,
            }
        ),
    )
