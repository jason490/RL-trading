"""Allowlisted numerical diagnostic archive with independent physical restoration."""

import json
from pathlib import Path
import tarfile
import tempfile

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.runtime.colab.jobs import validate_plan
from rl_stocks.training import sac_update_worker as worker
from rl_stocks.training.sac_update_verify import FILES, verify_directory


def validate_sac_update_archive(path, plan, dataset_id):
    validate_plan(plan)
    if dataset_id != worker.INPUT_ID:
        raise RuntimeError("bound numerical diagnostic dataset identity required")
    required = {
        "artifacts/completion.json",
        "artifacts/sac-update/input.json",
        "artifacts/sac-update/summary.json",
    }
    required |= {
        f"artifacts/sac-update/{arm}/{name}" for arm in worker.protocol()["arms"] for name in FILES
    }
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        if len({m.name for m in members}) != len(members):
            raise RuntimeError("unique diagnostic archive members required")
        files = {m.name for m in members if m.isfile()}
        directories = {"artifacts", "artifacts/sac-update"} | {
            f"artifacts/sac-update/{a}" for a in worker.protocol()["arms"]
        }
        if (
            not required <= files
            or files - required - {"artifacts/bootstrap-log.txt"}
            or any(not m.isfile() and (not m.isdir() or m.name not in directories) for m in members)
            or sum(m.size for m in members) > 64 * 1024 * 1024
        ):
            raise RuntimeError("complete bounded allowlisted numerical diagnostic archive required")
        with tempfile.TemporaryDirectory(prefix="sac-update-verify-") as temporary:
            archive.extractall(temporary, filter="data")
            root = Path(temporary) / "artifacts"
            completion = json.loads((root / "completion.json").read_text())
            python = completion["python"]
            git = completion["source_git"]
            if (
                canonical(completion["plan"]) != canonical(plan)
                or completion["dataset"] != dataset_id
                or completion["gpu"] is not None
                or len(python) != 3
                or any(type(v) is not int for v in python)
                or python[:2] != [3, 13]
                or git.get("dirty") != ""
                or not isinstance(git.get("revision"), str)
                or len(git["revision"]) != 40
                or not isinstance(completion.get("source_hash"), str)
                or len(completion["source_hash"]) != 64
            ):
                raise RuntimeError(
                    "actual clean-source CPU/default-kernel diagnostic completion required"
                )
            folder = root / "sac-update"
            payload = json.loads((folder / "input.json").read_text())
            if digest(payload) != dataset_id:
                raise RuntimeError("exact retained diagnostic input required")
            proof = verify_directory(folder, payload)
            summary = json.loads((folder / "summary.json").read_text())
            return [
                dict(
                    algorithm="sac_update_probe",
                    seed=11,
                    actual_steps=2,
                    bundle_id=digest(summary),
                    status="complete",
                    numerical_artifacts_verified=proof["saved_artifacts_verified"],
                )
            ]
