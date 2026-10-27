"""Allowlisted recurrent environment smoke archive with independent physical restoration."""

import json
from pathlib import Path
import tarfile
import tempfile

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.runtime.colab.jobs import validate_plan
from rl_stocks.training import recurrent_environment_worker as worker
from rl_stocks.training.recurrent_environment_verify import FILES, verify_directory


def validate_environment_archive(path, plan, dataset_id):
    validate_plan(plan)
    if dataset_id != worker.INPUT_ID:
        raise RuntimeError("bound recurrent environment smoke dataset identity required")
    required = {'artifacts/completion.json'} | {
        'artifacts/recurrent-loop/'+name for name in FILES}
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        if len({m.name for m in members}) != len(members):
            raise RuntimeError("unique diagnostic archive members required")
        files = {m.name for m in members if m.isfile()}
        directories = {"artifacts", "artifacts/recurrent-loop"} | {
            f"artifacts/recurrent-loop/{a}" for a in ("initial", "checkpoint-64", "checkpoint-128")
        }
        if (
            not required <= files
            or files - required - {"artifacts/bootstrap-log.txt"}
            or any(not m.isfile() and (not m.isdir() or m.name not in directories) for m in members)
            or sum(m.size for m in members) > 64 * 1024 * 1024
        ):
            raise RuntimeError("complete bounded allowlisted recurrent environment smoke archive required")
        with tempfile.TemporaryDirectory(prefix="recurrent-environment-verify-") as temporary:
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
            folder = root / "recurrent-loop"
            payload = json.loads((folder / "dataset.json").read_text())
            if digest(payload) != dataset_id:
                raise RuntimeError("exact retained diagnostic input required")
            proof = verify_directory(folder)
            summary = json.loads((folder / "run.json").read_text())
            return [
                dict(
                    algorithm="recurrent_environment_probe",
                    seed=11,
                    actual_steps=128,
                    bundle_id=digest(summary),
                    status="complete",
                    numerical_artifacts_verified=proof["saved_artifacts_verified"],
                )
            ]
