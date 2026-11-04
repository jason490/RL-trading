"""Single allocation ownership and bounded residual-boosting transport."""

import json
from pathlib import Path, PurePosixPath
import re
import tarfile

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.training.supervised.boosting_spec import PROTOCOL_SHA256, protocol

CLAIM = "residual-boosting-allocation-v1.json"


def admission(root, ledger):
    root = Path(root)
    protocol()
    if (ledger.path.parent / CLAIM).exists():
        raise RuntimeError("residual boosting already claimed; no automatic retry")
    if ledger.value["budget_stopped"] or any(
        a["status"] != "stopped" for a in ledger.value["attempts"]
    ):
        raise RuntimeError("unresolved ownership or original budget stop")
    path = ledger.path.parent / "residual-boosting-budget-v1.json"
    if not path.exists():
        write_once(
            path,
            canonical(
                {
                    "protocol_sha256": PROTOCOL_SHA256,
                    "start_units": ledger.value["consumed"],
                    "limit_units": 0.25,
                }
            ),
        )
    baseline = json.loads(path.read_text())
    if baseline["protocol_sha256"] != PROTOCOL_SHA256 or baseline["limit_units"] != 0.25:
        raise ValueError("unchanged residual-boosting baseline required")
    return baseline["limit_units"], baseline["start_units"]


def claim(ledger, session):
    if (ledger.path.parent / CLAIM).exists():
        raise RuntimeError("residual boosting allocation already claimed")
    if not re.fullmatch(r"rl-stocks-cpu-[a-f0-9]{12}", session):
        raise ValueError("owned CPU session required")
    write_once(
        ledger.path.parent / CLAIM,
        canonical(
            {"session": session, "protocol_sha256": PROTOCOL_SHA256, "automatic_retry": False}
        ),
    )


def extract_transport(path, plan, dataset_id, destination):
    from rl_stocks.runtime.colab.jobs import (
        boosting_plan,
        reviewed_boosting_plan,
        component_boosting_plan,
        validate_plan,
    )

    validate_plan(plan)
    if (
        canonical(plan)
        != canonical(
            component_boosting_plan()
            if plan.get("version") == 51
            else reviewed_boosting_plan()
            if plan.get("version") == 50
            else boosting_plan()
        )
        or dataset_id != plan["dataset_contract"]["identity"]
    ):
        raise ValueError("fixed boosting transport required")
    keys = {f"{a}-{s}.npz" for a in ("masked", "full") for s in (11, 23, 37)}
    prefix = "artifacts/boosting/"
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        files = {m.name for m in members if m.isfile()}
        if (
            len(members) > 64
            or len({m.name for m in members}) != len(members)
            or sum(m.size for m in members) > 512 * 1024 * 1024
        ):
            raise ValueError("bounded unique boosting archive required")
        if "artifacts/completion.json" not in files or prefix + "result.json" not in files:
            raise ValueError("complete boosting export required")
        for member in members:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or member.size < 0
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("safe ordinary boosting members required")
        result_member = archive.getmember(prefix + "result.json")
        completion_member = archive.getmember("artifacts/completion.json")
        if max(result_member.size, completion_member.size) > 2 * 1024 * 1024:
            raise ValueError("bounded completion metadata required")
        result = json.load(archive.extractfile(result_member))
        identity = result["bundle_id"]
        if not re.fullmatch(r"[a-f0-9]{64}", identity):
            raise ValueError("valid frozen identity required")
        frozen = prefix + "frozen/" + identity + "/"
        required = (
            {
                prefix + n
                for n in (
                    "prefix.npz",
                    "timing.json",
                    "progress.json",
                    "result.json",
                    "probe-predictions.npz",
                )
            }
            | {prefix + "models/" + k for k in keys}
            | {frozen + k for k in keys | {"manifest.json", "checksums.json"}}
            | {"artifacts/completion.json"}
        )
        if not required <= files or files - required - {"artifacts/bootstrap-log.txt"}:
            raise ValueError("exact complete boosting files required")
        if any(
            m.isdir() and not any(f.startswith(m.name.rstrip("/") + "/") for f in required)
            for m in members
        ):
            raise ValueError("unexpected transport directory")
        completion = json.load(archive.extractfile(completion_member))
        git = completion.get("source_git", {})
        python = completion.get("python", [])
        if (
            canonical(completion.get("plan")) != canonical(plan)
            or completion.get("dataset") != dataset_id
            or completion.get("gpu") is not None
            or len(python) != 3
            or python[:2] != [3, 13]
            or any(type(v) is not int for v in python)
            or git.get("dirty") != ""
            or not re.fullmatch(r"[a-f0-9]{40}", git.get("revision", ""))
            or not re.fullmatch(r"[a-f0-9]{64}", completion.get("source_hash", ""))
        ):
            raise ValueError("bound default-kernel CPU completion required")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination, filter="data")
    return completion
