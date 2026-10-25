"""Accept only fixed, retrieved and owned-cleaned Colab VIX smokes."""

import argparse
import json
from pathlib import Path
import tarfile
import tempfile

from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.report_contract import validate_development_report
from rl_stocks.inference.bundle import checksum
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.runtime.colab.jobs import vix_plan
from rl_stocks.training.vix_assets import binding, BINDING_SHA256
from rl_stocks.training.vix_context import (
    PROTOCOL_SHA256,
    validate_payload,
    validate_initial_records,
)
from rl_stocks.training.vix_restoration import verify_owned_run
from rl_stocks.training.vix_runtime import projection
from rl_stocks.training.vix_spec import preregistration


def accept(state_path, source_proof_path, data_root):
    state = json.loads(Path(state_path).read_text())
    source = json.loads(Path(source_proof_path).read_text())
    if state["status"] != "complete" or state.get("cleanup_verified") is not True:
        raise ValueError("complete VIX smokes with verified owned termination required")
    arms = preregistration()["arms"]
    expected = {f"vix-{arm}-11-smoke" for arm in arms}
    if (
        set(state["completed"]) != expected
        or len(state["sessions"]) != 3
        or {s["task"] for s in state["sessions"].values()} != expected
        or any(s["phase"] != "done" for s in state["sessions"].values())
    ):
        raise ValueError("exactly three fixed owned-cleaned VIX smoke tasks required")
    records = {}
    for arm in arms:
        task_id = f"vix-{arm}-11-smoke"
        item = state["completed"][task_id]
        if state["sessions"].get(item["session"]) != dict(task=task_id, phase="done"):
            raise ValueError("owned VIX smoke session/task mismatch")
        root = Path(item["root"])
        paths = list((root / "runs").glob("*/metadata.json"))
        if len(paths) != 1:
            raise ValueError("one owned VIX smoke run required")
        m = json.loads(paths[0].read_text())
        data_id = binding()["datasets"][arm]
        data = validate_payload(read_dataset(Path(data_root) / data_id), arm)
        if (
            m["git"]["revision"] != source["source_revision"]
            or m["git"]["dirty"]
            or m["source_hash"] != source["source_hash"]
            or m["study_protocol_sha256"] != PROTOCOL_SHA256
            or m["dataset"] != data_id
            or m["smoke"] is not True
            or m["seed"] != 11
            or m["actual_steps"] != 512
        ):
            raise ValueError("actual clean fixed VIX smoke source/steps required")
        plan = vix_plan(arm, smoke=True)
        if json.loads((root / "completion.json").read_text())["plan"] != plan:
            raise ValueError("actual VIX smoke completion plan mismatch")
        # Revalidate the retained extraction; duplicate transport archives need not
        # survive cleanup. Temporary repack is scoped and always removed.
        with tempfile.TemporaryDirectory(prefix="vix-accept-") as temporary:
            packed = Path(temporary) / "owned.tar"
            with tarfile.open(packed, "w") as archive:
                archive.add(root, arcname="artifacts")
            validate_archive(packed, plan, data_id)
        config = SimulationConfig(**m["simulation"])
        for filename in ("selected-validation.json", "validation.json"):
            validate_development_report(
                json.loads((paths[0].parent / filename).read_text()), data, config
            )
        bundle = root / "frozen" / m["bundle_id"]
        restored = verify_owned_run(paths[0].parent, bundle, data, arm, smoke=True)
        records[arm] = dict(
            session=item["session"],
            run=paths[0].parent.name,
            bundle_id=m["bundle_id"],
            duration_seconds=m["duration_seconds"],
            initial=m["vix_initialization"],
            metadata_sha256=checksum(paths[0]),
            python=m["python"],
            **restored,
        )
    initial = validate_initial_records(
        {arm: r["initial"] for arm, r in records.items()}, smoke=True
    )
    return dict(
        schema="vix-smoke-acceptance-v1",
        passed=True,
        protocol_sha256=PROTOCOL_SHA256,
        binding_sha256=BINDING_SHA256,
        source_proof=source,
        state_sha256=checksum(state_path),
        cleanup_verified=True,
        initialization_checks=initial,
        records=records,
        exact_frozen_predictions=sum(r["frozen_predictions"] for r in records.values()),
        runtime_projection=projection({a: r["duration_seconds"] for a, r in records.items()}),
        training_performed=False,
        optimizer_steps=0,
        test_evaluated=False,
        screen_allocated=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("state", "source-proof", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = accept(args.state, args.source_proof, args.data_root)
    write_once(args.output, (json.dumps(result, indent=2) + "\n").encode())
    print(
        "Accepted3 owned-cleaned512-transition smokes and at least234 exact frozen predictions; no screen allocated."
    )


if __name__ == "__main__":
    main()
