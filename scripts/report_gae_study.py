"""Validate owned GAE screens and report fixed gates without training."""

import argparse
import json
import math
from pathlib import Path
import tarfile
import tempfile

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.evaluation.report_contract import validate_development_report
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.gae_traces import historical_reproduction, screen_gate
from rl_stocks.reporting.action_execution import diagnose_groups
from rl_stocks.reporting.history import render_report
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.runtime.colab.jobs import gae_plan
from rl_stocks.training.curves import latest_points
from rl_stocks.training.gae_spec import protocol as preregistration
from rl_stocks.training.diversity import validate_counts
from rl_stocks.training.gae_context import (
    PROTOCOL_SHA256,
    validate_payload,
)
from rl_stocks.training.gae_restoration import verify_owned_run
from rl_stocks.training.gae_initialization import validate_initial_records
from rl_stocks.training.gae_runtime import (
    REFERENCE_METADATA_SHA256,
    make_runtime_profile,
)


def run_reports(root):
    paths = list((root / "runs").glob("*/metadata.json"))
    if len(paths) != 1:
        raise ValueError("one retained run per owned task required")
    run = paths[0].parent
    return (
        run,
        json.loads(paths[0].read_text()),
        {
            phase: json.loads((run / name).read_text())
            for phase, name in (
                ("selected", "selected-validation.json"),
                ("final", "validation.json"),
            )
        },
    )


def report(state_path, source_path, root):
    spec = preregistration()
    state = json.loads(state_path.read_text())
    source = json.loads(source_path.read_text())
    profile = make_runtime_profile(root)
    ids = {f"gae-{arm}-11" for arm in spec["arms"]}
    if (
        state["status"] != "complete"
        or state.get("cleanup_verified") is not True
        or set(state["completed"]) != ids
        or len(state["sessions"]) != 2
        or {r["task"] for r in state["sessions"].values()} != ids
        or any(r["phase"] != "done" for r in state["sessions"].values())
    ):
        raise ValueError("both completed owned-cleaned fixed screens required")
    cells, reports, metadata = {}, {}, {}
    baseline = contract = None
    for arm in spec["arms"]:
        item = state["completed"][f"gae-{arm}-11"]
        if state["sessions"].get(item["session"]) != dict(task=f"gae-{arm}-11", phase="done"):
            raise ValueError("owned GAE session/task mismatch")
        artifact = Path(item["root"])
        data_id = spec["dataset"]
        data = validate_payload(read_dataset(root / "data/datasets" / data_id))
        stocks = {e["symbol"] for e in data["episodes"].values() if e["split"] == "validation"} - {
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
        }
        plan = gae_plan(arm, profile=profile)
        archive = artifact.parent.parent / "artifacts.tar.gz"
        if archive.exists():
            validate_archive(archive, plan, data_id)
        else:
            with tempfile.TemporaryDirectory(prefix="gae-retained-") as work:
                archive = Path(work) / "retained.tar"
                with tarfile.open(archive, "w") as stream:
                    stream.add(artifact, arcname="artifacts")
                validate_archive(archive, plan, data_id)
        run, m, phases = run_reports(artifact)
        if (
            m["git"]["revision"] != source["source_revision"]
            or m["git"]["dirty"]
            or m["source_hash"] != source["source_hash"]
            or m["gae_arm"] != arm
            or m["seed"] != 11
            or m["smoke"] is not False
        ):
            raise ValueError("fixed clean GAE screen source/arm required")
        current = {
            k: m[k]
            for k in (
                "source_hash",
                "git",
                "simulation",
                "versions",
                "reward_config",
                "evaluation_every",
                "initialization",
                "policy_architecture",
            )
        }
        if contract is not None and canonical(contract) != canonical(current):
            raise ValueError("same GAE screen source/simulation/settings required")
        contract = current
        restored = verify_owned_run(run, artifact / "frozen" / m["bundle_id"], data, arm)
        cell = dict(
            run=run.name,
            bundle_id=m["bundle_id"],
            seed=11,
            learning_curve=latest_points(m["validation_checkpoints"]),
            initialization_record=m["gae_initialization"],
            restoration=restored,
            duration_seconds=m["duration_seconds"],
            sampling=m["sampling_diagnostics"],
            sampling_summary=validate_counts(
                m["sampling_diagnostics"], "expanded", False, spec["steps_per_cell"]
            ),
            reports={},
        )
        for phase, values in phases.items():
            validate_development_report(values, data, SimulationConfig(**m["simulation"]))
            current_baseline = digest({k: v["baselines"]["buy_hold"] for k, v in values.items()})
            if baseline is not None and current_baseline != baseline:
                raise ValueError("identical matched holding paths required")
            baseline = current_baseline
            cell[phase] = group_metrics(values, stocks)
            cell[phase]["declining_stocks"] = declining_stock_summary(values, stocks)
            diagnosis = diagnose_groups(values, stocks, period="development")
            cell[phase]["action_execution"] = {
                "groups": diagnosis["groups"],
                "interpretation": diagnosis["interpretation"],
            }
            cell["reports"][phase] = dict(
                sha256=checksum(
                    run / ("selected-validation.json" if phase == "selected" else "validation.json")
                )
            )
        if (
            not math.isclose(
                cell["selected"]["score"], m["selected_score"], abs_tol=1e-12, rel_tol=0
            )
            or not math.isclose(
                m["selected_score"],
                max(r["score"] for r in m["validation_checkpoints"]),
                abs_tol=1e-12,
                rel_tol=0,
            )
            or not math.isclose(
                cell["final"]["score"],
                cell["learning_curve"][-1]["score"],
                abs_tol=1e-12,
                rel_tol=0,
            )
        ):
            raise ValueError("selected/final scores must agree with recorded checkpoints")
        cells[arm], reports[arm], metadata[arm] = cell, phases, m
    historical_spec = spec["historical_control_reproduction"]
    historical_state = json.loads((root / historical_spec["source_state"]).read_text())
    original_root = Path(historical_state["completed"][historical_spec["source_task"]]["root"])
    original_run, original, original_reports = run_reports(original_root)
    if (
        checksum(original_run / "metadata.json") != REFERENCE_METADATA_SHA256
        or checksum(original_run / "selected-validation.json")
        != historical_spec["selected_report_sha256"]
        or checksum(original_run / "validation.json") != historical_spec["final_report_sha256"]
    ):
        raise ValueError("pinned historical recurrent control metadata/reports required")
    # Revalidate the pinned historical accounting against the same original data.
    data = validate_payload(read_dataset(root / "data/datasets" / spec["dataset"]))
    for values in original_reports.values():
        validate_development_report(values, data, SimulationConfig(**original["simulation"]))
    reproduction = historical_reproduction(
        metadata["control"], original, reports["control"], original_reports
    )
    initial = validate_initial_records(
        {a: c["initialization_record"] for a, c in cells.items()}, False
    )
    if any(c["sampling"] != cells["control"]["sampling"] for c in cells.values()):
        raise ValueError("actual GAE symbol/group/year sampling counts must match")
    return dict(
        protocol_sha256=PROTOCOL_SHA256,
        cells=cells,
        screen=screen_gate(cells, reproduction),
        historical_control_reproduction=reproduction,
        initialization_checks=initial,
        source_proof=source,
        runtime_profile=profile,
        matched_contract=contract,
        validation_baseline_sha256=baseline,
        cleanup_verified=True,
        objective_achieved=False,
        test_evaluated=False,
        confirmation_allocated=False,
        training_performed_by_report=False,
        optimizer_steps=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("state", "source-proof", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = report(args.state, args.source_proof, Path.cwd())
    write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    render_report(result, args.output.with_suffix(".html"), title="Fixed recurrent GAE trace study")
    print(json.dumps(result["screen"], indent=2))


if __name__ == "__main__":
    main()
