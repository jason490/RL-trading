"""Validate owned CrossQ screens and report fixed gates without training."""

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
from rl_stocks.reporting.crossq_comparison import historical_reproduction, screen_gate
from rl_stocks.reporting.action_execution import diagnose_groups
from rl_stocks.reporting.history import render_report
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.runtime.colab.jobs import crossq_plan
from rl_stocks.training.curves import latest_points
from rl_stocks.training.crossq_spec import PROTOCOL_SHA256, protocol
from rl_stocks.training.crossq_initialization import validate_initial_comparison
from rl_stocks.training.crossq_context import validate_payload
from rl_stocks.training.crossq_restoration import verify_owned_run
from rl_stocks.training.crossq_runtime import (
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
    spec = protocol()
    state = json.loads(state_path.read_text())
    source = json.loads(source_path.read_text())
    profile = make_runtime_profile(root)
    ids = {f"crossq-{arm}-11" for arm in spec["arms"]}
    if (
        state["status"] != "complete"
        or state.get("cleanup_verified") is not True
        or set(state["completed"]) != ids
    ):
        raise ValueError("both completed owned-cleaned fixed screens required")
    data = validate_payload(read_dataset(root / "data/datasets" / spec["dataset"]))
    stocks = {e["symbol"] for e in data["episodes"].values()} - {"SPY", "QQQ", "IWM", "DIA"}
    if (len(state.get('sessions', {})) != len(ids)
            or {s['task'] for s in state['sessions'].values()} != ids
            or any(s['phase'] != 'done' for s in state['sessions'].values())):
        raise ValueError('both fixed owned sessions must be verified done')
    cells, reports, metadata = {}, {}, {}
    baseline = contract = None
    for arm in spec["arms"]:
        item = state["completed"][f"crossq-{arm}-11"]
        if state['sessions'].get(item['session']) != dict(task=f'crossq-{arm}-11', phase='done'):
            raise ValueError('exact owned screen task/session binding required')
        artifact = Path(item["root"])
        plan = crossq_plan(arm, profile=profile)
        archive = artifact.parent.parent / "artifacts.tar.gz"
        if archive.exists():
            validate_archive(archive, plan, spec["dataset"])
        else:
            with tempfile.TemporaryDirectory(prefix="crossq-retained-") as work:
                archive = Path(work) / "retained.tar"
                with tarfile.open(archive, "w") as stream:
                    stream.add(artifact, arcname="artifacts")
                validate_archive(archive, plan, spec["dataset"])
        run, m, phases = run_reports(artifact)
        if (
            m["git"]["revision"] != source["source_revision"]
            or m["git"]["dirty"]
            or m["source_hash"] != source["source_hash"]
            or m["crossq_arm"] != arm
            or m["seed"] != 11
            or m["smoke"] is not False
        ):
            raise ValueError("fixed clean CrossQ screen source/arm required")
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
            raise ValueError("same CrossQ screen source/simulation/settings required")
        contract = current
        restored = verify_owned_run(run, artifact / "frozen" / m["bundle_id"], arm)
        cell = dict(
            run=run.name,
            bundle_id=m["bundle_id"],
            seed=11,
            learning_curve=latest_points(m["validation_checkpoints"]),
            initialization_record=m["crossq_initialization"],
            restoration=restored,
            duration_seconds=m["duration_seconds"],
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
    if checksum(original_run / "metadata.json") != REFERENCE_METADATA_SHA256:
        raise ValueError("pinned historical native SAC metadata required")
    for phase, values in original_reports.items():
        filename = 'selected-validation.json' if phase == 'selected' else 'validation.json'
        if checksum(original_run / filename) != historical_spec[phase + '_report_sha256']:
            raise ValueError('pinned historical SAC symbol report required')
        validate_development_report(values, data, SimulationConfig(**original["simulation"]))
    reproduction = historical_reproduction(
        metadata["sac"], original, reports["sac"], original_reports
    )
    initial = validate_initial_comparison(
        {a: c["initialization_record"] for a, c in cells.items()}, False
    )
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
    render_report(result, args.output.with_suffix(".html"), title="Fixed SAC/CrossQ study")
    print(json.dumps(result["screen"], indent=2))


if __name__ == "__main__":
    main()
