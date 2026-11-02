"""Post-shutdown candidate acceptance and unchanged fitted gate; never learns."""

import json
from pathlib import Path
import time

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once
from rl_stocks.core.provenance import repository_state
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.evaluation.canonical_diagnostic import inference_only
from rl_stocks.evaluation.canonical_task import evaluate_canonical_training
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.evaluation.task_gates import training_gate
from rl_stocks.evaluation.task_reference_controls import store_report
from rl_stocks.inference.canonical_bundle import FrozenCanonicalTaskPolicy, freeze_canonical_task
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.runtime.colab.jobs import task_reference_candidate_plan
from rl_stocks.training.task_conditioning import (
    PROTOCOL_SHA256 as BASE_SHA256,
    protocol as base_protocol,
)
from rl_stocks.training.task_reference_assets import verify_asset_case
from rl_stocks.training.task_reference_runtime import (
    CLAIM,
    REFERENCE_SHA256,
    REVIEW_SHA256,
    checksum,
    execution_spec,
)


def reference_fitted_gate(controls, candidate):
    role = "tagged_routed"
    points = {k: v["final"] for k, v in controls.items()}
    points[role] = candidate
    if set(points) != {"masked_shared", "tagged_shared", "tagged_routed"}:
        raise ValueError("complete three-role reference fitted matrix required")
    cells = {}
    for name, point in points.items():
        if (
            point["role"] != name
            or point["protocol_sha256"] != REFERENCE_SHA256
            or point["phase"] != "final"
            or point["steps"] != 100000
            or point["period"] != "training-window"
            or point["performance_execution"] != "reference-actor-v1"
            or point["seed"] != 11
        ):
            raise ValueError("fixed final reference metrics required")
        # Translate only the gate's identity field after validating the explicit
        # numerical contract; no persisted point is relabeled as native output.
        cells[name] = {
            "steps": 100000,
            "phase": "final",
            "period": "training-window",
            "seed": 11,
            "protocol_sha256": BASE_SHA256,
            "holding_path_sha256": point["holding_path_sha256"],
            "metrics": point["metrics"],
        }
    verdict = training_gate(cells)
    return {
        "schema": "task-reference-fitted-gate-v1",
        "reference_protocol_sha256": REFERENCE_SHA256,
        "original_gate": verdict,
        "passed": verdict["passed"],
        "development_allowed": verdict["development_allowed"],
        "reference_points": points,
        "performance_promotion": False,
        "objective_achieved": False,
    }


def accept(root, state_path, output):
    from rl_stocks.runtime.colab.budget import Ledger

    root, output = Path(root).resolve(), Path(output).resolve()
    spec = execution_spec(root)
    state = json.loads(Path(state_path).read_text())
    ledger = Ledger(root / ".runtime/usage-ledger.json")
    if (
        state["status"] != "complete"
        or state.get("cleanup_verified") is not True
        or len(state["completed"]) != 1
    ):
        raise ValueError("complete retrieved transport and verified cleanup required")
    completed = next(iter(state["completed"].values()))
    session = completed["session"]
    attempts = [a for a in ledger.value["attempts"] if a["session"] == state["fleet"]]
    if (
        len(attempts) != 1
        or attempts[0]["status"] != "stopped"
        or set(attempts[0]["children"]) != {session}
        or attempts[0]["children"][session]["status"] != "stopped"
        or attempts[0]["children"][session].get("cleanup_verified") is not True
    ):
        raise ValueError("exact stopped owned candidate required before local acceptance")
    claim = json.loads((ledger.path.parent / CLAIM).read_text())
    if claim["session"] != session or claim["review_sha256"] != REVIEW_SHA256:
        raise ValueError("exact original one-time candidate claim required")
    archive = root / "artifacts/colab" / session / "artifacts.tar.gz"
    completion = completed["completion"]
    if checksum(archive) != completed["archive_sha256"] or canonical(
        completion["plan"]
    ) != canonical(task_reference_candidate_plan()):
        raise ValueError("bound completed candidate transport required")
    captured = json.loads((root / ".runtime/payloads" / session / "source-git.json").read_text())
    if captured != completion["source_git"]:
        raise ValueError("actual committed candidate source mismatch")
    controls_path = root / "results/task-conditioning-2026-10-31/reference-controls.json"
    if checksum(controls_path) != spec["control_results_sha256"]:
        raise ValueError("original reference controls changed")
    controls = json.loads(controls_path.read_text())
    source = repository_state(root)
    if source["dirty"]:
        raise ValueError("clean committed offline verifier required")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        dataset = read_dataset(root / "data/datasets" / base_protocol()["dataset"])
        origin = {
            "session": session,
            "archive_sha256": checksum(archive),
            "source_revision": captured["revision"],
            "source_hash": completion["source_hash"],
        }
        print("Post-shutdown physical candidate verification", flush=True)
        physical = verify_asset_case(root, dataset, role="tagged_routed", origin=origin)
        (output / "physical.json").write_bytes(canonical(physical))
        if (
            not completion["initial_parity"]["all_mapped_arrays_exact"]
            or physical["checkpoints"][0]["policy_sha256"]
            != completion["initial_parity"]["routed_policy_sha256"]
        ):
            raise ValueError("actual initial common-start parity missing")
        stocks = set(dataset["manifest"]["evaluation_symbols"]) - {"SPY", "QQQ", "IWM", "DIA"}
        phases, cached = {}, {}
        with inference_only():
            for phase in ("selected", "final"):
                native_id = physical["frozen_bundles"][phase]
                native = (
                    root
                    / "artifacts/colab"
                    / session
                    / "retrieved/artifacts/task-screen"
                    / f"frozen-{phase}"
                    / native_id
                )
                path = freeze_canonical_task(native, output / "frozen" / phase)
                frozen = FrozenCanonicalTaskPolicy(
                    path,
                    feature_schema=dataset["schema"],
                    feed=dataset["manifest"]["provenance"]["capabilities"]["feed"],
                )
                model_sha = frozen.manifest["source_manifest"]["model_sha256"]
                print("Post-shutdown reference candidate:", phase, flush=True)
                if model_sha in cached:
                    results, paths, html = cached[model_sha]
                    reused = True
                else:
                    evidence = frozen.manifest["source_manifest"]["metadata"]["task_contract"][
                        "type_manifest"
                    ]
                    results = evaluate_canonical_training(frozen, dataset, evidence)
                    html = output / f"{phase}.html"
                    paths = store_report(results, html)
                    cached[model_sha] = results, paths, html
                    reused = False
                point = {
                    "role": "tagged_routed",
                    "phase": phase,
                    "period": "training-window",
                    "seed": 11,
                    "steps": physical["selected_step"] if phase == "selected" else 100000,
                    "protocol_sha256": REFERENCE_SHA256,
                    "performance_execution": "reference-actor-v1",
                    "selection_execution": "retained-native-training-curve",
                    "native_bundle_id": native_id,
                    "reference_bundle_id": path.name,
                    "model_sha256": model_sha,
                    "metrics": group_metrics(results, stocks),
                    "short_side": declining_stock_summary(results, stocks),
                    "results_sha256": digest(results),
                    "identical_checkpoint_reused": reused,
                    "holding_path_sha256": digest(
                        {k: r["baselines"]["buy_hold"] for k, r in results.items()}
                    ),
                    "paths_path": str(paths.relative_to(root)),
                    "paths_sha256": checksum(paths),
                    "report_path": str(html.relative_to(root)),
                    "report_sha256": checksum(html),
                }
                if point["holding_path_sha256"] != controls["holding_path_sha256"]:
                    raise ValueError("candidate and reference controls holding paths differ")
                phases[phase] = point
                (output / f"{phase}.json").write_bytes(canonical(point))
        gate = reference_fitted_gate(controls["controls"], phases["final"])
        result = {
            "schema": "task-reference-candidate-acceptance-v1",
            "review_sha256": REVIEW_SHA256,
            "reference_protocol_sha256": REFERENCE_SHA256,
            "session": session,
            "source": source,
            "training_source": captured,
            "archive_sha256": checksum(archive),
            "physical": physical,
            "candidate": phases,
            "fitted_gate": gate,
            "cleanup_verified": True,
            "post_stop_elapsed_seconds": time.monotonic() - started,
            "performance_promotion": False,
            "development_evaluated": False,
            "retired_test_access": False,
            "native_acceptance_changed": False,
        }
        if not gate["passed"]:
            closure = {
                "schema": "task-reference-family-closure-v1",
                "reference_protocol_sha256": REFERENCE_SHA256,
                "reason": "Complete seed11 reference final matrix fails original fitted gate",
                "gate_sha256": digest(gate),
                "additional_training_allowed": False,
                "development_allowed": False,
                "automatic_retry_allowed": False,
                "objective_achieved": False,
            }
            write_once(root / ".runtime/task-reference-family-closure-v1.json", canonical(closure))
            write_once(
                root / ".runtime/task-conditioning-family-closure-v1.json", canonical(closure)
            )
        (output / "result.json").write_bytes(canonical(result))
        return result
    except BaseException as error:
        (output / "failure.json").write_bytes(
            canonical(
                {
                    "schema": "task-reference-offline-failure-v1",
                    "session": session,
                    "error_type": type(error).__name__,
                    "cleanup_previously_verified": True,
                    "performance_promotion": False,
                }
            )
        )
        raise
