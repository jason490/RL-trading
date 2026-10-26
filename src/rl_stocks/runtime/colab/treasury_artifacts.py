"""Treasury archive semantics; physical model restoration is checked independently."""

from dataclasses import asdict
import hashlib
import json
import math

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.data.generation.treasury_contract import validate_frozen_contract
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.environment.sampling.years import validate_year_counts
from rl_stocks.risk.exposure import execution_contract
from rl_stocks.rl.registry.distribution import distribution_spec
from rl_stocks.runtime.colab.distributional_artifacts import DEVELOPMENT_SYMBOLS
from rl_stocks.training.diversity import validate_counts, sampling_summary
from rl_stocks.training.recurrent_alignment import (
    validate_execution_counts,
    validate_execution_report,
)
from rl_stocks.training.shared_initialization import read_initialization
from rl_stocks.training.treasury_spec import PROTOCOL_SHA256, BINDING_SHA256, binding
from rl_stocks.training.treasury_initialization import (
    initialization_spec,
    initialization_entry,
    validate_record,
)


def validate_treasury_artifacts(read, prefix, folder, run, plan):
    context = plan["treasury_study"]
    arm, seed, smoke = context["arm"], context["seed"], context["smoke"]
    job = plan["jobs"][0]
    frozen = json.loads(read(folder + "manifest.json"))
    spec = binding()
    contract = spec["dataset_contracts"][arm]
    duration = run.get("duration_seconds")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("positive actual Treasury run duration required")
    if (
        run.get("treasury_arm") != arm
        or run.get("study_protocol_sha256") != PROTOCOL_SHA256
        or run.get("treasury_binding_sha256") != BINDING_SHA256
        or run.get("action_execution") != execution_contract("unrestricted", "unit_magnitude")
        or canonical(run.get("action_distribution")) != canonical(distribution_spec())
        or canonical(run.get("simulation")) != canonical(asdict(SimulationConfig()))
    ):
        raise RuntimeError("fixed Treasury arm/execution/distribution/simulation required")
    proof = initialization_spec(arm, seed, smoke)
    if canonical(run.get("treasury_shared_initialization")) != canonical(proof) or canonical(
        plan["initialization_contract"]
    ) != canonical(proof):
        raise RuntimeError("actual Treasury initialization provenance mismatch")
    read_initialization(
        read(prefix + "treasury-initialization.npz"), initialization_entry(arm, seed, smoke)
    )
    validate_record(run["treasury_initialization"], arm, smoke, initial=True, seed=seed)
    validate_record(run["treasury_final"], arm, smoke, seed=seed)
    validate_record(run["treasury_selected"], arm, smoke, selected_steps=run["selected_step"], seed=seed)
    if (
        run["treasury_initialization"]["policy_tensor_sha256"]
        == run["treasury_final"]["policy_tensor_sha256"]
    ):
        raise RuntimeError("Treasury training did not change weights")
    counts = run["training_execution"]
    validate_execution_counts(counts, "posthoc_unit", job["steps"])
    if counts != json.loads(read(prefix + "training-execution.json")):
        raise RuntimeError("actual Treasury training execution sidecar mismatch")
    sampling = run["sampling_diagnostics"]
    validate_counts(sampling, "expanded", smoke, job["steps"])
    years = validate_year_counts(sampling.get("year_counts", {}), sampling["starts"], job["steps"])
    if any(
        not 2010 <= int(y) <= 2020
        for field in ("start_years", "decision_years", "execution_years")
        for y, n in years[field].items()
        if n
    ):
        raise RuntimeError("training-only Treasury sampling years required")
    if run.get("start_sampling") != sampling_summary("expanded", smoke) or sampling != json.loads(
        read(prefix + "training-sampling.json")
    ):
        raise RuntimeError("actual Treasury sampling sidecar mismatch")
    for filename in ("selected-validation.json", "validation.json"):
        reports = json.loads(read(prefix + filename))
        validate_execution_report(reports, "posthoc_unit")
        if (
            len(reports) != 26
            or {r["symbol"] for r in reports.values()} != set(DEVELOPMENT_SYMBOLS)
            or any(r["split"] != "validation" for r in reports.values())
        ):
            raise RuntimeError("all26 matched Treasury development symbols required")
    if (
        run.get("dataset") != spec["datasets"][arm]
        or frozen["metadata"] != {k: v for k, v in run.items() if k != "bundle_id"}
        or frozen["names"] != contract["names"]
        or frozen["feature_schema"] != contract["schema"]
        or digest(frozen["normalizer"]) != contract["normalizer_sha256"]
        or "clip_abs" in frozen["normalizer"]
        or frozen["data_manifest"]["included_splits"] != ["train", "validation"]
        or frozen["model_sha256"] != hashlib.sha256(read(prefix + "best.zip")).hexdigest()
        or read(prefix + "best.zip") != read(folder + "model.zip")
    ):
        raise RuntimeError("Treasury frozen model/data/preprocessing contract mismatch")
    if arm == "legacy":
        if "treasury_context" in frozen["data_manifest"]:
            raise RuntimeError("legacy Treasury context must be absent")
    elif (
        validate_frozen_contract(frozen) != arm
        or frozen["data_manifest"]["treasury_context"]["source_dataset"] != spec["datasets"]["legacy"]
        or frozen["data_manifest"]["treasury_context"]["source_context"] != spec["source_context"]
        or frozen["data_manifest"]["treasury_context"]["eligibility_sha256"]
        != contract["eligibility_sha256"]
        or frozen["data_manifest"]["treasury_context"]["extra_normalizer_fit_rows"]
        != contract["extra_normalizer_fit_rows"]
    ):
        raise RuntimeError("Treasury frozen source lineage mismatch")
    read(prefix + "latest.zip")
