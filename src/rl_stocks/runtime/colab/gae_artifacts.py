"""GAE archive semantics; physical model restoration is checked independently."""

from dataclasses import asdict
import hashlib
import json
import math

from rl_stocks.core.contracts import canonical, digest
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
from rl_stocks.training.gae_spec import PROTOCOL_SHA256, protocol
from rl_stocks.training.gae_initialization import (
    initialization_spec,
    initialization_entry,
    validate_record,
)
from rl_stocks.features.registry.assemble import NAMES
from rl_stocks.runtime.colab.gae_rollout_artifacts import validate_rollout_evidence


def validate_gae_artifacts(read, prefix, folder, run, plan, names):
    context = plan["gae_study"]
    arm, seed, smoke = context["arm"], context["seed"], context["smoke"]
    job = plan["jobs"][0]
    frozen = json.loads(read(folder + "manifest.json"))
    spec = protocol()
    duration = run.get("duration_seconds")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("positive actual GAE run duration required")
    if (
        run.get("gae_arm") != arm
        or run.get("study_protocol_sha256") != PROTOCOL_SHA256
        or canonical(run.get("action_execution"))
        != canonical(execution_contract("unrestricted", "unit_magnitude"))
        or canonical(run.get("action_distribution")) != canonical(distribution_spec())
        or canonical(run.get("simulation")) != canonical(asdict(SimulationConfig()))
    ):
        raise RuntimeError("fixed GAE arm/execution/distribution/simulation required")
    proof = initialization_spec(seed, smoke)
    if canonical(run.get("gae_shared_initialization")) != canonical(proof) or canonical(
        plan["initialization_contract"]
    ) != canonical(proof):
        raise RuntimeError("actual GAE initialization provenance mismatch")
    read_initialization(read(prefix + "gae-initialization.npz"), initialization_entry(seed, smoke))
    validate_record(run["gae_initialization"], arm, smoke, initial=True, seed=seed)
    validate_record(run["gae_final"], arm, smoke, seed=seed)
    validate_record(run["gae_selected"], arm, smoke, selected_steps=run["selected_step"], seed=seed)
    if (
        run["gae_initialization"]["policy_tensor_sha256"]
        == run["gae_final"]["policy_tensor_sha256"]
    ):
        raise RuntimeError("GAE training did not change weights")
    counts = run["training_execution"]
    validate_execution_counts(counts, "posthoc_unit", job["steps"])
    if counts != json.loads(read(prefix + "training-execution.json")):
        raise RuntimeError("actual GAE training execution sidecar mismatch")
    sampling = run["sampling_diagnostics"]
    validate_counts(sampling, "expanded", smoke, job["steps"])
    years = validate_year_counts(sampling.get("year_counts", {}), sampling["starts"], job["steps"])
    if any(
        not 2010 <= int(y) <= 2020
        for field in ("start_years", "decision_years", "execution_years")
        for y, n in years[field].items()
        if n
    ):
        raise RuntimeError("training-only GAE sampling years required")
    if run.get("start_sampling") != sampling_summary("expanded", smoke) or sampling != json.loads(
        read(prefix + "training-sampling.json")
    ):
        raise RuntimeError("actual GAE sampling sidecar mismatch")
    for filename in ("selected-validation.json", "validation.json"):
        reports = json.loads(read(prefix + filename))
        validate_execution_report(reports, "posthoc_unit")
        if (
            len(reports) != 26
            or {r["symbol"] for r in reports.values()} != set(DEVELOPMENT_SYMBOLS)
            or any(r["split"] != "validation" for r in reports.values())
        ):
            raise RuntimeError("all26 matched GAE development symbols required")
    if (
        run.get("dataset") != spec["dataset"]
        or frozen["metadata"] != {k: v for k, v in run.items() if k != "bundle_id"}
        or frozen["names"] != list(NAMES)
        or frozen["feature_schema"] != spec["feature_schema"]
        or digest(frozen["normalizer"]) != spec["normalizer_sha256"]
        or "clip_abs" in frozen["normalizer"]
        or frozen["data_manifest"]["included_splits"] != ["train", "validation"]
        or frozen["model_sha256"] != hashlib.sha256(read(prefix + "best.zip")).hexdigest()
        or read(prefix + "best.zip") != read(folder + "model.zip")
    ):
        raise RuntimeError("GAE frozen model/data/preprocessing contract mismatch")
    if "vix_context" in frozen["data_manifest"]:
        raise RuntimeError("original unaugmented GAE payload required")
    validate_rollout_evidence(read, prefix, run, plan, names)
    read(prefix + "latest.zip")
