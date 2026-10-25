"""Independently recheck every retained pre-update buffer from owned archives."""

import hashlib
import json

from rl_stocks.core.contracts import canonical
from rl_stocks.training.gae_rollouts import read_arrays, validate_arrays
from rl_stocks.training.gae_spec import parameters, PROTOCOL_SHA256


def validate_rollout_evidence(read, prefix, run, plan, names):
    context = plan["gae_study"]
    arm = context["arm"]
    smoke = context["smoke"]
    job = plan["jobs"][0]
    params = parameters(arm, smoke)
    n = params["n_steps"]
    steps = job["steps"]
    if prefix + "gae-rollout-failure.json" in names:
        raise RuntimeError("failed GAE target evidence cannot be accepted")
    records = [json.loads(line) for line in read(prefix + "gae-rollouts.jsonl").splitlines()]
    if len(records) != steps // n:
        raise RuntimeError("complete actual GAE rollout series required")
    expected_files = set()
    for i, record in enumerate(records):
        filename = f"gae-rollouts/rollout-{(i + 1) * n:06d}.npz"
        expected_files.add(prefix + filename)
        raw = read(prefix + filename)
        try:
            expected = validate_arrays(read_arrays(raw), arm, smoke)
        except AssertionError as exc:
            raise RuntimeError("actual GAE rollout target arithmetic mismatch") from exc
        expected.update(
            protocol_sha256=PROTOCOL_SHA256,
            arm=arm,
            steps=(i + 1) * n,
            rollout_index=i + 1,
            arrays_file=filename,
            arrays_sha256=hashlib.sha256(raw).hexdigest(),
            before_optimizer_update=True,
            sb3_nominal_epoch_counter=i * params["n_epochs"],
        )
        if canonical(record) != canonical(expected):
            raise RuntimeError("actual GAE array/step/math evidence mismatch")
    actual = {name for name in names if name.startswith(prefix + "gae-rollouts/")}
    if actual != expected_files:
        raise RuntimeError("exact retained GAE array file set required")
    summary = dict(
        protocol_sha256=PROTOCOL_SHA256,
        arm=arm,
        rollouts=len(records),
        transitions=steps,
        all_checked_before_optimizer=True,
        all_actual_arrays_retained=True,
        maximum_return_error=max(r["maximum_return_error"] for r in records),
        maximum_advantage_error=max(r["maximum_advantage_error"] for r in records),
        candidate_mc_identity_checked=arm == "mc_trace",
    )
    if canonical(run.get("gae_rollouts")) != canonical(summary):
        raise RuntimeError("actual GAE rollout summary mismatch")
    return dict(
        actual_rollout_arrays_verified=True,
        rollouts=len(records),
        transitions=steps,
        candidate_mc_identity_checked=arm == "mc_trace",
    )
