"""Read-only paired trace variance on accepted actual training buffers."""

import hashlib
import json
from pathlib import Path

import numpy as np

from rl_stocks.core.contracts import digest
from rl_stocks.runtime.colab.gae_rollout_artifacts import validate_rollout_evidence
from rl_stocks.training.gae_rollouts import read_arrays

PROTOCOL_SHA256 = "6dd450bb05d716524a951b929bbcc32be88ec6a954bdcc840ecf9472ad663b68"


def protocol(root):
    value = json.loads((Path(root) / "configs/experiments/gae_variance_audit.json").read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError("fixed saved GAE variance audit required")
    return value


def trace(arrays, gamma, lam):
    rewards = arrays["rewards"][:, 0].astype(np.float64)
    values = arrays["values"][:, 0].astype(np.float64)
    starts = arrays["episode_starts"][:, 0]
    advantages = np.empty(len(rewards), dtype=np.float64)
    carry = 0.0
    for i in reversed(range(len(rewards))):
        next_value = arrays["last_values"][0] if i == len(rewards) - 1 else values[i + 1]
        continuing = 1 - (arrays["dones"][0] if i == len(rewards) - 1 else starts[i + 1])
        delta = rewards[i] + gamma * float(next_value) * float(continuing) - values[i]
        carry = delta + gamma * lam * float(continuing) * carry
        advantages[i] = carry
    return advantages, advantages + values


def components(arrays, gamma):
    """Segment-wise discounted dot products, independent of the trace recurrence."""
    raw = arrays["raw_rewards"][:, 0].astype(np.float64)
    dones = arrays["step_dones"][:, 0]
    rewards, endpoint, rounding = np.empty(len(raw)), np.empty(len(raw)), np.empty(len(raw))
    residual = arrays["rewards"][:, 0].astype(np.float64) - raw
    residual -= gamma * arrays["timeout_values"][:, 0].astype(np.float64) * arrays["timeouts"][:, 0]
    start = 0
    ends = sorted(set(np.flatnonzero(dones).tolist() + [len(raw) - 1]))
    for end in ends:
        if dones[end]:
            tail = float(arrays["timeout_values"][end, 0]) if arrays["timeouts"][end, 0] else 0.0
        else:
            tail = float(arrays["last_values"][0]) if not arrays["dones"][0] else 0.0
        for i in range(start, end + 1):
            weights = gamma ** np.arange(end - i + 1, dtype=np.float64)
            rewards[i] = weights @ raw[i : end + 1]
            rounding[i] = weights @ residual[i : end + 1]
            endpoint[i] = gamma ** (end - i + 1) * tail
        start = end + 1
    return rewards, endpoint, rounding


def ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def statistics(arrays, gamma=0.99):
    a95, r95 = trace(arrays, gamma, 0.95)
    a1, r1 = trace(arrays, gamma, 1.0)
    raw, endpoint, rounding = components(arrays, gamma)
    error = float(np.max(np.abs(raw + endpoint + rounding - r1)))
    if error > 1e-12:
        raise ValueError("lambda1 raw reward/endpoint identity failed")
    av95, av1 = float(np.var(a95)), float(np.var(a1))
    rv95, rv1 = float(np.var(r95)), float(np.var(r1))
    return dict(
        raw_reward_variance=float(np.var(arrays["raw_rewards"].astype(np.float64))),
        advantage_variance_095=av95,
        advantage_variance_1=av1,
        advantage_variance_ratio=ratio(av1, av95),
        return_variance_095=rv95,
        return_variance_1=rv1,
        return_variance_ratio=ratio(rv1, rv95),
        mean_abs_raw_reward_component=float(np.mean(np.abs(raw))),
        mean_abs_endpoint_component=float(np.mean(np.abs(endpoint))),
        mean_abs_rounding_component=float(np.mean(np.abs(rounding))),
        maximum_endpoint_identity_error=error,
        timeouts=int(np.count_nonzero(arrays["timeouts"])),
        true_terminations=int(np.count_nonzero(arrays["step_dones"] & ~arrays["timeouts"])),
    )


def aggregate(rows):
    result = dict(rollouts=len(rows))
    for field in ["advantage_variance_ratio", "return_variance_ratio"]:
        values = [r[field] for r in rows if r[field] is not None]
        result[field] = dict(
            defined=len(values),
            zero_denominators=len(rows) - len(values),
            median=None if not values else float(np.median(values)),
            fraction_above_one=None if not values else sum(v > 1 for v in values) / len(values),
        )
    for field in [
        "raw_reward_variance", "advantage_variance_095", "advantage_variance_1",
        "return_variance_095", "return_variance_1", "mean_abs_raw_reward_component",
        "mean_abs_endpoint_component", "mean_abs_rounding_component",
    ]:
        result["mean_" + field] = float(np.mean([r[field] for r in rows]))
    for field in ["timeouts", "true_terminations"]:
        result[field] = sum(r[field] for r in rows)
    result["maximum_endpoint_identity_error"] = max(r["maximum_endpoint_identity_error"] for r in rows)
    return result


def audit(root):
    root = Path(root)
    spec = protocol(root)
    summary_path = root / spec["source_summary"]
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(summary_path) != spec["source_summary_sha256"]:
        raise ValueError("accepted fixed GAE source summary required")
    summary = json.loads(summary_path.read_text())
    state_path = root / spec["source_state"]
    state = json.loads(state_path.read_text())
    ids = {f"gae-{a}-11" for a in spec["arms"]}
    if (
        state.get("status") != "complete"
        or state.get("cleanup_verified") is not True
        or set(state["completed"]) != ids
        or len(state["sessions"]) != 2
        or {v["task"] for v in state["sessions"].values()} != ids
        or any(v["phase"] != "done" for v in state["sessions"].values())
        or summary["protocol_sha256"] != spec["source_protocol_sha256"]
        or summary["test_evaluated"] is not False
    ):
        raise ValueError("accepted owned-cleaned GAE training sources required")
    protected = {str(summary_path): sha(summary_path), str(state_path): sha(state_path)}
    ledger = root / ".runtime/usage-ledger.json"
    protected[str(ledger)] = sha(ledger)
    arms = {}
    for arm in spec["arms"]:
        cell = summary["cells"][arm]
        item = state["completed"][f"gae-{arm}-11"]
        if state["sessions"][item["session"]] != dict(task=f"gae-{arm}-11", phase="done"):
            raise ValueError("owned source session mismatch")
        run = Path(item["root"]) / "runs" / cell["run"]
        original = cell["restoration"]["protected_source_sha256"]
        for path, expected in original.items():
            if sha(Path(path)) != expected:
                raise ValueError("accepted source file changed")
            protected[path] = expected
        if str(run / "metadata.json") not in original:
            raise ValueError("accepted source run location required")
        metadata = json.loads((run / "metadata.json").read_text())
        names = {str(p.relative_to(run)) for p in run.rglob("*") if p.is_file()}
        validate_rollout_evidence(
            lambda name: (run / name).read_bytes(), "", metadata,
            dict(gae_study=dict(arm=arm, smoke=False), jobs=[dict(steps=100000)]), names,
        )
        rows = []
        for i in range(1, spec["rollouts_per_arm"] + 1):
            filename = f"gae-rollouts/rollout-{i * spec['transitions_per_rollout']:06d}.npz"
            arrays = read_arrays((run / filename).read_bytes())
            rows.append(dict(rollout=i, **statistics(arrays, spec["gamma"])))
        arms[arm] = dict(
            run=cell["run"], rows=rows,
            blocks={name: aggregate(rows[first - 1 : last]) for name, (first, last) in spec["blocks"].items()},
        )
    if any(sha(Path(p)) != expected for p, expected in protected.items()):
        raise ValueError("read-only audit changed protected evidence or ledger")
    return dict(
        protocol_sha256=PROTOCOL_SHA256, source_summary_sha256=spec["source_summary_sha256"],
        arms=arms, protected_files=len(protected), protected_files_unchanged=True,
        optimizer_steps=0, new_colab_units=0, model_or_market_replay=False,
        retired_test_access=False, objective_achieved=False, limits=spec["limits"],
    )
