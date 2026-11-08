"""Exact full economic-path parity; frozen inference only, never model selection."""

import json
import math
from pathlib import Path

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.environment.observations.state import portfolio_state
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.evaluation.metrics import metrics
from rl_stocks.evaluation.direction import directional_attribution
from rl_stocks.evaluation.training_fit import training_window
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.canonical_actor import encode32
from rl_stocks.inference.canonical_tanh_policy import CanonicalTanhPolicy
from rl_stocks.inference.tanh_bundle import FrozenTanhPolicy

PROTOCOL_SHA256 = "f81d5de49c3206ae92272accf1c35db154cf997ad8a59177b5807ebb68668834"


def protocol():
    path = Path(__file__).resolve().parents[3] / "configs/experiments/tanh_full_paths.json"
    value = json.loads(path.read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError("fixed full-path engineering protocol required")
    return value


def lossless(value):
    """Preserve every binary64 bit, including signed zero; reject nonfinite state."""
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("finite economic state required")
        return {"float64": value.hex()}
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("string state fields required")
        return {k: lossless(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [lossless(v) for v in value]
    raise ValueError("unsupported economic state type")


def snapshot(env):
    p = env.sim.portfolio
    return {"cash": p.cash, "quantity": p.quantity, "basis": p.basis,
            "age": p.age, "costs": p.costs, "turnover": p.turnover,
            "protection": env.sim.protection, "near_done": env.sim.near_done,
            "last_protective_action": env.sim.last_protective_action,
            "event_count": len(p.events), "index": env.i, "done": env.done}


def step_record(step, env):
    observation, reward, terminated, truncated, info = step
    return {"next_observation_hex": [encode32(v) for v in observation],
            "reward": reward, "terminated": terminated, "truncated": truncated,
            "info": info, "state": snapshot(env)}


def paired_episode(episode, dataset, frozen, direct, output):
    """Run two independent accounts and preserve exact per-decision evidence."""
    if episode["split"] != "train" or any(b["session"] >= "2021-01-01" for b in episode["bars"]):
        raise ValueError("training-only engineering replay required")
    output = Path(output)
    environments = {}
    rows = []
    count = 0
    try:
        for name in ("direct", "frozen"):
            environments[name] = TradingEnv([episode], normalizer_from(dataset["normalizer"]),
                                             dataset["names"], observation_profile="execution_full")
            environments[name].reset(seed=11, options={"episode": 0})
        left, right = environments["direct"], environments["frozen"]
        with output.open("xb") as stream:
            while True:
                bar = right.bars[right.i]
                state = portfolio_state(right.sim, bar.close, episode["atr"][right.i], "execution_full")
                prepared = frozen.observation(episode["features"][right.i], dataset["names"], state)
                obs = left.observation()
                if [encode32(v) for v in obs] != [encode32(v) for v in prepared]:
                    raise ValueError("full-path raw preprocessing mismatch")
                action = direct.predict(obs)
                bundled = frozen.predict_features(
                    episode["features"][right.i], dataset["names"], state, symbol=episode["symbol"],
                    available_time=bar.available_time, decision_time=bar.available_time,
                    now=bar.available_time)
                if [encode32(v) for v in action] != [encode32(v) for v in bundled]:
                    raise ValueError("full-path frozen action mismatch")
                a, b = left.step(action), right.step(bundled)
                record = step_record(a, left)
                if canonical(lossless(record)) != canonical(lossless(step_record(b, right))):
                    raise ValueError("full-path accounting/event/state mismatch")
                evidence = {"observation_hex": [encode32(v) for v in obs],
                            "action_hex": [encode32(v) for v in action], **record}
                stream.write(canonical(lossless(evidence)) + b"\n")
                rows.append(a[4])
                count += 1
                if a[2] or a[3]:
                    break
            if count != len(episode["bars"]) - 1 and not (a[2] and a[4]["equity"] <= 0):
                raise ValueError("incomplete economic replay")
        results = [metrics(rows, e.sim.portfolio.events) for e in (left, right)]
        if canonical(lossless(results[0])) != canonical(lossless(results[1])):
            raise ValueError("full-path final metrics mismatch")
        return {"symbol": episode["symbol"], "decisions": count,
                "path_sha256": checksum(output), "metrics": results[0],
                "final_state": lossless(snapshot(left)),
                "directional": directional_attribution(episode, rows, left.sim.portfolio.events),
                "exact_paired_path": True}
    finally:
        for env in environments.values():
            env.close()


def run(dataset, bundle_root, output):
    from rl_stocks.training.trpo_smoke import validate_dataset
    validate_dataset(dataset)
    spec = protocol()
    root = Path(__file__).resolve().parents[3]
    if checksum(root / "results/trpo-smoke-2026-11-08/frozen-preprocessing.json") != spec["frozen_receipt_sha256"]:
        raise ValueError("accepted frozen preprocessing receipt required")
    view = training_window(dataset, original_panel_only=True)
    if (digest(view["episodes"]) != spec["training_window_episodes_sha256"]
            or len(view["episodes"]) != spec["episodes_per_case"]
            or sum(len(e["bars"]) - 1 for e in view["episodes"].values()) != spec["maximum_decisions_per_case"]):
        raise ValueError("complete fixed training-panel economic replay required")
    original = digest(dataset)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    normalizer_before = canonical(dataset["normalizer"])
    cases = []
    for case in spec["cases"]:
        bundle = Path(bundle_root) / case["case"] / case["bundle_id"]
        frozen = FrozenTanhPolicy(bundle, feature_schema=dataset["schema"],
                                  feed=dataset["manifest"]["provenance"]["capabilities"]["feed"],
                                  expected_bundle_id=case["bundle_id"])
        if (checksum(bundle / "actor.json") != case["actor_sha256"]
                or frozen.manifest["model_sha256"] != case["source_model_sha256"]):
            raise ValueError("fixed accepted learned actor required")
        protected = {p.name: checksum(p) for p in bundle.iterdir()}
        direct = CanonicalTanhPolicy.load(bundle / "actor.json")
        folder = output / case["case"]
        folder.mkdir()
        rows = []
        for key, episode in sorted(view["episodes"].items()):
            row = paired_episode(episode, view, frozen, direct, folder / (key + ".jsonl"))
            rows.append(row)
            (folder / "progress.json").write_bytes(canonical({"episodes": rows}))
            print(case["case"], episode["symbol"], "exact decisions", row["decisions"], flush=True)
        if (protected != {p.name: checksum(p) for p in bundle.iterdir()}
                or canonical(dataset["normalizer"]) != normalizer_before):
            raise ValueError("frozen inputs mutated during economic replay")
        case_result = {"case": case["case"], "bundle_id": bundle.name,
                       "episodes": rows, "decisions": sum(r["decisions"] for r in rows),
                       "paths_digest": digest([r["path_sha256"] for r in rows])}
        (folder / "result.json").write_bytes(canonical(case_result))
        cases.append(case_result)
        print(case["case"], "exact episodes", len(rows), "decisions", case_result["decisions"], flush=True)
    if digest(dataset) != original:
        raise ValueError("economic replay mutated source dataset")
    result = {"schema": "tanh-full-path-parity-result-v1", "protocol_sha256": PROTOCOL_SHA256,
              "dataset": original, "cases": cases, "source_dataset_unchanged": True,
              "training_performed": False, "development_evaluated": False,
              "retired_test_access": False, "performance_promotion": False}
    (output / "result.json").write_bytes(canonical(result))
    return result
