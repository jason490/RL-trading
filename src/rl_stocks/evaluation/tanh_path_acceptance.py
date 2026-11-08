"""Read-only saved-path restoration and exact cross-machine comparison."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

from rl_stocks.core.contracts import Bar, canonical, digest
from rl_stocks.environment.execution.simulator import Simulator
from rl_stocks.environment.observations.state import portfolio_state
from rl_stocks.environment.rewards.equity import equity_reward
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.evaluation.direction import directional_attribution
from rl_stocks.evaluation.metrics import metrics
from rl_stocks.evaluation.tanh_paths import lossless, protocol, snapshot, PROTOCOL_SHA256
from rl_stocks.evaluation.training_fit import training_window
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.canonical_actor import _unique, encode32


def restore(value):
    if isinstance(value, dict):
        if set(value) == {"float64"}:
            number = float.fromhex(value["float64"])
            if not math.isfinite(number) or number.hex() != value["float64"]:
                raise ValueError("canonical finite binary64 evidence required")
            return number
        return {k: restore(v) for k, v in value.items()}
    if isinstance(value, list):
        return [restore(v) for v in value]
    if isinstance(value, float):
        raise ValueError("untagged float in lossless economic evidence")
    return value


def exact(left, right):
    return canonical(lossless(left)) == canonical(lossless(right))


def verify_episode(path, episode, dataset):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("bounded regular saved economic path required")
    normalizer = normalizer_from(dataset["normalizer"])
    sim = Simulator()
    rows, events = [], []
    previous_next = None
    final_state = None
    previous_equity = 100000.
    ended = False
    with path.open("rb") as stream:
        while line := stream.readline(1024 * 1024 + 1):
            index = len(rows)
            if ended or len(line) > 1024 * 1024 or index >= len(episode["bars"]) - 1:
                raise ValueError("saved path exceeds complete episode bounds")
            row = restore(json.loads(line, object_pairs_hook=_unique))
            if set(row) != {"observation_hex", "action_hex", "next_observation_hex", "reward",
                            "terminated", "truncated", "info", "state"}:
                raise ValueError("exact saved-path fields required")
            before, bar = episode["bars"][index:index + 2]
            fields = portfolio_state(sim, before["close"], episode["atr"][index], "execution_full")
            expected = normalizer.transform(episode["features"][index], dataset["names"]) + fields
            if row["observation_hex"] != [encode32(v) for v in expected]:
                raise ValueError("saved raw-feature/portfolio observation mismatch")
            if previous_next is not None and previous_next != row["observation_hex"]:
                raise ValueError("saved observation continuity mismatch")
            info, state = row["info"], row["state"]
            if (
                len(row["action_hex"]) != 4 or len(row["next_observation_hex"]) != 21
                or row["action_hex"] != [encode32(v) for v in info["raw_action"]]
                or info["session"] != bar["session"] or not exact(info["price"], bar["close"])
                or not exact(info["previous_equity"], previous_equity)
                or not exact(info["equity"], info["cash"] + info["quantity"] * bar["close"])
                or not exact(row["reward"], equity_reward(previous_equity, info["equity"]))
                or type(row["terminated"]) is not bool or type(row["truncated"]) is not bool
                or row["terminated"] != (info["equity"] <= 0)
                or row["truncated"] != (not row["terminated"] and index + 2 == len(episode["bars"]))
                or state["index"] != index + 1
            ):
                raise ValueError("saved action/accounting/reward/termination mismatch")
            events.extend(info["events"])
            if state["event_count"] != len(events):
                raise ValueError("saved event continuity mismatch")
            for name in ("cash", "quantity", "age", "costs", "turnover"):
                if not exact(state[name], info[name]):
                    raise ValueError("saved account state disagrees with ledger")
            event_start = len(sim.portfolio.events)
            replayed = sim.advance(info["raw_action"], Bar(**before), Bar(**bar), episode["atr"][index])
            replayed.update(session=bar["session"], price=bar["close"],
                            events=sim.portfolio.events[event_start:],
                            unscaled_reward=equity_reward(previous_equity, replayed["equity"]),
                            reward_kind="net_log_equity")
            expected_state = snapshot(SimpleNamespace(sim=sim, i=index + 1,
                                                      done=row["terminated"] or row["truncated"]))
            if not exact(replayed, info) or not exact(expected_state, state):
                raise ValueError("saved fills/protection/accounting fail action replay")
            next_fields = portfolio_state(sim, bar["close"], episode["atr"][index + 1], "execution_full")
            next_obs = normalizer.transform(episode["features"][index + 1], dataset["names"]) + next_fields
            if row["next_observation_hex"] != [encode32(v) for v in next_obs]:
                raise ValueError("saved next-state preprocessing mismatch")
            previous_next = row["next_observation_hex"]
            previous_equity = info["equity"]
            rows.append(info)
            final_state = state
            ended = row["terminated"] or row["truncated"]
            if state["done"] != ended:
                raise ValueError("saved episode completion state mismatch")
    if not rows or not ended:
        raise ValueError("incomplete saved economic episode")
    return {"symbol": episode["symbol"], "decisions": len(rows), "path_sha256": checksum(path),
            "metrics": metrics(rows, events), "final_state": lossless(final_state),
            "directional": directional_attribution(episode, rows, events), "exact_paired_path": True}


def verify_directory(folder, dataset):
    from rl_stocks.training.trpo_smoke import validate_dataset
    validate_dataset(dataset)
    spec = protocol()
    view = training_window(dataset, original_panel_only=True)
    if digest(view["episodes"]) != spec["training_window_episodes_sha256"]:
        raise ValueError("fixed original training panel required")
    folder = Path(folder)
    result = json.loads((folder / "result.json").read_text(), object_pairs_hook=_unique)
    if (result["schema"] != "tanh-full-path-parity-result-v1"
            or result["protocol_sha256"] != PROTOCOL_SHA256 or result["dataset"] != digest(dataset)
            or result["source_dataset_unchanged"] is not True
            or any(result[k] is not False for k in ("training_performed", "development_evaluated",
                                                    "retired_test_access", "performance_promotion"))
            or [c["case"] for c in result["cases"]] != [c["case"] for c in spec["cases"]]):
        raise ValueError("complete fixed inference-only path result required")
    for case, expected in zip(result["cases"], spec["cases"], strict=True):
        path = folder / case["case"]
        rows = [verify_episode(path / (key + ".jsonl"), ep, view)
                for key, ep in sorted(view["episodes"].items())]
        restored = {"case": case["case"], "bundle_id": expected["bundle_id"], "episodes": rows,
                    "decisions": sum(r["decisions"] for r in rows),
                    "paths_digest": digest([r["path_sha256"] for r in rows])}
        if not exact(restored, case) or not exact(restored, json.loads((path / "result.json").read_text())):
            raise ValueError("saved path metrics/summary restoration mismatch")
    return result


def compare(left, right, dataset):
    a, b = verify_directory(left, dataset), verify_directory(right, dataset)
    if not exact(a, b):
        raise ValueError("cross-machine complete economic paths differ")
    return {"schema": "tanh-cross-machine-path-acceptance-v1", "protocol_sha256": PROTOCOL_SHA256,
            "case_path_digests": [c["paths_digest"] for c in a["cases"]],
            "episodes": sum(len(c["episodes"]) for c in a["cases"]),
            "decisions": sum(c["decisions"] for c in a["cases"]),
            "exact_full_paths": True, "runtime_locations_verified": False,
            "performance_promotion": False}
