"""Replay fixed training windows with saved stochastic PPO; never optimize."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset, normalizer_from
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.recurrent_values import (
    protocol,
    windows,
    targets,
    error_metrics,
    PROTOCOL_SHA256,
)
from rl_stocks.rl.registry.factory import CLASSES
from rl_stocks.training.vix_context import model_record, validate_record, validate_payload
from rl_stocks.training.vix_restoration import same_restored


def collect(model, episode, data, spec, seed, deadline):
    """Mirror installed RecurrentPPO's stochastic collector without its counters."""
    env = TradingEnv(
        [episode],
        normalizer_from(data["normalizer"]),
        data["names"],
        SimulationConfig(),
        episode_steps=spec["window_transitions"],
        observation_profile=spec["observation_profile"],
        exposure_execution=spec["exposure_execution"],
    )
    torch.manual_seed(seed)
    shape = (model.policy.lstm_actor.num_layers, 1, model.policy.lstm_actor.hidden_size)
    states = RNNStates(
        pi=(torch.zeros(shape), torch.zeros(shape)), vf=(torch.zeros(shape), torch.zeros(shape))
    )
    values, rewards, clipped = [], [], []
    try:
        observation, _ = env.reset(seed=seed, options={"episode": 0})
        terminated = truncated = False
        with torch.no_grad():
            for step in range(spec["window_transitions"]):
                if time.monotonic() >= deadline:
                    raise TimeoutError("fixed value audit deadline exceeded")
                obs = model.policy.obs_to_tensor(observation)[0]
                starts = torch.tensor([step == 0], dtype=torch.float32)
                previous = deepcopy(states)
                raw, value, _, following = model.policy.forward(
                    obs, states, starts, deterministic=False
                )
                independent = model.policy.predict_values(obs, states.vf, starts)
                torch.testing.assert_close(value, independent, rtol=0, atol=0)
                for left, right in zip(
                    (*previous.pi, *previous.vf), (*states.pi, *states.vf), strict=True
                ):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                action = raw.cpu().numpy()[0]
                bounded = np.clip(action, -1.0, 1.0)
                clipped.append((action != bounded).tolist())
                observation, reward, terminated, truncated, _ = env.step(bounded)
                values.append(float(value.item()))
                rewards.append(float(np.float32(reward)))
                states = following
                if terminated or truncated:
                    break
            if not (terminated or truncated):
                raise ValueError("fixed complete training window required")
            tail = (
                0.0
                if terminated
                else float(
                    model.policy.predict_values(
                        model.policy.obs_to_tensor(observation)[0],
                        states.vf,
                        torch.tensor([False], dtype=torch.float32),
                    ).item()
                )
            )
    finally:
        env.close()
    result = targets(rewards, values, spec["gamma"], spec["gae_lambda"], tail, bool(terminated))
    return dict(
        symbol=episode["symbol"],
        seed=seed,
        steps=len(values),
        terminated=bool(terminated),
        truncated=bool(truncated),
        first_session=episode["bars"][0]["session"],
        final_session=episode["bars"][len(values)]["session"],
        tail_value=tail,
        rewards=rewards,
        values=values,
        clipping_fraction=np.mean(clipped, axis=0).tolist(),
        targets={k: v.tolist() for k, v in result.items()},
    )


def run(output):
    spec = protocol()
    root = Path.cwd()
    started = time.monotonic()
    deadline = started + spec["maximum_seconds"]
    state_path = root / spec["source_state"]
    state = json.loads(state_path.read_text())
    if state["status"] != "complete" or not state["cleanup_verified"]:
        raise ValueError("owned-cleaned source required")
    item = state["completed"][spec["source_task"]]
    run_path = next((Path(item["root"]) / "runs").iterdir())
    metadata_path = run_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if checksum(metadata_path) != spec["metadata_sha256"]:
        raise ValueError("pinned legacy source metadata required")
    data_path = root / "data/datasets" / spec["dataset"]
    data = validate_payload(read_dataset(data_path), "legacy")
    panel = windows(data, spec)
    protected = {
        str(metadata_path): checksum(metadata_path),
        str(data_path / "dataset.json"): checksum(data_path / "dataset.json"),
    }
    for entry in spec["weights"].values():
        path = run_path / entry["file"]
        if checksum(path) != entry["sha256"]:
            raise ValueError("pinned saved source weights required")
        protected[str(path)] = checksum(path)
    ledger = root / ".runtime/usage-ledger.json"
    protected[str(ledger)] = checksum(ledger)
    output.mkdir(parents=True, exist_ok=False)
    journal = output / "progress.json"
    progress = dict(status="running", completed=[], protocol_sha256=PROTOCOL_SHA256)

    def save():
        temporary = journal.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2))
        temporary.replace(journal)

    save()
    cells = {}
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            for phase, entry in spec["weights"].items():
                model = CLASSES["recurrent_ppo"].load(run_path / entry["file"], device="cpu")
                model.policy.set_training_mode(False)
                record = model_record(model)
                validate_record(
                    record,
                    "legacy",
                    selected_steps=metadata["selected_step"] if phase == "selected" else None,
                )
                if not same_restored(record, metadata["vix_" + phase]):
                    raise ValueError("actual learned model/Adam source mismatch")
                normalization = canonical(asdict(normalizer_from(data["normalizer"])))
                records = []
                with (output / (phase + "-rows.jsonl")).open("x") as stream:
                    for i, symbol in enumerate(spec["panel"]):
                        row = collect(
                            model,
                            panel[symbol],
                            data,
                            spec,
                            spec["sampling_seed_base"] + i,
                            deadline,
                        )
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        records.append(row)
                        progress["completed"].append(phase + ":" + symbol)
                        save()
                if not same_restored(model_record(model), record) or normalization != canonical(
                    asdict(normalizer_from(data["normalizer"]))
                ):
                    raise ValueError("audit changed model/Adam or normalization")
                groups = {}
                for group in ["all", "stock", "etf"]:
                    rows = [
                        r
                        for r in records
                        if group == "all"
                        or (r["symbol"] in {"SPY", "QQQ", "IWM", "DIA"}) == (group == "etf")
                    ]
                    values = np.concatenate(
                        [r["values"][: spec["primary_decisions"]] for r in rows]
                    )
                    metrics = {}
                    for name in ["observed_zero_tail", "bootstrapped_return", "lambda_return"]:
                        target = np.concatenate(
                            [r["targets"][name][: spec["primary_decisions"]] for r in rows]
                        )
                        metrics[name] = error_metrics(values, target)
                    difference = np.concatenate(
                        [
                            np.array(
                                r["targets"]["bootstrapped_return"][: spec["primary_decisions"]]
                            )
                            - np.array(
                                r["targets"]["observed_zero_tail"][: spec["primary_decisions"]]
                            )
                            for r in rows
                        ]
                    )
                    groups[group] = dict(
                        episodes=len(rows),
                        metrics=metrics,
                        maximum_abs_bootstrap_effect=float(np.max(np.abs(difference))),
                        mean_clipping_fraction=np.mean(
                            [r["clipping_fraction"] for r in rows], axis=0
                        ).tolist(),
                    )
                cells[phase] = dict(
                    groups=groups,
                    model_record=record,
                    rows_sha256=checksum(output / (phase + "-rows.jsonl")),
                    episodes=len(records),
                    steps=sum(r["steps"] for r in records),
                    actual_saved_model_unchanged=True,
                )
        if digest(data) != spec["dataset"] or any(checksum(p) != h for p, h in protected.items()):
            raise ValueError("source data/files changed")
        value = dict(
            protocol_sha256=PROTOCOL_SHA256,
            cells=cells,
            limitations=spec["limitations"],
            source_sha256=protected,
            source_files_unchanged=True,
            normalization_unchanged=True,
            training_performed=False,
            optimizer_steps=0,
            test_evaluated=False,
            development_replayed=False,
            colab_allocated=False,
            objective_achieved=False,
            duration_seconds=time.monotonic() - started,
        )
        write_once(
            output / "summary.json", (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
        progress.update(status="complete", duration_seconds=value["duration_seconds"])
        save()
        print(
            "Completed52 fixed training-window stochastic replays; no optimizer steps or Colab units."
        )
        return value
    except BaseException as exc:
        progress.update(status="failed", error_type=type(exc).__name__)
        save()
        raise
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.set_num_threads(threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
