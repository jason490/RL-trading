"""Decompose saved training reward variation with one fixed holding subtraction."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from rl_stocks.core.contracts import digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.benchmark_variance import (
    protocol,
    PROTOCOL_SHA256,
    benchmark_rewards,
    discounted,
    decomposition,
    aggregate,
    feasibility,
)
from rl_stocks.reporting.recurrent_values import protocol as value_protocol, windows


def run(output):
    root = Path.cwd()
    spec = protocol()
    started = time.monotonic()
    deadline = started + spec["maximum_seconds"]
    protected = {}
    for name, expected in [
        (spec["source_summary"], spec["source_summary_sha256"]),
        (spec["source_protocol"], spec["source_protocol_sha256"]),
    ]:
        path = root / name
        if checksum(path) != expected:
            raise ValueError("fixed prior audit source required")
        protected[str(path)] = expected
    prior = json.loads((root / spec["source_summary"]).read_text())
    traces = {}
    for phase, entry in spec["source_traces"].items():
        path = root / entry["file"]
        if checksum(path) != entry["sha256"]:
            raise ValueError("pinned stored trajectory required")
        traces[phase] = [json.loads(line) for line in path.read_text().splitlines()]
        protected[str(path)] = entry["sha256"]
    data_path = root / "data/datasets" / spec["dataset"]
    data = read_dataset(data_path)
    if digest(data) != spec["dataset"]:
        raise ValueError("fixed immutable source dataset required")
    protected[str(data_path / "dataset.json")] = checksum(data_path / "dataset.json")
    ledger = root / ".runtime/usage-ledger.json"
    protected[str(ledger)] = checksum(ledger)
    parent = value_protocol()
    panel = windows(data, parent)
    for phase, rows in traces.items():
        if [r["symbol"] for r in rows] != parent["panel"] or len(rows) != prior["cells"][phase][
            "episodes"
        ]:
            raise ValueError("complete fixed trace panel/order required")
        for index, row in enumerate(rows):
            episode = panel[row["symbol"]]
            if (
                row["steps"] != 512
                or row["terminated"]
                or not row["truncated"]
                or row["seed"] != parent["sampling_seed_base"] + index
                or row["first_session"] != episode["bars"][0]["session"]
                or row["final_session"] != episode["bars"][-1]["session"]
            ):
                raise ValueError("exact fixed training trace/window alignment required")
    output.mkdir(parents=True, exist_ok=False)
    progress = dict(status="running", completed=[])

    def save():
        p = output / "progress.json"
        temporary = p.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2))
        temporary.replace(p)

    save()
    benchmarks = {}
    cells = {}
    try:
        for symbol, episode in panel.items():
            if time.monotonic() >= deadline:
                raise TimeoutError("benchmark audit deadline")
            rewards, baseline = benchmark_rewards(episode, SimulationConfig())
            benchmarks[symbol] = dict(
                rewards=rewards.tolist(),
                final_equity=baseline["rows"][-1]["equity"],
                first_execution_session=baseline["rows"][0]["session"],
                final_execution_session=baseline["rows"][-1]["session"],
            )
            progress["completed"].append("benchmark:" + symbol)
            save()
        for phase, rows in traces.items():
            per_symbol = {}
            series = {}
            for row in rows:
                if time.monotonic() >= deadline:
                    raise TimeoutError("benchmark audit deadline")
                symbol = row["symbol"]
                policy = np.array(row["rewards"])
                benchmark = np.array(benchmarks[symbol]["rewards"])
                target = discounted(policy, spec["gamma"])
                base_target = discounted(benchmark, spec["gamma"])
                np.testing.assert_allclose(
                    target, row["targets"]["observed_zero_tail"], rtol=0, atol=1e-12
                )
                np.testing.assert_allclose(
                    discounted(policy - benchmark, spec["gamma"]),
                    target - base_target,
                    rtol=0,
                    atol=1e-12,
                )
                n = spec["primary_decisions"]
                series[symbol] = dict(
                    rewards=(policy, benchmark), discounted_targets=(target[:n], base_target[:n])
                )
                per_symbol[symbol] = {
                    level: decomposition(*pair) for level, pair in series[symbol].items()
                }
            groups = {}
            for group in ["all", "stock", "etf"]:
                symbols = [
                    s
                    for s in panel
                    if group == "all" or (s in {"SPY", "QQQ", "IWM", "DIA"}) == (group == "etf")
                ]
                groups[group] = {
                    level: dict(
                        aggregate([per_symbol[s][level] for s in symbols]),
                        pooled_descriptive=decomposition(
                            np.concatenate([series[s][level][0] for s in symbols]),
                            np.concatenate([series[s][level][1] for s in symbols]),
                        ),
                    )
                    for level in ["rewards", "discounted_targets"]
                }
            cells[phase] = dict(groups=groups, per_symbol=per_symbol)
            progress["completed"].append("phase:" + phase)
            save()
        if any(checksum(p) != h for p, h in protected.items()):
            raise ValueError("source files/ledger changed")
        write_once(
            output / "benchmark-rows.json", (json.dumps(benchmarks, indent=2) + "\n").encode()
        )
        value = dict(
            protocol_sha256=PROTOCOL_SHA256,
            cells=cells,
            feasibility=feasibility(cells, spec),
            coefficient=1.0,
            source_sha256=protected,
            benchmark_rows_sha256=checksum(output / "benchmark-rows.json"),
            source_files_unchanged=True,
            limitations=spec["limitations"],
            duration_seconds=time.monotonic() - started,
            training_performed=False,
            model_replayed=False,
            optimizer_steps=0,
            colab_allocated=False,
            test_evaluated=False,
            objective_achieved=False,
        )
        write_once(
            output / "summary.json", (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
        progress.update(status="complete", duration_seconds=value["duration_seconds"])
        save()
        print(json.dumps(value["feasibility"], indent=2))
        return value
    except BaseException as exc:
        progress.update(status="failed", error_type=type(exc).__name__)
        save()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)


if __name__ == "__main__":
    main()
