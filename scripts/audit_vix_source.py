"""One bounded source request and outcome-free lagged-context coverage audit."""

from datetime import date
import json
from pathlib import Path
import signal
import time

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.data.providers.yahoo.volatility import fetch_closes
from rl_stocks.features.volatility_context import NAMES, context
from rl_stocks.inference.bundle import checksum

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "a5fc356f4d16f4567e2fdb06c72ddab831533eecc1569ea058d5425d66b3c389"


def save(value):
    path = ROOT / ".runtime/vix-source-progress.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run():
    started = time.monotonic()
    spec_path = ROOT / "configs/experiments/vix_source_audit.json"
    spec = json.loads(spec_path.read_text())
    if digest(spec) != PROTOCOL_SHA256 or list(NAMES) != spec["features"]:
        raise ValueError("fixed outcome-free source protocol required")
    dataset_path = ROOT / "data/datasets" / spec["dataset"]
    data = read_dataset(dataset_path)
    if (
        digest(data) != spec["dataset"]
        or data["manifest"]["included_splits"] != ["train", "validation"]
        or any(
            e["split"] not in ("train", "validation")
            or any(b["session"] >= "2024-01-01" for b in e["bars"])
            for e in data["episodes"].values()
        )
    ):
        raise ValueError("fixed retired-free original dataset required")
    dataset_sha = checksum(dataset_path / "dataset.json")
    normalizer_sha = digest(data["normalizer"])
    import yfinance as yf

    yf.set_tz_cache_location(str(ROOT / ".runtime/yahoo-cache"))
    save(dict(status="source_request", protocol_sha256=PROTOCOL_SHA256))
    source = fetch_closes(
        date.fromisoformat(spec["start"]), date.fromisoformat(spec["end_exclusive"])
    )
    source_id = digest(source)
    source_path = ROOT / "data/context/vix" / source_id / "snapshot.json"
    write_once(source_path, canonical(source))
    save(dict(status="source_retained", source_id=source_id, path=str(source_path)))
    decision_times = {}
    for episode in data["episodes"].values():
        for bar in episode["bars"]:
            day, available = bar["session"], bar["available_time"]
            decision_times[day] = min(decision_times.get(day, available), available)
    sessions = sorted(decision_times)
    joined = context(source["rows"], sessions, decision_times=decision_times)

    def coverage(days):
        ready = sum(joined[d]["ready"] for d in days)
        return dict(
            decisions=len(days),
            ready=ready,
            fraction=ready / len(days),
            missing=[d for d in days if not joined[d]["ready"]],
        )

    overall = coverage(sessions)
    eras = {
        name: coverage([d for d in sessions if first <= d <= last])
        for name, (first, last) in spec["eras"].items()
    }
    years = {
        str(y): coverage([d for d in sessions if d.startswith(str(y))]) for y in range(2010, 2024)
    }
    gate = spec["coverage_gate"]
    passed = (
        overall["fraction"] >= gate["minimum_overall_fraction"]
        and all(v["fraction"] >= gate["minimum_era_fraction"] for v in eras.values())
        and all(v["fraction"] >= gate["minimum_year_fraction"] for v in years.values())
    )
    if (
        checksum(dataset_path / "dataset.json") != dataset_sha
        or digest(data["normalizer"]) != normalizer_sha
    ):
        raise ValueError("immutable dataset/preprocessing changed")
    result = dict(
        protocol_sha256=PROTOCOL_SHA256,
        source_id=source_id,
        source_snapshot=str(source_path),
        source_sha256=checksum(source_path),
        source_observations=len(source["rows"]),
        first_source_session=source["rows"][0]["session"],
        last_source_session=source["rows"][-1]["session"],
        overall=overall,
        eras=eras,
        years=years,
        coverage_passed=passed,
        availability_assumption=spec["availability"],
        historical_publication_vintages_verified=False,
        context_features=list(NAMES),
        model_inputs_enabled=False,
        training_performed=False,
        normalization_fitted=False,
        normalizer_sha256=normalizer_sha,
        dataset_sha256=dataset_sha,
        dataset_unchanged=True,
        returns_evaluated=False,
        colab_allocated=False,
        test_evaluated=False,
        duration_seconds=time.monotonic() - started,
        interpretation=spec["interpretation"],
    )
    output = ROOT / "results/vix-source-2026-10-25/summary.json"
    write_once(output, (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    return result


def main():
    journal = ROOT / ".runtime/vix-source-progress.json"
    if journal.exists():
        raise ValueError("existing source journal requires review; no automatic restart")
    write_once(journal, b'{"status":"starting"}\n')

    def expired(*_):
        raise TimeoutError("fixed180-second source-audit deadline")

    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(180)
    try:
        result = run()
        save(
            dict(
                status="complete",
                source_id=result["source_id"],
                coverage_passed=result["coverage_passed"],
            )
        )
    except BaseException as error:
        save(dict(status="failed", error_type=type(error).__name__))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    print(
        "Source retained and coverage audited; coverage_passed="
        + str(result["coverage_passed"])
        + "; model inputs remain disabled."
    )


if __name__ == "__main__":
    main()
