"""One bounded Treasury source/causal-coverage audit; no fitting or model replay."""

import fcntl
import json
from pathlib import Path
import signal
import time

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.data.providers.treasury.collection import collect
from rl_stocks.data.validation.treasury_coverage import decisions, report
from rl_stocks.features.treasury_context import NAMES, context
from rl_stocks.inference.bundle import checksum

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "0beb46986f3279047bd47ac3e245d1a9f3b749c72cae72b9ab3c45e840238161"
JOURNAL = ROOT / ".runtime/treasury-source-progress.json"


def save(value):
    previous = json.loads(JOURNAL.read_text()) if JOURNAL.exists() else {}
    path = JOURNAL.with_suffix(".tmp")
    path.write_text(json.dumps(dict(previous, **value), indent=2, allow_nan=False) + "\n")
    path.replace(JOURNAL)


def run(spec, started):
    dataset = ROOT / "data/datasets" / spec["dataset"]
    data = read_dataset(dataset)
    if digest(data) != spec["dataset"] or list(NAMES) != spec["features"]:
        raise ValueError("fixed Treasury consumer/source feature contract required")
    times, groups, splits = decisions(data)
    if splits != dict(train=111, validation=26):
        raise ValueError("fixed original111/26 consumer episodes required")
    protected = {
        str(dataset / "dataset.json"): checksum(dataset / "dataset.json"),
        str(ROOT / ".runtime/usage-ledger.json"): checksum(ROOT / ".runtime/usage-ledger.json"),
    }
    normalizer_sha = digest(data["normalizer"])
    cache = ROOT / "data/context/treasury"
    sources, requests_started = collect(
        cache, spec, started + spec["operation_deadline_seconds"], save
    )
    source = dict(
        schema="treasury-annual-source-v1",
        protocol_sha256=PROTOCOL_SHA256,
        receipts=[s["receipt_id"] for s in sources],
        rows=sorted(
            [row for s in sources for row in s["rows"]], key=lambda r: r["observation_date"]
        ),
        historical_publication_vintages_verified=False,
    )
    source_id = digest(source)
    path = cache / "snapshots" / source_id / "snapshot.json"
    write_once(path, canonical(source))
    save(dict(status="coverage", source_id=source_id, source_path=str(path)))
    joined = context(source["rows"], sorted(times), decision_times=times)
    coverage = report(times, groups, joined, spec)
    alignment = cache / "snapshots" / source_id / "alignment.json"
    write_once(alignment, canonical(joined))
    if (
        any(checksum(Path(p)) != sha for p, sha in protected.items())
        or digest(data["normalizer"]) != normalizer_sha
    ):
        raise ValueError("Treasury audit changed immutable consumers/preprocessing/ledger")
    return dict(
        protocol_sha256=PROTOCOL_SHA256,
        source_id=source_id,
        source_path=str(path),
        source_sha256=checksum(path),
        alignment_path=str(alignment),
        alignment_sha256=checksum(alignment),
        source_observations=len(source["rows"]),
        first_source_date=source["rows"][0]["observation_date"],
        last_source_date=source["rows"][-1]["observation_date"],
        requests_started=requests_started,
        annual_sources=[
            dict(
                receipt_id=s["receipt_id"],
                year=s["receipt"]["year"],
                bytes=s["receipt"]["bytes"],
                raw_sha256=s["receipt"]["raw_sha256"],
            )
            for s in sources
        ],
        consumer_episodes=splits,
        coverage=coverage,
        coverage_passed=coverage["coverage_passed"],
        availability_assumption=spec["availability"],
        historical_publication_vintages_verified=False,
        features=list(NAMES),
        model_inputs_enabled=False,
        normalization_fitted=False,
        normalizer_sha256=normalizer_sha,
        protected_files_sha256=protected,
        protected_files_unchanged=True,
        training_performed=False,
        returns_evaluated=False,
        retired_test_requested=False,
        colab_allocated=False,
        duration_seconds=time.monotonic() - started,
        interpretation="Current-vintage source/coverage feasibility under explicit lag and bounded-age assumptions only; no predictive or exact historical publication claim. Separate matched model preregistration required.",
    )


def main():
    spec = json.loads((ROOT / "configs/experiments/treasury_source_audit.json").read_text())
    if digest(spec) != PROTOCOL_SHA256:
        raise ValueError("fixed Treasury source audit required")
    if JOURNAL.exists():
        raise ValueError("existing Treasury journal requires review; no automatic restart")
    write_once(
        JOURNAL,
        canonical(dict(status="starting", requests_started=0, protocol_sha256=PROTOCOL_SHA256)),
    )
    started = time.monotonic()

    def expired(*_):
        raise TimeoutError("fixed600-second Treasury source deadline")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(spec["operation_deadline_seconds"])
    try:
        with (ROOT / ".runtime/treasury-source.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run(spec, started)
        output = ROOT / "results/treasury-source-2026-10-25/summary.json"
        write_once(output, (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
        save(
            dict(
                status="complete",
                coverage_passed=result["coverage_passed"],
                source_id=result["source_id"],
            )
        )
        print(
            json.dumps(
                dict(
                    status="complete",
                    coverage_passed=result["coverage_passed"],
                    source_observations=result["source_observations"],
                    requests_started=result["requests_started"],
                )
            )
        )
    except BaseException as error:
        save(
            dict(
                status="failed",
                error_type=type(error).__name__,
                duration_seconds=time.monotonic() - started,
            )
        )
        failure = json.loads(JOURNAL.read_text())
        failure["retained_receipts"] = [
            p.stem for p in sorted((ROOT / "data/context/treasury/receipts").glob("*.json"))
        ]
        failure.update(
            model_inputs_enabled=False,
            training_performed=False,
            colab_allocated=False,
            automatic_retry=False,
        )
        write_once(ROOT / "results/treasury-source-2026-10-25/failure.json", canonical(failure))
        print(
            json.dumps(
                dict(
                    status="failed",
                    error_type=type(error).__name__,
                    requests_started=failure["requests_started"],
                    automatic_retry=False,
                )
            )
        )
        raise SystemExit(1) from None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
