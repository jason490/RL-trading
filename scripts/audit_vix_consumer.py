"""Offline readiness of the retained VIX source for the accepted expanded panel."""

import json
from pathlib import Path
import signal
import time

from rl_stocks.core.contracts import digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.data.generation.volatility_context import align_context
from rl_stocks.inference.bundle import checksum
from rl_stocks.training.diversity import validate_payload

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "658d3ee5cfd20b8148c50175c23764a7ba470c294c3f9c65caa31461b8dc1484"


def save(value):
    path = ROOT / ".runtime/vix-consumer-progress.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def audit():
    started = time.monotonic()
    protocol_path = ROOT / "configs/experiments/vix_consumer_audit.json"
    spec = json.loads(protocol_path.read_text())
    if digest(spec) != PROTOCOL_SHA256:
        raise ValueError("fixed consumer coverage protocol required")
    audit_path = ROOT / spec["source_audit"]
    if checksum(audit_path) != spec["source_audit_sha256"]:
        raise ValueError("fixed retained source-audit evidence required")
    accepted = json.loads(audit_path.read_text())
    source_path = Path(accepted["source_snapshot"])
    if (
        accepted["coverage_passed"] is not True
        or accepted["source_id"] != spec["source_id"]
        or checksum(source_path) != spec["source_id"]
    ):
        raise ValueError("verified retained source snapshot required")
    source = json.loads(source_path.read_text())
    dataset_path = ROOT / "data/datasets" / spec["dataset"]
    data = read_dataset(dataset_path)
    validate_payload(data, "expanded")
    if (
        digest(data) != spec["dataset"]
        or digest(data["normalizer"]) != spec["original_normalizer_sha256"]
    ):
        raise ValueError("fixed expanded consumer and original preprocessing required")
    files = [
        protocol_path,
        audit_path,
        source_path,
        dataset_path / "dataset.json",
        Path(__file__),
        ROOT / "src/rl_stocks/data/generation/volatility_context.py",
    ]
    protected = {str(p): checksum(p) for p in files}
    joined, counts, symbols = align_context(data, source["rows"])
    if (
        len(symbols["train"]) != spec["train_symbols"]
        or len(symbols["validation"]) != spec["development_symbols"]
    ):
        raise ValueError("exact accepted111train/26development panel required")
    save(dict(status="aligned", counts=counts))

    def coverage(days):
        ready = sum(joined[d]["ready"] for d in days)
        return dict(
            decisions=len(days),
            ready=ready,
            fraction=ready / len(days),
            missing=[d for d in days if not joined[d]["ready"]],
        )

    days = sorted(joined)
    overall = coverage(days)
    eras = {
        name: coverage([d for d in days if first <= d <= last])
        for name, (first, last) in spec["eras"].items()
    }
    years = {str(y): coverage([d for d in days if d.startswith(str(y))]) for y in range(2010, 2024)}
    weighted = {
        split: sum(
            joined[b["session"]]["ready"]
            for e in data["episodes"].values()
            if e["split"] == split
            for b in e["bars"]
        )
        for split in counts
    }
    gate = spec["coverage_gate"]
    passed = (
        overall["fraction"] >= gate["minimum_overall_fraction"]
        and all(v["fraction"] >= gate["minimum_era_fraction"] for v in eras.values())
        and all(v["fraction"] >= gate["minimum_year_fraction"] for v in years.values())
    )
    if any(checksum(Path(p)) != sha for p, sha in protected.items()):
        raise ValueError("retained source/consumer changed")
    result = dict(
        protocol_sha256=PROTOCOL_SHA256,
        source_id=spec["source_id"],
        dataset=spec["dataset"],
        normalizer_sha256=digest(data["normalizer"]),
        train_symbols=len(symbols["train"]),
        development_symbols=len(symbols["validation"]),
        row_counts=counts,
        ready_rows=weighted,
        overall=overall,
        eras=eras,
        years=years,
        coverage_passed=passed,
        source_files_sha256=protected,
        source_files_unchanged=True,
        provider_requested=False,
        training_performed=False,
        normalization_fitted=False,
        returns_evaluated=False,
        test_evaluated=False,
        colab_allocated=False,
        model_inputs_enabled=False,
        historical_publication_vintages_verified=False,
        duration_seconds=time.monotonic() - started,
        interpretation=spec["interpretation"],
    )
    output = ROOT / "results/vix-source-2026-10-25/consumer.json"
    write_once(output, (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    return result


def main():
    progress = ROOT / ".runtime/vix-consumer-progress.json"
    if progress.exists():
        raise ValueError("existing consumer journal requires review; no automatic restart")
    write_once(progress, b'{"status":"starting"}\n')

    def expired(*_):
        raise TimeoutError("fixed180-second consumer-audit deadline")

    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(180)
    try:
        result = audit()
        save(dict(status="complete", coverage_passed=result["coverage_passed"]))
    except BaseException as error:
        save(dict(status="failed", error_type=type(error).__name__))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    print(
        "Offline111-symbol consumer coverage_passed="
        + str(result["coverage_passed"])
        + "; no requests, fitting or training."
    )


if __name__ == "__main__":
    main()
