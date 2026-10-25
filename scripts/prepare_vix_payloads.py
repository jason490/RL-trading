"""Prepare/revalidate immutable VIX payloads; local preprocessing, no policy fitting."""

import gc
import json
import math
from pathlib import Path
import signal
import time

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset, normalizer_from
from rl_stocks.data.generation.volatility_payload import build_payload, validate_payload_pair
from rl_stocks.inference.bundle import checksum
from rl_stocks.training.vix_spec import preregistration, PREREGISTRATION_SHA256

ROOT = Path(__file__).resolve().parents[1]


def save(value):
    path = ROOT / ".runtime/vix-payload-progress.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare():
    started = time.monotonic()
    spec = preregistration()
    audit_path = ROOT / spec["consumer_audit"]
    verification_path = audit_path.with_name("consumer-verification.json")
    if (
        checksum(audit_path) != spec["consumer_audit_sha256"]
        or checksum(verification_path) != spec["consumer_verification_sha256"]
    ):
        raise ValueError("pinned expanded consumer evidence required")
    audit = json.loads(audit_path.read_text())
    verification = json.loads(verification_path.read_text())
    if audit["coverage_passed"] is not True or verification["all_consumer_rows_ready"] is not True:
        raise ValueError("passing accepted expanded-consumer coverage required")
    data_path = ROOT / "data/datasets" / spec["source_dataset"]
    source = read_dataset(data_path)
    context_path = ROOT / "data/context/vix" / spec["source_context"] / "snapshot.json"
    if checksum(context_path) != spec["source_context"]:
        raise ValueError("fixed immutable VIX source required")
    volatility = json.loads(context_path.read_text())
    if digest(source["normalizer"]) != spec["normalization"]["original_seven_sha256"]:
        raise ValueError("original immutable seven-field preprocessing required")
    files = [
        audit_path,
        verification_path,
        data_path / "dataset.json",
        context_path,
        Path(__file__),
        ROOT / "configs/experiments/vix_context_preregistration.json",
        ROOT / "src/rl_stocks/data/generation/volatility_payload.py",
    ]
    protected = {str(p): checksum(p) for p in files}
    records = {}
    for mode in ("masked", "full"):
        save(dict(status="building", mode=mode, completed=records))
        payload = build_payload(source, volatility, mode)
        identity = digest(payload)
        target = ROOT / "data/datasets" / identity
        write_once(target / "dataset.json", canonical(payload))
        del payload
        gc.collect()
        restored = read_dataset(target)
        validate_payload_pair(source, restored, volatility, mode)
        normalizer = normalizer_from(restored["normalizer"])
        maximum = 0.0
        for episode in restored["episodes"].values():
            for row in episode["features"]:
                values = normalizer.transform(row, restored["names"])
                if any(not math.isfinite(v) or abs(v) > 3.4028234663852886e38 for v in values):
                    raise ValueError("finite float32 VIX inputs required")
                if mode == "masked" and values[7:] != [0.0, 0.0]:
                    raise ValueError("exact raw/normalized VIX mask required")
                maximum = max(maximum, max(abs(v) for v in values))
        records[mode] = dict(
            dataset=identity,
            schema=restored["schema"],
            names=restored["names"],
            normalizer=restored["normalizer"],
            normalizer_sha256=digest(restored["normalizer"]),
            extra_normalizer_fit_rows=restored["manifest"]["vix_context"][
                "extra_normalizer_fit_rows"
            ],
            maximum_abs_normalized_value=maximum,
            finite_float32_inputs=True,
            original_rows_bars_atr_and_seven_features_exact=True,
        )
        del restored
        gc.collect()
    if any(checksum(Path(p)) != sha for p, sha in protected.items()):
        raise ValueError("source/protocol/preprocessing changed")
    result = dict(
        preregistration_sha256=PREREGISTRATION_SHA256,
        source_dataset=spec["source_dataset"],
        source_context=spec["source_context"],
        passed=True,
        records=records,
        protected_sha256=protected,
        source_files_unchanged=True,
        training_rows=audit["row_counts"]["train"],
        development_rows=audit["row_counts"]["validation"],
        extra_preprocessing_fitted_on_training_only=True,
        validation_used_for_fit=False,
        policy_training_performed=False,
        provider_requested=False,
        colab_allocated=False,
        test_evaluated=False,
        historical_publication_vintages_verified=False,
        duration_seconds=time.monotonic() - started,
    )
    output = ROOT / "results/vix-context-2026-10-25/payloads.json"
    write_once(output, (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    return result


def main():
    progress = ROOT / ".runtime/vix-payload-progress.json"
    if progress.exists():
        raise ValueError("existing payload journal requires review; no automatic restart")
    write_once(progress, b'{"status":"starting"}\n')

    def expire(*_):
        raise TimeoutError("fixed600-second local payload preparation deadline")

    old = signal.signal(signal.SIGALRM, expire)
    signal.alarm(600)
    try:
        result = prepare()
        save(
            dict(
                status="complete", datasets={k: r["dataset"] for k, r in result["records"].items()}
            )
        )
    except BaseException as error:
        save(dict(status="failed", error_type=type(error).__name__))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    print(
        "Prepared/reloaded full and masked immutable VIX payloads; original data/preprocessing exact; no policy optimization."
    )


if __name__ == "__main__":
    main()
