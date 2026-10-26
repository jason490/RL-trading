"""Build/reload fixed readiness-matched Treasury datasets; no policy learning."""

import gc
import json
import math
from pathlib import Path
import signal
import time

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset, normalizer_from
from rl_stocks.data.generation.treasury_payload import build_payload, validate_payload_pair
from rl_stocks.environment.sampling.windows import StockEtfCycles
from rl_stocks.inference.bundle import checksum

ROOT = Path(__file__).resolve().parents[1]
RULES_SHA256 = "201005f50e266f8dbfb0a992dbd38a2839b85b04fa00947457d17f22fd0e9d12"


def save(value):
    path = ROOT / ".runtime/treasury-payload-progress.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare():
    started = time.monotonic()
    rules_path = ROOT / "configs/experiments/treasury_payload_rules.json"
    rules = json.loads(rules_path.read_text())
    if digest(rules) != RULES_SHA256:
        raise ValueError("fixed Treasury readiness/preprocessing rules required")
    audit_path = ROOT / rules["source_audit"]
    verification_path = audit_path.with_name("verification.json")
    if (
        checksum(audit_path) != rules["source_audit_sha256"]
        or checksum(verification_path) != rules["source_verification_sha256"]
    ):
        raise ValueError("accepted Treasury source/verification hashes required")
    audit = json.loads(audit_path.read_text())
    if audit["coverage_passed"] is not True or audit["model_inputs_enabled"] is not False:
        raise ValueError("source acceptance required; no implicit training")
    source_path = ROOT / "data/datasets" / rules["source_dataset"]
    source = read_dataset(source_path)
    if digest(source["normalizer"]) != rules["original_normalizer_sha256"]:
        raise ValueError("original frozen seven-field preprocessing required")
    context_path = Path(audit["source_path"])
    alignment_path = Path(audit["alignment_path"])
    if (
        checksum(context_path) != rules["source_context"]
        or checksum(alignment_path) != rules["alignment_sha256"]
    ):
        raise ValueError("immutable accepted Treasury source/alignment required")
    treasury = json.loads(context_path.read_text())
    files = [
        rules_path,
        audit_path,
        verification_path,
        source_path / "dataset.json",
        context_path,
        alignment_path,
        ROOT / ".runtime/usage-ledger.json",
        Path(__file__),
        ROOT / "src/rl_stocks/data/generation/treasury_payload.py",
    ]
    protected = {str(p): checksum(p) for p in files}
    records = {}
    eligibility = None
    bounds = None
    original_validation = {
        k: e for k, e in source["episodes"].items() if e["split"] == "validation"
    }
    for mode in rules["arms"]:
        save(dict(status="building", mode=mode, completed=records))
        payload = build_payload(source, treasury, mode)
        identity = digest(payload)
        target = ROOT / "data/datasets" / identity
        write_once(target / "dataset.json", canonical(payload))
        del payload
        gc.collect()
        restored = read_dataset(target)
        validate_payload_pair(source, restored, treasury, mode)
        lineage = restored["manifest"]["treasury_context"]
        if lineage["source_alignment_sha256"] != rules["alignment_sha256"]:
            raise ValueError("recomputed Treasury alignment differs from source acceptance")
        if eligibility is not None and eligibility != lineage["eligibility"]:
            raise ValueError("matched full/masked readiness and eligibility required")
        eligibility = lineage["eligibility"]
        training = [e for e in restored["episodes"].values() if e["split"] == "train"]
        if len(training) != 111 or len({e["symbol"] for e in training}) != 111:
            raise ValueError("all111 unique training symbols must remain")
        sampler = StockEtfCycles(training, 512)
        if bounds is not None and bounds != sampler.bounds:
            raise ValueError("masked/full512-step sampler bounds must agree")
        bounds = sampler.bounds
        validation = {k: e for k, e in restored["episodes"].items() if e["split"] == "validation"}
        if set(validation) != set(original_validation) or len(validation) != 26:
            raise ValueError("original26 validation keys required")
        for key, e in validation.items():
            old = original_validation[key]
            if (
                e["bars"] != old["bars"]
                or e["atr"] != old["atr"]
                or [r[:7] for r in e["features"]] != old["features"]
            ):
                raise ValueError("validation prices/ATR/dates/original fields changed")
        normalizer = normalizer_from(restored["normalizer"])
        maximum = 0.0
        for e in restored["episodes"].values():
            for row in e["features"]:
                values = normalizer.transform(row, restored["names"])
                if any(not math.isfinite(v) or abs(v) > 3.4028234663852886e38 for v in values):
                    raise ValueError("finite float32 Treasury input required")
                if mode == "masked" and values[7:] != [0.0, 0.0]:
                    raise ValueError("exact raw/normalized mask required")
                maximum = max(maximum, max(abs(v) for v in values))
        train_lineage = [
            eligibility[k] for k, e in restored["episodes"].items() if e["split"] == "train"
        ]
        records[mode] = dict(
            dataset=identity,
            schema=restored["schema"],
            names=restored["names"],
            normalizer=restored["normalizer"],
            normalizer_sha256=digest(restored["normalizer"]),
            eligibility_sha256=lineage["eligibility_sha256"],
            extra_normalizer_fit_rows=lineage["extra_normalizer_fit_rows"],
            training_rows=sum(len(e["bars"]) for e in training),
            development_rows=sum(len(e["bars"]) for e in validation.values()),
            exclusions={
                name: sum(row[name] for row in train_lineage)
                for name in ["unavailable", "purged", "short_segment_rows"]
            },
            eligible_starts=sum(sampler.bounds),
            sampler_bounds_sha256=digest(sampler.bounds),
            training_symbols_sha256=digest([e["symbol"] for e in training]),
            finite_float32_inputs=True,
            maximum_abs_normalized_value=maximum,
            validation_rows_preserved=True,
        )
        del restored, training, validation, sampler
        gc.collect()
    if any(checksum(Path(p)) != sha for p, sha in protected.items()):
        raise ValueError("source/rules/data/normalizer/ledger changed")
    result = dict(
        payload_rules_sha256=RULES_SHA256,
        source_dataset=rules["source_dataset"],
        source_context=rules["source_context"],
        passed=True,
        records=records,
        eligibility=eligibility,
        protected_sha256=protected,
        source_files_unchanged=True,
        normalization_training_only=True,
        validation_used_for_fit=False,
        model_initialized=False,
        training_performed=False,
        training_enabled=False,
        provider_requested=False,
        colab_allocated=False,
        test_evaluated=False,
        historical_publication_vintages_verified=False,
        duration_seconds=time.monotonic() - started,
    )
    write_once(
        ROOT / "results/treasury-context-2026-10-26/payloads.json",
        (json.dumps(result, indent=2, allow_nan=False) + "\n").encode(),
    )
    return result


def main():
    path = ROOT / ".runtime/treasury-payload-progress.json"
    if path.exists():
        raise ValueError("existing payload journal requires review; no automatic restart")
    write_once(path, b'{"status":"starting"}\n')

    def expire(*_):
        raise TimeoutError("fixed600-second Treasury preprocessing deadline")

    old = signal.signal(signal.SIGALRM, expire)
    signal.alarm(600)
    try:
        result = prepare()
        save(
            dict(
                status="complete", datasets={m: r["dataset"] for m, r in result["records"].items()}
            )
        )
    except BaseException as error:
        save(dict(status="failed", error_type=type(error).__name__))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    print(json.dumps(dict(passed=result["passed"], records=result["records"]), indent=2))


if __name__ == "__main__":
    main()
