"""Bounded transport and root identity checks before physical TRPO acceptance."""

import json
from pathlib import Path

from rl_stocks.core.contracts import digest
from rl_stocks.runtime.colab.bounded_archive import extract_artifacts
from rl_stocks.evaluation.trpo_smoke import verify_matrix
from rl_stocks.inference.bundle import checksum
from rl_stocks.training.trpo_smoke import PROTOCOL_SHA256, protocol

MAXIMUM_BYTES = 128 * 1024 * 1024


def extract_owned(path, destination):
    return extract_artifacts(path, destination, maximum_bytes=MAXIMUM_BYTES, maximum_members=100)


def verify_root(artifacts, *, source, source_hash, dataset):
    artifacts = Path(artifacts)
    complete = json.loads((artifacts / "completion.json").read_text())
    result = json.loads((artifacts / "trpo-smoke/result.json").read_text())
    spec = protocol()
    if (
        complete["schema"] != "trpo-smoke-completion-v1"
        or complete["python"][:2] != [3, 13]
        or complete["source"] != source or complete["source_hash"] != source_hash
        or complete["protocol_sha256"] != PROTOCOL_SHA256
        or complete["dataset"] != spec["dataset"] or digest(dataset) != spec["dataset"]
        or complete["training_location"] != "Colab CPU"
        or complete["development_evaluated"] is not False
        or complete["retired_test_access"] is not False
        or result["schema"] != "trpo-smoke-result-v1"
        or result["source"] != source or result["source_hash"] != source_hash
        or result["protocol_sha256"] != PROTOCOL_SHA256
        or result["dataset"] != spec["dataset"]
        or result["normalizer_sha256"] != spec["normalizer_sha256"]
        or digest(result["normalizer"]) != spec["normalizer_sha256"]
        or result["normalizer"] != dataset["normalizer"] or result["names"] != dataset["names"]
        or result["development_evaluated"] is not False
        or result["retired_test_access"] is not False
        or result["performance_promotion"] is not False
        or result["training_location"] != "Colab CPU"
        or [r["case"] for r in result["results"]] != spec["cases"]
    ):
        raise ValueError("root kernel/source/dataset/preprocessing/experiment identity mismatch")
    actual_files = {str(p.relative_to(artifacts)): checksum(p)
                    for p in sorted((artifacts / "trpo-smoke").rglob("*")) if p.is_file()}
    if complete["files"] != actual_files:
        raise ValueError("complete smoke artifact manifest mismatch")
    for case, row in zip(spec["cases"], result["results"], strict=True):
        folder = artifacts / "trpo-smoke" / f'{case["mode"]}-{case["algorithm"]}'
        if json.loads((folder / "result.json").read_text()) != row:
            raise ValueError("root/case smoke receipts disagree")
    return complete


def accept(path, destination, *, source, source_hash, dataset):
    artifacts = extract_owned(path, destination)
    completion = verify_root(artifacts, source=source, source_hash=source_hash, dataset=dataset)
    proof = verify_matrix(artifacts / "trpo-smoke")
    return {"schema": "trpo-smoke-physical-acceptance-v1", "completion": completion,
            "matrix": proof, "archive_sha256": checksum(path),
            "normalization_unchanged": True, "performance_promotion": False}
