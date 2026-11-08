"""Exact full-path acceptance after owned remote session termination."""

import json
from pathlib import Path

from rl_stocks.core.contracts import digest
from rl_stocks.evaluation.tanh_path_acceptance import compare
from rl_stocks.evaluation.tanh_paths import PROTOCOL_SHA256
from rl_stocks.inference.bundle import checksum
from rl_stocks.runtime.colab.bounded_archive import extract_artifacts


def accept_remote(archive, destination, *, source, source_hash, dataset,
                  local, admission, session_record, limits):
    if (session_record["status"] != "stopped" or session_record["accelerator"] != "CPU"
            or session_record["protocol"]["training"] is not False
            or session_record["protocol"]["source"] != source
            or session_record["protocol"]["source_hash"] != source_hash
            or session_record["protocol"]["admission"] != admission):
        raise ValueError("owned stopped inference session and exact admission required")
    local = Path(local)
    if (checksum(local / "result.json") != admission["local_result_sha256"]
            or checksum(local / "provenance.json") != admission["local_provenance_sha256"]):
        raise ValueError("complete admitted local result changed")
    artifacts = extract_artifacts(archive, destination,
                                 maximum_bytes=limits["maximum_archive_bytes"],
                                 maximum_members=limits["maximum_archive_members"])
    complete = json.loads((artifacts / "completion.json").read_text())
    if (complete["schema"] != "tanh-path-inference-completion-v1"
            or complete["protocol_sha256"] != PROTOCOL_SHA256
            or complete["dataset"] != digest(dataset)
            or complete["source"] != source or complete["source_hash"] != source_hash
            or complete["python"][:2] != [3, 13] or complete["execution"] != "Colab CPU"
            or complete["numerical_sources_sha256"] != digest(admission["numerical_sources"])
            or complete["durable_checkpoint_interval_seconds"] != 300
            or any(complete[k] is not False for k in ("training_performed", "development_evaluated",
                                                       "retired_test_access", "performance_promotion"))):
        raise ValueError("remote inference provenance/source/experiment mismatch")
    files = {str(p.relative_to(artifacts)): checksum(p)
             for p in sorted((artifacts / "tanh-paths").rglob("*")) if p.is_file()}
    if files != complete["files"]:
        raise ValueError("complete remote path file manifest mismatch")
    proof = compare(local, artifacts / "tanh-paths", dataset)
    if (checksum(local / "result.json") != admission["local_result_sha256"]
            or checksum(local / "provenance.json") != admission["local_provenance_sha256"]):
        raise ValueError("local result mutated during remote acceptance")
    return {**proof, "runtime_locations_verified": True,
            "session": session_record["session"], "cleanup_verified": True,
            "remote_completion": complete, "archive_sha256": checksum(archive),
            "local_result_sha256": admission["local_result_sha256"]}
