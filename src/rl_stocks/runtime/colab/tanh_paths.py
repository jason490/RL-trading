"""Gated one-claim full economic-path inference on a metered Colab CPU runtime."""

import argparse
import fcntl
import json
from pathlib import Path

from rl_stocks.core.contracts import digest
from rl_stocks.evaluation.tanh_paths import PROTOCOL_SHA256
from rl_stocks.runtime.colab.tanh_path_artifacts import accept_remote
from rl_stocks.runtime.colab.tanh_path_inputs import (
    admit_local, prepare_path_upload, reviewed_numerical_sources,
)
from rl_stocks.runtime.colab.trpo_smoke import TrpoSmokeJob

RUNTIME_SHA256 = "7b44a183961fcc4e51a6952b814412766c234c91b549a87bfa769d0335f3ad43"


def runtime_spec(root):
    value = json.loads((Path(root) / "configs/experiments/tanh_paths_runtime.json").read_text())
    if digest(value) != RUNTIME_SHA256 or value["full_path_protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("fixed full-path inference runtime contract required")
    return value


class TanhPathJob(TrpoSmokeJob):
    worker_script = "scripts/remote_tanh_paths.py"
    runtime_sha256 = RUNTIME_SHA256
    kind = "tanh-full-path-inference"
    training_performed = False

    def configuration(self):
        return runtime_spec(self.root)

    def preflight(self, payload):
        self.local_admission = admit_local(self.root, payload)

    def admission_record(self):
        return self.local_admission

    def prepare_input(self, dataset, output, source):
        return prepare_path_upload(self.root, dataset, output, source, self.local_admission)

    def accept_artifacts(self, archive, destination, *, source, source_hash, dataset):
        if reviewed_numerical_sources(self.root) != self.local_admission["numerical_sources"]:
            raise ValueError("admitted calculation sources changed during remote replay")
        return accept_remote(archive, destination, source=source, source_hash=source_hash,
                             dataset=dataset, local=self.root / "artifacts/trpo-full-path-local-v1",
                             admission=self.local_admission, session_record=self.ledger.active,
                             limits=self.spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--cli-python", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    with (root / ".runtime/allocation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        TanhPathJob(root, args.cli_python, "CPU").run_smoke(args.dataset)


if __name__ == "__main__":
    main()
