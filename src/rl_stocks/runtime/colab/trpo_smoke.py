"""Single-claim CPU engineering run with original ledger and mandatory cleanup."""

import argparse
import fcntl
import json
from pathlib import Path
import signal
import tarfile
import time

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.provenance import repository_state, source_identity
from rl_stocks.inference.bundle import checksum
from rl_stocks.runtime.colab.budget import BudgetStop, execution_window
from rl_stocks.runtime.colab.runner import ColabJob
from rl_stocks.runtime.colab.transport import TransportLost, export_ready
from rl_stocks.runtime.colab.trpo_smoke_artifacts import accept
from rl_stocks.training.trpo_smoke import PROTOCOL_SHA256, validate_dataset

RUNTIME_SHA256 = "f798e3cdc0e37ac0e923519312ff2257bbb1d0394bf4909fdea565d0bb43f372"


def runtime_spec(root):
    spec = json.loads((Path(root) / "configs/experiments/trpo_smoke_runtime.json").read_text())
    if digest(spec) != RUNTIME_SHA256 or spec["smoke_protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("fixed engineering allocation contract required")
    receipt = Path(root) / "results/trpo-feasibility-2026-11-04/cross-machine-acceptance.json"
    if checksum(receipt) != spec["reference_acceptance_sha256"]:
        raise ValueError("prior exact Colab reference acceptance required")
    return spec


def prepare_upload(root, dataset, output, source):
    """Allowlist research files; never include secrets, artifacts or Python caches."""
    root, output = Path(root), Path(output)
    captured = output.parent / "source-git.json"
    captured.write_bytes(canonical(source))
    with tarfile.open(output, "w:gz", compresslevel=1) as archive:
        for name in ("src", "scripts", "configs", "pyproject.toml", "requirements.txt", "requirements.lock"):
            path = root / name
            paths = sorted(path.rglob("*")) if path.is_dir() else [path]
            for item in paths:
                if item.is_file() and not item.is_symlink() and "__pycache__" not in item.parts:
                    archive.add(item, arcname=str(item.relative_to(root)), recursive=False)
        archive.add(dataset, arcname="dataset.json")
        archive.add(captured, arcname="source-git.json")
    return output


class TrpoSmokeJob(ColabJob):
    download_timeout = 120

    def guard_allowance(self, seconds):
        active = self.ledger.active
        projected = self.ledger.value["consumed"] - active["base_consumed"] + (
            self.ledger.clock() - active["last_time"] + seconds) * (active["rate"] + .01) / 3600
        if projected >= self.spec["maximum_units"]:
            self.ledger.value["budget_stopped"] = True
            self.ledger.save()
            raise BudgetStop("TRPO smoke allowance reached; no automatic retry")
        self.ledger.guard(seconds)

    def tick(self):
        super().tick()
        self.guard_allowance(self.spec["reserve_seconds"])
        self.last_snapshot_meter = self.last_meter  # Replace generic overwrite snapshots.
        if self.allocated and time.monotonic() - self.last_snapshot >= self.spec["snapshot_seconds"]:
            self.last_snapshot = time.monotonic()
            self.guard_allowance(30 + self.spec["reserve_seconds"])
            target = self.output / f"checkpoint-{len(self.snapshots):03d}.tar.gz"
            try:
                self.short(["download", "-s", self.session,
                            "/content/research-checkpoint.tar.gz", str(target)], 30)
            except Exception:
                return  # The first durable checkpoint may not yet exist.
            self.snapshots.append(str(target))
            self.ledger.active["retrieved"].append(str(target))
            self.ledger.save()

    def operation(self, args, timeout):
        self.guard_allowance(timeout + self.spec["reserve_seconds"])
        return super().operation(args, timeout)

    def run_smoke(self, dataset):
        self.spec = runtime_spec(self.root)
        if self.accelerator != "CPU":
            raise ValueError("fixed CPU smoke allocation required")
        dataset = Path(dataset)
        payload = validate_dataset(json.loads(dataset.read_text()))
        source = repository_state(self.root)
        if not source["revision"] or source["dirty"]:
            raise ValueError("clean committed smoke source required")
        source_hash = source_identity()
        if self.ledger.value.get("minimum_account_balance", 0) < 150:
            raise BudgetStop("persisted account floor150 required")
        claim = self.root / self.spec["claim"]
        if claim.exists():
            raise ValueError("smoke allocation already claimed; no automatic retry")
        for command in ("usage", "new", "exec", "upload", "download", "stop", "sessions", "status"):
            self.short([command, "--help"])
        version, usage = self.short(["version"]), self.meter()
        self.output = self.root / "artifacts/colab" / self.session
        self.output.mkdir(parents=True, exist_ok=False)
        upload = prepare_upload(self.root, dataset, self.output / "upload.tar.gz", source)
        binding = {"session": self.session, "source": source, "source_hash": source_hash,
                   "dataset": digest(payload), "upload_sha256": checksum(upload),
                   "runtime_sha256": RUNTIME_SHA256, "training": True,
                   "automatic_retry": False, "performance_allocation": False}
        with claim.open("x") as stream:
            json.dump(binding, stream, sort_keys=True)
        active = self.ledger.begin(self.session, usage)
        active.update(accelerator="CPU", kind="trpo-native-engineering-smoke", protocol=binding,
                      cli_version=version.strip())
        self.ledger.save()
        self.last_meter = time.monotonic()
        self.last_snapshot = time.monotonic()
        self.snapshots = []
        retrieved = False
        clean = False
        archive = self.output / "artifacts.tar.gz"

        def interrupted(signum, frame):
            raise KeyboardInterrupt()

        previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            self.operation(["new", "-s", self.session], 180)
            self.allocated = True
            active["status"] = "active"
            self.ledger.save()
            try:
                self.ledger.observe(self.meter())
            except Exception as error:
                active["metering_unreliable"] = True
                self.ledger.value["budget_stopped"] = True
                self.ledger.save()
                raise BudgetStop("post-allocation metering failed; teardown required") from error
            self.last_meter = time.monotonic()
            self.tick()
            if "CPU" not in self.short(["status", "-s", self.session]).upper():
                raise RuntimeError("owned CPU allocation not verified")
            self.operation(["exec", "-s", self.session, "--timeout", "60", "-f",
                            str(self.root / "scripts/check_colab_kernel.py")], 60)
            kernel = self.output / "kernel.json"
            if not self.retrieve("/content/task-reference-kernel.json", kernel):
                raise RuntimeError("default kernel inspection retrieval failed")
            if json.loads(kernel.read_text())["python"][:2] != [3, 13]:
                raise RuntimeError("actual Colab Python changed; align project before installation")
            self.operation(["upload", "-s", self.session, str(upload),
                            "/content/research-upload.tar.gz"], 120)
            self.operation(["exec", "-s", self.session, "--timeout", "180", "-f",
                            str(self.root / "scripts/install_colab_bootstrap.py")], 180)
            window = execution_window(active["deadline"] - self.ledger.clock(),
                                      self.spec["preferred_execution_seconds"],
                                      self.spec["reserve_seconds"] + 10)
            deadline = time.monotonic() + window
            try:
                self.operation(["exec", "-s", self.session, "--timeout", str(window),
                                "--env", "RL_ALLOCATION_DEADLINE=" + str(active["deadline"]),
                                "-f", str(self.root / "scripts/remote_trpo_smoke.py")], window)
            except TransportLost:
                # A failed output stream does not authorize another worker or allocation.
                active["awaiting_same_worker_export"] = True
                self.ledger.save()
                while True:
                    self.tick()
                    if time.monotonic() + 20 >= deadline:
                        raise TimeoutError("same worker export deadline reached") from None
                    self.guard_allowance(20 + self.spec["reserve_seconds"])
                    if export_ready(self.root, self.session):
                        break
                    time.sleep(5)
            retrieved = self.retrieve("/content/research-artifacts.tar.gz", archive)
            if not retrieved:
                raise RuntimeError("bounded artifact retrieval exhausted")
        except BudgetStop:
            self.ledger.value["budget_stopped"] = True
            self.ledger.save()
            raise
        finally:
            for s in previous:
                signal.signal(s, signal.SIG_IGN)
            try:
                if self.allocated and not retrieved:
                    try:
                        self.retrieve("/content/research-artifacts.tar.gz",
                                      self.output / "partial-artifacts.tar.gz")
                    except Exception:
                        pass
                clean = self.cleanup()
                summary = {"session": self.session, "cleanup_verified": clean,
                           "training_verified": False, "measured_units": active["measured"],
                           "estimated_units": active["estimated"], "charged_units": active["charged"],
                           "project_units": self.ledger.value["consumed"],
                           "remaining_project_units": 150 - self.ledger.value["consumed"],
                           "retrieved": active["retrieved"], "binding": binding}
                (self.output / "summary.json").write_bytes(canonical(summary))
                print(json.dumps(summary), flush=True)
            finally:
                for s, handler in previous.items():
                    signal.signal(s, handler)
        if not clean:
            raise RuntimeError("owned session cleanup not verified; see persisted recovery ID")
        proof = accept(archive, self.output / "retrieved", source=source,
                       source_hash=source_hash, dataset=payload)
        (self.output / "acceptance.json").write_bytes(canonical(proof))
        active["training_verified"] = True
        self.ledger.save()
        summary["training_verified"] = True
        (self.output / "summary.json").write_bytes(canonical(summary))
        return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--cli-python", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    with (root / ".runtime/allocation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        TrpoSmokeJob(root, args.cli_python, "CPU").run_smoke(args.dataset)


if __name__ == "__main__":
    main()
