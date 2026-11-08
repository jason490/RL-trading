"""Dedicated default-kernel bootstrap and worker for the fixed TRPO smoke matrix."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback


def remaining(deadline, reserve=900):
    value = int(deadline - time.time() - reserve)
    if value < 60:
        raise TimeoutError("insufficient allocation time after retrieval/cleanup reserve")
    return value


def learn(root):
    sys.path.insert(0, str(root / "src"))
    from rl_stocks.core.contracts import canonical
    from rl_stocks.inference.bundle import checksum
    from rl_stocks.training.checkpoint_export import export_artifacts
    from rl_stocks.training.trpo_smoke import run

    result = run(json.loads((root / "dataset.json").read_text()), root / "artifacts/trpo-smoke")
    files = {str(p.relative_to(root / "artifacts")): checksum(p)
             for p in sorted((root / "artifacts/trpo-smoke").rglob("*")) if p.is_file()}
    completion = {
        "schema": "trpo-smoke-completion-v1", "python": list(sys.version_info[:3]),
        "source": result["source"], "source_hash": result["source_hash"],
        "protocol_sha256": result["protocol_sha256"], "dataset": result["dataset"],
        "files": files, "training_location": "Colab CPU",
        "development_evaluated": False, "retired_test_access": False,
    }
    (root / "artifacts/completion.json").write_bytes(canonical(completion))
    export_artifacts(root / "artifacts")


def main(*, worker=False):
    # Inspect the default kernel BEFORE extraction, installation or fitting.
    if sys.version_info[:2] != (3, 13) or not os.environ.get("COLAB_RELEASE_TAG"):
        raise RuntimeError("inspect and match actual default Colab Python before training")
    root = Path("/content/rl-stocks")
    if worker:
        if Path.cwd() != root:
            raise RuntimeError("owned Colab working directory required")
        learn(root)
        return
    deadline = float(os.environ["RL_ALLOCATION_DEADLINE"])
    root.mkdir(exist_ok=False)
    (root / "artifacts").mkdir()
    try:
        with tarfile.open("/content/research-upload.tar.gz") as archive:
            archive.extractall(root, filter="data")
        sys.path.insert(0, str(root / "src"))
        from rl_stocks.runtime.colab.jobs import bootstrap_commands, bootstrap_environment
        environment = bootstrap_environment(os.environ, root)
        environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("existing bounded remote uv bootstrap required")
        commands = bootstrap_commands(uv, root, sys.executable)
        commands[-1] = [str(root / ".venv/bin/python"),
                        str(root / "scripts/remote_trpo_smoke.py"), "--worker"]
        for command in commands:
            with (root / "artifacts/bootstrap-log.txt").open("a") as log:
                subprocess.run(command, cwd=root, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, check=True,
                               timeout=remaining(deadline))
    except BaseException:
        (root / "artifacts/failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        temporary = Path("/content/research-artifacts.partial.tar.gz")
        with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
            archive.add(root / "artifacts", arcname="artifacts")
        temporary.replace("/content/research-artifacts.tar.gz")
        Path("/content/research-artifacts.ready").touch()


if __name__ == "__main__":
    main(worker=sys.argv[1:] == ["--worker"])
