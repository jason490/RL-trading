"""Controlled Colab full-path inference replay; no learning or checkpoint selection."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback


def replay(root):
    sys.path.insert(0, str(root / "src"))
    from rl_stocks.training.recurrent_sac_learner import require_colab_cpu
    require_colab_cpu()  # Before reading datasets/bundles or constructing accounts.
    from rl_stocks.training.dependencies import verify_environment
    verify_environment()
    from rl_stocks.core.contracts import canonical
    from rl_stocks.core.provenance import repository_state, source_identity
    from rl_stocks.inference.bundle import checksum
    from rl_stocks.training.checkpoint_export import export_artifacts
    from rl_stocks.evaluation import tanh_paths

    original = tanh_paths.paired_episode
    last_checkpoint = time.monotonic()

    def observed(*args, **kwargs):
        nonlocal last_checkpoint
        result = original(*args, **kwargs)
        if time.monotonic() - last_checkpoint >= 300:
            export_artifacts(root / "artifacts")
            last_checkpoint = time.monotonic()
        return result

    tanh_paths.paired_episode = observed
    try:
        result = tanh_paths.run(json.loads((root / "dataset.json").read_text()),
                                root / "tanh-bundles", root / "artifacts/tanh-paths")
    finally:
        tanh_paths.paired_episode = original
        export_artifacts(root / "artifacts")
    files = {str(p.relative_to(root / "artifacts")): checksum(p)
             for p in sorted((root / "artifacts/tanh-paths").rglob("*")) if p.is_file()}
    completion = {"schema": "tanh-path-inference-completion-v1",
                  "protocol_sha256": result["protocol_sha256"], "dataset": result["dataset"],
                  "source": repository_state(root), "source_hash": source_identity(),
                  "python": list(sys.version_info[:3]), "files": files,
                  "training_performed": False, "development_evaluated": False,
                  "retired_test_access": False, "performance_promotion": False,
                  "execution": "Colab CPU", "durable_checkpoint_interval_seconds": 300}
    (root / "artifacts/completion.json").write_bytes(canonical(completion))
    export_artifacts(root / "artifacts")


def main(*, worker=False):
    if sys.version_info[:2] != (3, 13) or not os.environ.get("COLAB_RELEASE_TAG"):
        raise RuntimeError("inspect and match default Colab Python before inference setup")
    root = Path("/content/rl-stocks")
    if worker:
        if Path.cwd() != root:
            raise RuntimeError("owned Colab working directory required")
        replay(root)
        return
    deadline = float(os.environ["RL_ALLOCATION_DEADLINE"])
    root.mkdir(exist_ok=False)
    (root / "artifacts").mkdir()
    try:
        with tarfile.open("/content/research-upload.tar.gz") as archive:
            archive.extractall(root, filter="data")
        sys.path.insert(0, str(root / "src"))
        sys.path.insert(0, str(root / "scripts"))
        from remote_trpo_smoke import remaining
        from rl_stocks.runtime.colab.jobs import bootstrap_commands, bootstrap_environment
        environment = bootstrap_environment(os.environ, root)
        environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("existing bounded uv bootstrap required")
        commands = bootstrap_commands(uv, root, sys.executable)
        commands[-1] = [str(root / ".venv/bin/python"),
                        str(root / "scripts/remote_tanh_paths.py"), "--worker"]
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
