"""One metered Colab forecast study, then independent restoration after shutdown."""

import argparse
import json
from pathlib import Path
import signal

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.runtime.colab.fleet import CLI, Fleet
from rl_stocks.runtime.colab.jobs import boosting_plan, reviewed_boosting_plan
from rl_stocks.training.supervised.boosting_artifacts import verify
from rl_stocks.training.supervised.spec import SIGNAL_DATASET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli-python", required=True)
    parser.add_argument("--reviewed-bootstrap", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    state = root / (
        ".runtime/residual-boosting-reviewed-fleet-v1.json"
        if args.reviewed_bootstrap
        else ".runtime/residual-boosting-fleet-v1.json"
    )
    dataset = root / "data/signals" / SIGNAL_DATASET
    tasks = [
        {
            "id": "residual-boosting",
            "dataset": str(dataset),
            "plan": reviewed_boosting_plan() if args.reviewed_bootstrap else boosting_plan(),
        }
    ]
    runner = Fleet(root, CLI(root, args.cli_python), tasks, 1, state)
    runner.run()
    status = json.loads(state.read_text())
    if status["status"] != "complete" or not status.get("cleanup_verified"):
        raise RuntimeError("verified shutdown required before offline restoration")
    completed = status["completed"]["residual-boosting"]
    original = json.loads(
        (root / ".runtime/payloads" / completed["session"] / "source-git.json").read_text()
    )
    if original != completed["completion"]["source_git"]:
        raise ValueError("captured source/completion mismatch")

    def expired(signum, frame):
        raise TimeoutError("post-shutdown restoration deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(1800)
    try:
        receipt = verify(Path(completed["root"]) / "boosting", dataset)
        receipt.update(
            session=completed["session"],
            archive_sha256=completed["archive_sha256"],
            cleanup_verified=True,
        )
        write_once(root / "artifacts/residual-boosting-restoration-v1.json", canonical(receipt))
        if not receipt["gate"]["passed"]:
            write_once(
                root / ".runtime/residual-boosting-family-closure-v1.json", canonical(receipt)
            )
        print(json.dumps(receipt["gate"], indent=2), flush=True)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
