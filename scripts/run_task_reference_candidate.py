"""One Colab routed100k cell, then offline acceptance after verified shutdown."""

import argparse
from pathlib import Path
import signal

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.runtime.colab.fleet import CLI, Fleet
from rl_stocks.runtime.colab.jobs import task_reference_candidate_plan
from rl_stocks.training.task_reference_candidate_acceptance import accept
from rl_stocks.training.task_reference_runtime import execution_spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli-python", required=True)
    parser.add_argument("--state", default=".runtime/task-reference-candidate-fleet-v1.json")
    parser.add_argument("--output", default="artifacts/task-reference-candidate-acceptance-v1")
    args = parser.parse_args()
    root = Path.cwd()
    spec = execution_spec(root)
    plan = task_reference_candidate_plan()
    tasks = [
        {
            "id": "task-reference-routed-11",
            "dataset": "data/datasets/" + plan["dataset_contract"]["identity"],
            "plan": plan,
        }
    ]
    runner = Fleet(root, CLI(root, args.cli_python), tasks, 1, root / args.state)
    write_once(
        root / ".runtime/task-reference-candidate-admission-v1.json",
        canonical(runner.reference_admission),
    )
    runner.run()  # Always owns metering/retrieval/verified cleanup, including failures.

    def expired(signum, frame):
        raise TimeoutError("post-shutdown candidate acceptance deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(spec["post_stop_verification_seconds"])
    try:
        result = accept(root, root / args.state, root / args.output)
        print("Reference fitted gate:", result["fitted_gate"]["passed"], flush=True)
    finally:
        signal.alarm(0)
