"""Prepare fixed CPU Treasury jobs; actual smoke/runtime acceptance gates screens."""

import argparse
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.runtime.colab.jobs import treasury_plan
from rl_stocks.training.treasury_spec import PROTOCOL_SHA256, protocol, binding
from rl_stocks.training.treasury_runtime import make_runtime_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--screen-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = protocol()
    if not args.smoke and args.seed != 11:
        if args.screen_summary is None:
            parser.error("confirmation requires the same-protocol passing screen")
        summary = json.loads(args.screen_summary.read_text())
        if (
            summary.get("protocol_sha256") != PROTOCOL_SHA256
            or summary.get("screen", {}).get("passed") is not True
        ):
            parser.error("fixed passing Treasury screen required")
    profile = None if args.smoke else make_runtime_profile(Path.cwd())
    tasks = [
        dict(
            id=f"treasury-{arm}-{args.seed}" + ("-smoke" if args.smoke else ""),
            dataset="data/datasets/" + binding()["datasets"][arm],
            plan=treasury_plan(arm, args.seed, args.smoke, profile),
        )
        for arm in spec["arms"]
    ]
    write_once(args.output, (json.dumps(tasks, indent=2) + "\n").encode())
    print("Prepared3 CPU Treasury tasks; no allocation or training.")


if __name__ == "__main__":
    main()
