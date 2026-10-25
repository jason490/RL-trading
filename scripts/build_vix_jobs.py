"""Prepare fixed CPU VIX jobs; actual smoke/runtime acceptance gates screens."""

import argparse
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.runtime.colab.jobs import vix_plan
from rl_stocks.training.vix_spec import (
    PREREGISTRATION_SHA256 as PROTOCOL_SHA256,
    preregistration as protocol,
)
from rl_stocks.training.vix_runtime import make_runtime_profile
from rl_stocks.training.vix_assets import binding


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
            parser.error("fixed passing VIX screen required")
    profile = None if args.smoke else make_runtime_profile(Path.cwd())
    tasks = [
        dict(
            id=f"vix-{arm}-{args.seed}" + ("-smoke" if args.smoke else ""),
            dataset="data/datasets/" + binding()["datasets"][arm],
            plan=vix_plan(arm, args.seed, args.smoke, profile),
        )
        for arm in spec["arms"]
    ]
    write_once(args.output, (json.dumps(tasks, indent=2) + "\n").encode())
    print("Prepared3 CPU VIX tasks; no allocation or training.")


if __name__ == "__main__":
    main()
