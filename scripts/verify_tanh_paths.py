"""Run the preregistered training-only full-path inference parity check."""

import argparse
import json
from pathlib import Path

from rl_stocks.evaluation.tanh_paths import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.dataset.read_text()), args.bundles, args.output)


if __name__ == "__main__":
    main()
