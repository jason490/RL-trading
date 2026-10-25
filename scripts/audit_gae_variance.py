"""Run the fixed read-only actual-buffer trace diagnostic; never trains."""

import argparse
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.reporting.gae_variance import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path.cwd())
    write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({a: r["blocks"]["all"] for a, r in result["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
