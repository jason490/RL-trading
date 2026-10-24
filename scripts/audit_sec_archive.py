"""Inspect a local SEC FSDS archive without network, model inputs or fitting."""
import argparse
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.archive import audit_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--quarter', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_archive(args.archive, args.quarter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(args.output, (json.dumps(result, indent=2, sort_keys=True) + '\n').encode())
    print('Structural audit saved; causal readiness remains unverified.')


if __name__ == '__main__':
    main()
