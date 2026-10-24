"""Inspect original extracted XBRL money facts without enabling financial inputs."""
import argparse
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.facts import extract_facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    for name in ('sha256', 'accession', 'cik', 'report-end', 'filed'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = extract_facts(args.source, args.sha256, args.accession, args.cik,
                           args.report_end, args.filed)
    write_once(args.output, (json.dumps(result, indent=2) + '\n').encode())
    print('Original fact inspection:', result['counts'])
    print('Publication/identity/feature readiness remains unverified.')


if __name__ == '__main__':
    main()
