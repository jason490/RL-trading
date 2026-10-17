import argparse
import json
from pathlib import Path

from rl_stocks.reporting.universe import report_universe


def main():
    parser = argparse.ArgumentParser(description='Report the fixed universe comparison; validation only.')
    parser.add_argument('roots', type=Path, nargs='+')
    parser.add_argument('--protocol', default='configs/experiments/universe_expansion.json')
    parser.add_argument('--data', default='.runtime/universe-expansion-data.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = report_universe(args.roots, json.loads(Path(args.protocol).read_text()),
                            json.loads(Path(args.data).read_text()), args.output, Path.cwd())
    print('Completed cells:', value['completed_cells'], 'Complete:', value['complete'])


if __name__ == '__main__':
    main()
