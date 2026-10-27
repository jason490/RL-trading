"""Build the single fixed CPU recurrent environment smoke task; no allocation."""
import argparse
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.runtime.colab.jobs import recurrent_environment_plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    plan = recurrent_environment_plan()
    task = dict(id='recurrent-environment-smoke', dataset='data/datasets/' + plan['dataset_contract']['identity'], plan=plan)
    write_once(args.output, (json.dumps([task], indent=2) + '\n').encode())
    print('Prepared one fixed CPU recurrent learned-smoke task; no allocation.')
