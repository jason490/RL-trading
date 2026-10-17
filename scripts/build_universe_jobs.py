"""Prepare the six fixed PPO universe cells or one lower-credit smoke task."""
import argparse
import json
from pathlib import Path

from rl_stocks.runtime.colab.jobs import single_smoke_plan, study_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default='configs/experiments/universe_expansion.json')
    parser.add_argument('--data', default='.runtime/universe-expansion-data.json')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    data = json.loads(Path(args.data).read_text())
    if (protocol['algorithm'] != 'ppo' or protocol['steps'] != 50000
            or protocol['seeds'] != [11, 23, 37] or protocol['arms'] != ['control', 'expanded']
            or protocol['device'] != 'cpu' or protocol['reward_scale'] != 1.0):
        raise ValueError('Unsupported fixed universe experiment')
    tasks = []
    for seed in ([11] if args.smoke else protocol['seeds']):
        for arm in (['expanded'] if args.smoke else protocol['arms']):
            tasks.append({'id': arm + '-ppo-' + str(seed), 'dataset': data['datasets'][arm]['path'],
                          'plan': single_smoke_plan('ppo') if args.smoke else study_plan('ppo', 'CPU', [seed])})
    Path(args.output).write_text(json.dumps(tasks, indent=2) + '\n')
    print('Prepared', len(tasks), 'CPU tasks; no sessions allocated.')


if __name__ == '__main__':
    main()
