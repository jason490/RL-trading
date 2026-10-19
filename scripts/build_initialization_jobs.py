"""Prepare bounded CPU initialization-study tasks; never provisions sessions."""
import argparse
import json
from pathlib import Path
from rl_stocks.runtime.colab.jobs import initialization_plan
from rl_stocks.training.initialization import ARMS
from rl_stocks.core.contracts import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('configs/experiments/action_initialization.json'))
    parser.add_argument('--phase', choices=('smoke', 'screen', 'confirm'), required=True)
    parser.add_argument('--candidate', choices=ARMS[1:])
    parser.add_argument('--screen-summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if (protocol['arms'] != list(ARMS) or protocol['steps_per_cell'] != 50000
            or protocol['device'] != 'cpu' or protocol['reward_scale'] != 1.):
        raise ValueError('unsupported initialization protocol')
    if (args.phase == 'confirm') != bool(args.candidate):
        raise ValueError('confirmation requires a promoted candidate; other phases reject it')
    if args.phase == 'confirm':
        if args.screen_summary is None:
            raise ValueError('verified screen summary required before confirmation')
        screen = json.loads(args.screen_summary.read_text())
        if (screen['protocol_sha256'] != digest(protocol) or not screen['screen']['passed']
                or screen['screen']['promoted_candidate'] != args.candidate):
            raise ValueError('candidate did not pass this declared screen')
    elif args.screen_summary is not None:
        raise ValueError('screen evidence applies only to confirmation')
    arms = ('standard', args.candidate) if args.phase == 'confirm' else ARMS
    seeds = (23, 37) if args.phase == 'confirm' else (11,)
    tasks = [{'id': f'{arm}-{seed}', 'dataset': 'data/datasets/' + protocol['dataset'],
              'plan': initialization_plan(arm, seed, args.phase == 'smoke')}
             for seed in seeds for arm in arms]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, indent=2) + '\n')
    print(f'Prepared {len(tasks)} {args.phase} tasks; no sessions allocated.')


if __name__ == '__main__':
    main()
