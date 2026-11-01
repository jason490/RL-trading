"""Prepare the complete reviewed seed11 task matrix after physical profile checks."""
import argparse
from pathlib import Path

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.runtime.colab.budget import Ledger
from rl_stocks.runtime.colab.jobs import reviewed_task_screen_plan
from rl_stocks.training.task_execution import verify_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    acceptance = verify_execution(root, Ledger(root / '.runtime/usage-ledger.json'))
    tasks = []
    for role in ('masked_shared', 'tagged_shared', 'tagged_routed'):
        plan = reviewed_task_screen_plan(role)
        tasks.append(dict(id='task-screen-' + role.replace('_', '-'),
                          dataset='data/datasets/' + plan['dataset_contract']['identity'], plan=plan))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(args.output, canonical(tasks))
    print('Prepared three reviewed cells; no allocation. Acceptance:', acceptance)


if __name__ == '__main__':
    main()
