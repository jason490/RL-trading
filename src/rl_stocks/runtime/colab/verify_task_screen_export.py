"""Asynchronous inference-only full-cell acceptance; never optimizes locally."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from stable_baselines3 import SAC

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.runtime.colab.task_screen_artifacts import extract_screen_transport
from rl_stocks.training.task_screen_verify import verify_cell


@contextmanager
def no_optimization():
    def forbidden(*args, **kwargs):
        raise RuntimeError('local verification cannot optimize')
    saved = [(owner, name, getattr(owner, name)) for owner, name in
             [(SAC, 'learn'), (SAC, 'train'), (torch.optim.Adam, 'step'), (torch.Tensor, 'backward')]]
    try:
        for owner, name, _ in saved:
            setattr(owner, name, forbidden)
        yield
    finally:
        for owner, name, value in saved:
            setattr(owner, name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'plan', 'output', 'retrieved'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dataset-id', required=True)
    args = parser.parse_args()
    root = Path.cwd()
    plan = json.loads(args.plan.read_text())
    with no_optimization(), torch.no_grad():
        completion = extract_screen_transport(args.archive, plan, args.dataset_id, args.retrieved)
        data = read_dataset(root / 'data/datasets' / args.dataset_id)
        evidence = json.loads((root / 'configs/universes/task_types_research_initial_v1.json').read_text())
        role = plan['task_screen']['task_role']
        proof = verify_cell(args.retrieved / 'artifacts/task-screen', data, evidence,
            role=role, source=completion['source_hash'], git=completion['source_git'])
        runs = [dict(algorithm='task_sac_screen', seed=11, actual_steps=100000,
                     bundle_id=proof['frozen_bundles']['final'], status='complete', cell=proof)]
        write_once(args.output, canonical(runs))
    print('Task screen checkpoints independently replayed; no local optimization.')


if __name__ == '__main__':
    main()
