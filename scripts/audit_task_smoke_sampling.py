"""Nominal fixed smoke starts only; no environment rollout or learner construction."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from rl_stocks.core.contracts import digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.sampling.windows import StockEtfCycles
from rl_stocks.inference.bundle import checksum
from rl_stocks.training.task_conditioning import protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new sampling audit output required')
    spec = protocol()
    folder = Path('data/datasets') / spec['dataset']
    protected = {str(p): checksum(p) for p in
                 (folder / 'dataset.json', Path('.runtime/usage-ledger.json'), Path(__file__),
                  Path('src/rl_stocks/environment/sampling/windows.py'),
                  Path('configs/experiments/task_conditioning.json'))}
    data = read_dataset(folder)
    if digest(data['normalizer']) != spec['normalizer_sha256']:
        raise ValueError('original immutable preprocessing required')
    window = spec['sampling']['smoke_window_transitions']
    if spec['smoke_steps'] % window:
        raise ValueError('complete nominal smoke windows required')
    episodes = [e for e in data['episodes'].values() if e['split'] == 'train']
    sampler = StockEtfCycles(episodes, window)
    rng = np.random.default_rng(spec['screen_seed'])
    starts = []
    for _ in range(spec['smoke_steps'] // window):
        index, start = sampler.choose(rng)
        symbol = episodes[index]['symbol']
        starts.append({'symbol': symbol, 'start': start,
                       'group': 'etf' if symbol in sampler.etfs else 'stock'})
    groups = dict(Counter(r['group'] for r in starts))
    if set(groups) != {'stock', 'etf'}:
        raise ValueError('fixed smoke nominal starts miss a required group; no seed sweep')
    if not all(checksum(p) == sha for p, sha in protected.items()):
        raise ValueError('protected source changed')
    result = {'schema': 'task-smoke-nominal-sampling-v1', 'seed': spec['screen_seed'],
              'dataset': spec['dataset'], 'starts': starts, 'group_start_counts': groups,
              'protected_sha256': protected, 'protected_sources_unchanged': True,
              'actual_transition_coverage_verified': False,
              'limitation': 'Full-window nominal schedule only; actual resets, terminations, replay and gradients require Colab smoke acceptance',
              'model_constructed': False, 'training_performed': False, 'colab_allocated': False,
              'seed_or_sampling_changed': False, 'historical_metadata_admitted': False}
    write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print('Fixed nominal smoke starts:', groups, '; actual coverage remains unverified')


if __name__ == '__main__':
    main()
