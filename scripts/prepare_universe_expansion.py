"""Cache a declared retrospective universe and create matched train/validation panels."""
import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path

from rl_stocks.core.contracts import Request
from rl_stocks.data.accumulation.backfill import accumulate
from rl_stocks.data.generation.datasets import generate, read_dataset
from rl_stocks.data.providers.yahoo.adapter import YahooProvider
from rl_stocks.data.storage.immutable import load
from rl_stocks.data.universe.config import read_universe, symbols, history_starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default='configs/experiments/universe_expansion.json')
    parser.add_argument('--output', default='.runtime/universe-expansion-data.json')
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    expanded = read_universe(protocol['expanded_universe'])
    control = read_universe(protocol['control_universe'])
    request = Request(symbols(expanded), date.fromisoformat(protocol['start']),
                      date.fromisoformat(protocol['end_exclusive']))
    if request.end != date(2024, 1, 1) or protocol['included_splits'] != ['train', 'validation']:
        raise ValueError('This protocol excludes the retired test interval')
    print('Caching', len(request.symbols), 'symbols through 2023; failures are not excluded.', flush=True)
    snapshot = accumulate(YahooProvider(), request, 'data/raw/universe-expansion',
                          history_starts=history_starts(expanded), full_range=True)
    _, provenance = load(snapshot)
    failures = {symbol: value for symbol, value in provenance['coverage'].items()
                if value['missing'] or not value['calendar_verified']}
    if failures:
        Path('.runtime/universe-expansion-coverage-failures.json').write_text(json.dumps(failures, indent=2))
        raise ValueError('Coverage failures retained; inspect .runtime/universe-expansion-coverage-failures.json')
    result = {'snapshot': str(snapshot), 'datasets': {}, 'coverage': provenance['coverage'],
              'test_evaluated': False, 'survivorship': protocol['survivorship']}
    for arm, universe in (('control', control), ('expanded', expanded)):
        destination = generate(snapshot, 'data/datasets', universe=universe,
                               evaluation_symbols=protocol['evaluation_symbols'], include_test=False)
        data = read_dataset(destination)
        result['datasets'][arm] = {'path': str(destination), 'id': destination.name,
            'training_symbols': sorted({e['symbol'] for e in data['episodes'].values() if e['split'] == 'train'}),
            'rows': dict(Counter({split: sum(len(e['bars']) for e in data['episodes'].values()
                                         if e['split'] == split) for split in ('train', 'validation')}))}
        print(arm, destination, result['datasets'][arm]['rows'], flush=True)
    a, b = [read_dataset(result['datasets'][arm]['path']) for arm in ('control', 'expanded')]
    validation = lambda data: {k: v for k, v in data['episodes'].items() if v['split'] == 'validation'}
    if a['schema'] != b['schema'] or a['names'] != b['names'] or validation(a) != validation(b):
        raise ValueError('Unmatched validation observations or features')
    common = set(result['datasets']['control']['training_symbols'])
    training = lambda data: {k: v for k, v in data['episodes'].items()
                             if v['split'] == 'train' and v['symbol'] in common}
    if training(a) != training(b):
        raise ValueError('Unmatched common training observations')
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print('Matched panels verified; no training or Colab allocation performed.', flush=True)


if __name__ == '__main__':
    main()
