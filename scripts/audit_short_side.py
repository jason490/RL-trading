"""Audit saved validation paths without retraining or selecting new checkpoints."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.evaluation.direction import directional_attribution, declining_stock_summary
from rl_stocks.inference.bundle import checksum


def audit(roots, universe, data_root):
    stocks = {i['symbol'] for i in universe['instruments'] if i['kind'] == 'stock'}
    datasets, runs, records, seen = {}, [], [], set()
    matched = None
    for root in roots:
        metadata_path = root / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        if metadata['status'] != 'complete':
            raise ValueError('complete runs required')
        contract = (metadata['dataset'], metadata['simulation'], metadata['requested_steps'])
        if matched is not None and contract != matched:
            raise ValueError('dataset, simulation and requested steps must match')
        matched = contract
        identity = (metadata['algorithm'], metadata['seed'])
        if identity in seen:
            raise ValueError('duplicate algorithm/seed; pass one matched study only')
        seen.add(identity)
        dataset_id = metadata['dataset']
        if dataset_id not in datasets:
            datasets[dataset_id] = read_dataset(data_root / dataset_id)
        dataset = datasets[dataset_id]
        for phase, filename in [('selected', 'selected-validation.json'), ('final', 'validation.json')]:
            path = root / filename
            results = json.loads(path.read_text())
            for key, result in results.items():
                if result['split'] != 'validation':
                    raise ValueError('validation paths required; retired test must not be audited')
                episode = dataset['episodes'][key]
                result['directional'] = directional_attribution(
                    episode, result['rows'], result['events'], metadata['simulation']['initial_equity'])
                records.append({'algorithm': metadata['algorithm'], 'seed': metadata['seed'],
                                'phase': phase, 'symbol': result['symbol'],
                                'kind': 'stock' if result['symbol'] in stocks else 'etf',
                                'policy_return': result['metrics']['net_return'],
                                'buy_hold_return': result['baselines']['buy_hold']['metrics']['net_return'],
                                **result['directional']})
            runs.append({'algorithm': metadata['algorithm'], 'seed': metadata['seed'],
                         'phase': phase, 'dataset': dataset_id,
                         'requested_steps': metadata['requested_steps'],
                         'source_report_sha256': checksum(path),
                         'declining_stocks': declining_stock_summary(results, stocks)})
    seed_sets = [{r['seed'] for r in runs if r['algorithm'] == algorithm}
                 for algorithm in {r['algorithm'] for r in runs}]
    if any(seeds != seed_sets[0] for seeds in seed_sets):
        raise ValueError('matched seed sets required')
    summaries = []
    for algorithm in sorted({r['algorithm'] for r in runs}):
        for phase in ('selected', 'final'):
            group = [r for r in runs if r['algorithm'] == algorithm and r['phase'] == phase]
            cells = [r for g in group for r in g['declining_stocks']['records']]
            summaries.append({'algorithm': algorithm, 'phase': phase,
                              'seeds': sorted(g['seed'] for g in group),
                              'stock_seed_observations': len(cells),
                              'positive_policy_and_short_pnl_count': sum(r['positive_policy_and_short_pnl'] for r in cells),
                              'mean_policy_return': mean(r['policy_return'] for r in cells) if cells else None,
                              'mean_buy_hold_return': mean(r['buy_hold_return'] for r in cells) if cells else None,
                              'mean_short_pnl': mean(r['short_net_pnl'] for r in cells) if cells else None})
    return {'scope': 'retrospective validation diagnostic; not checkpoint selection or independent evidence',
            'runs': runs, 'summaries': summaries}, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path, help='Explicit matched run directories')
    parser.add_argument('--universe', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data/datasets'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result, records = audit(args.roots, json.loads(args.universe.read_text()), args.data_root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (args.output / 'per-symbol.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(result['summaries'], indent=2))


if __name__ == '__main__':
    main()
