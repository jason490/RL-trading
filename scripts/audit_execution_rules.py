"""Compare fixed causal rules under policy execution constraints; no training."""
import argparse
import csv
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
from statistics import mean

from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import normalizer_from, read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.evaluation.rules import ExecutionRule
from rl_stocks.evaluation.runner import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--universe', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    dataset = read_dataset(args.dataset)
    universe = json.loads(args.universe.read_text())
    stocks = {i['symbol'] for i in universe['instruments'] if i['kind'] == 'stock'}
    normalizer = normalizer_from(dataset['normalizer'])
    config = SimulationConfig()
    records, summaries = [], []
    rules = ('cash', 'long', 'short', 'ema_trend')
    protocol = {'rules': list(rules), 'nonzero_exposure': .95,
                'protective_action': [1., 1., 1.], 'simulation': asdict(config),
                'split': 'validation', 'dataset': args.dataset.name,
                'universe_sha256': digest(universe),
                'purpose': 'fixed causal execution diagnostic; no learned policy or checkpoint selection'}
    source_root = Path(__file__).resolve().parents[1] / 'src' / 'rl_stocks'
    protocol['source_sha256'] = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in ('evaluation/rules.py', 'evaluation/runner.py', 'evaluation/direction.py',
                     'environment/execution/simulator.py', 'environment/accounting/portfolio.py')}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2, allow_nan=False) + '\n')
    for rule in rules:
        results = evaluate(ExecutionRule(rule, normalizer), dataset, config=config)
        for result in results.values():
            records.append({'rule': rule, 'symbol': result['symbol'],
                            'kind': 'stock' if result['symbol'] in stocks else 'etf',
                            'net_return': result['metrics']['net_return'],
                            'buy_hold_return': result['baselines']['buy_hold']['metrics']['net_return'],
                            'max_drawdown': result['metrics']['max_drawdown'],
                            'costs': result['metrics']['costs'], **result['directional']})
        groups = {}
        for kind in ('stock', 'etf'):
            rows = [r for r in records if r['rule'] == rule and r['kind'] == kind]
            groups[kind] = {'count': len(rows), 'net_return': mean(r['net_return'] for r in rows),
                            'buy_hold_return': mean(r['buy_hold_return'] for r in rows),
                            'excess_return': mean(r['net_return'] - r['buy_hold_return'] for r in rows),
                            'max_drawdown': mean(r['max_drawdown'] for r in rows)} if rows else None
        summaries.append({'rule': rule, 'groups': groups,
                          'declining_stocks': declining_stock_summary(results, stocks)})
        print(json.dumps({'rule': rule, 'groups': groups}), flush=True)
    (args.output / 'summary.json').write_text(json.dumps(
        {'protocol': protocol, 'summaries': summaries}, indent=2, allow_nan=False) + '\n')
    with (args.output / 'per-symbol.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


if __name__ == '__main__':
    main()
