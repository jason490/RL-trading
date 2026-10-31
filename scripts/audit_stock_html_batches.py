"""Read-only review hints from explicitly selected complete source batches."""
import argparse
from collections import Counter
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.html_stock_type import inspect_html_stock
from rl_stocks.inference.bundle import checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', required=True, help='Fixed comma-separated batch numbers')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    batches = [int(x) for x in args.batches.split(',')]
    if batches != sorted(set(batches)) or not batches or not set(batches) <= set(range(9)):
        raise ValueError('unique ordered fixed source batch numbers required')
    if args.output.exists():
        raise ValueError('fresh HTML review output required')
    root = Path('results/task-conditioning-2026-10-31')
    protected = {str(p): checksum(p) for p in
                 (Path(__file__), Path('src/rl_stocks/data/providers/sec/html_stock_type.py'),
                  Path('.runtime/usage-ledger.json'))}
    records = []
    selection = None
    for batch in batches:
        source = root / f'stock-master-batch-{batch}.json'
        collected = json.loads(source.read_text())
        if collected['status'] != 'complete' or collected['batch'] != batch:
            raise ValueError('explicit complete source batches required')
        if selection is None:
            selection = collected['selection_sha256']
        elif selection != collected['selection_sha256']:
            raise ValueError('same original master source selection required')
        protected[str(source)] = checksum(source)
        for case in collected['cases']:
            if case['status'] != 'retrieved':
                raise ValueError('complete original source records required')
            item = case['primary']
            protected[item['path']] = item['receipt']['sha256']
            result = inspect_html_stock(item['path'], item['receipt']['sha256'], symbol=case['symbol'])
            records.append({'symbol': case['symbol'], 'batch': batch,
                            'source_url': item['receipt']['source_url'],
                            'accession': case['master_record']['accession'],
                            'publication': case['publication'], 'hints': result})
    if len({r['symbol'] for r in records}) != len(records):
        raise ValueError('source symbols must not repeat across selected batches')
    if not all(checksum(p) == value for p, value in protected.items()):
        raise ValueError('protected sources changed')
    counts = Counter({'documents': len(records)})
    for row in records:
        counts['common_equity_cover_hints'] += row['hints']['common_equity_cover_hint']
        counts['explicit_ticker_clause_hints'] += row['hints']['explicit_ticker_clause_hint']
    result = {'schema': 'task-stock-html-batches-v1', 'batches': batches,
              'selection_sha256': selection, 'records': records, 'counts': dict(counts),
              'protected_sha256': protected, 'protected_sources_unchanged': True,
              'historical_metadata_admitted': False, 'network_requests': 0,
              'training_performed': False, 'colab_allocated': False}
    write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print(dict(counts))


if __name__ == '__main__':
    main()
