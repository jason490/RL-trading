"""Read cached master indexes and freeze a bounded unresolved stock source batch."""
import argparse
from collections import Counter
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.master import read_master
from rl_stocks.data.providers.sec.type_sources import select_stock_sources
from rl_stocks.inference.bundle import checksum

QUARTERS = {'2009q2', '2009q3', '2009q4', '2010q1', '2012q4', '2013q1'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new source selection output required')
    paths = {name: Path('results/task-conditioning-2026-10-31') / name for name in
             ('type-evidence.json', 'cached-stock-evidence.json', 'original-cover-collection.json')}
    ledger_path = Path('artifacts/filings/original-candidate-ledger.json')
    protected = {str(p): checksum(p) for p in [*paths.values(), ledger_path,
                 Path('.runtime/usage-ledger.json'), Path(__file__),
                 Path('src/rl_stocks/data/providers/sec/type_sources.py')]}
    inputs = {name: json.loads(p.read_text()) for name, p in paths.items()}
    ledger = json.loads(ledger_path.read_text())
    if ledger['source_dataset'] != inputs['type-evidence.json']['dataset_identity']:
        raise ValueError('same fixed dataset required')
    ciks = {r['cik'] for r in ledger['records']}
    catalog = []
    seen_quarters = set()
    for receipt_path in sorted(Path('data/raw/sec-masters').glob('*/acquisition-*.json')):
        receipt = json.loads(receipt_path.read_text())
        if receipt['quarter'] not in QUARTERS:
            continue
        if receipt['quarter'] in seen_quarters:
            raise ValueError('one immutable source per selected quarter required')
        seen_quarters.add(receipt['quarter'])
        archive = receipt_path.parent / 'archive.zip'
        protected[str(receipt_path)] = checksum(receipt_path)
        protected[str(archive)] = receipt['archive_sha256']
        catalog.extend(read_master(archive, receipt['archive_sha256'], ciks)['records'])
    if seen_quarters != QUARTERS:
        raise ValueError('all fixed cached master quarters required')
    covered = {r['symbol'] for r in inputs['cached-stock-evidence.json']['records']
               if r['status'] == 'exact_document_hint'}
    covered.update(r['symbol'] for r in inputs['original-cover-collection.json']['cases']
                   if r['status'] == 'retrieved')
    records = select_stock_sources(inputs['type-evidence.json']['records'], ledger['records'],
                                   catalog, covered)
    batch = [r for r in records if r['collection_needed']][:8]
    if len(records) != 107 or len(batch) != 8:
        raise ValueError('fixed107 stock scope and eight unresolved source cases required')
    result = {'schema': 'task-stock-master-selection-v1', 'records': records, 'batch': batch,
              'selection': 'Latest eligible original annual, else latest original quarterly; original acceptance remains unverified; first eight unresolved alphabetically',
              'counts': dict(Counter('collection_needed' if r['collection_needed'] else
                                    'document_already_present' if r['already_has_document_observation']
                                    else 'no_eligible_candidate' for r in records)),
              'protected_sha256': protected, 'maximum_acquisition_operations': 16,
              'historical_metadata_admitted': False, 'network_requests': 0,
              'training_performed': False, 'colab_allocated': False}
    result['protected_sources_unchanged'] = all(checksum(Path(p)) == value
                                               for p, value in protected.items())
    if not result['protected_sources_unchanged']:
        raise ValueError('protected sources changed')
    write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print(result['counts'], 'selected', [r['symbol'] for r in batch])


if __name__ == '__main__':
    main()
