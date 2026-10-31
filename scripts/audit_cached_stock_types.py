"""Inspect only existing pre-training SEC source instances; never download or admit."""
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.instrument_type import inspect_stock_type
from rl_stocks.inference.bundle import checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new stock evidence output required')
    audit_path = Path('results/task-conditioning-2026-10-31/type-evidence.json')
    ledger_path = Path('artifacts/filings/original-candidate-ledger.json')
    source_audit = json.loads(audit_path.read_text())
    source_rows = source_audit['records']
    ledger = json.loads(ledger_path.read_text())
    if ledger['source_dataset'] != source_audit['dataset_identity']:
        raise ValueError('same fixed source dataset required')
    protected = {str(p): checksum(p) for p in
                 (audit_path, ledger_path, Path('.runtime/usage-ledger.json'),
                  Path('scripts/audit_cached_stock_types.py'),
                  Path('src/rl_stocks/data/providers/sec/instrument_type.py'))}
    cache = Path('data/raw/sec-xbrl')
    rows = []
    for item in source_rows:
        if item['kind'] != 'stock':
            continue
        candidates = [r for r in ledger['records'] if item['symbol'] in r['discovery_symbols']
                      and r['archive_accepted'][:10] < item['first_training_session']
                      and r['archive_filed'] < item['first_training_session'].replace('-', '')]
        documents = []
        for record in sorted(candidates, key=lambda r: (r['archive_accepted'], r['accession'])):
            receipt_path = cache / 'original-receipts' / (
                hashlib.sha256(record['source_url'].encode()).hexdigest() + '.json')
            if not receipt_path.exists():
                continue
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get('schema') != 'sec-original-acquisition-v1'
                    or not re.fullmatch(r'[a-f0-9]{64}', receipt.get('source_sha256', ''))
                    or receipt['source_url'] != record['source_url']):
                raise ValueError('original receipt endpoint mismatch')
            path = cache / receipt['source_sha256'] / 'instance.xml'
            if path.stat().st_size != receipt['bytes']:
                raise ValueError('original receipt byte count mismatch')
            protected[str(receipt_path)] = checksum(receipt_path)
            protected[str(path)] = receipt['source_sha256']
            day = record['archive_filed']
            evidence = inspect_stock_type(path, receipt['source_sha256'], cik=record['cik'],
                                         symbol=item['symbol'], form=record['form'],
                                         filed=day[:4] + '-' + day[4:6] + '-' + day[6:])
            documents.append({'accession': record['accession'], 'source_url': record['source_url'],
                              'accepted': record['archive_accepted'], 'evidence': evidence})
        status = ('exact_document_hint' if any(d['evidence']['document_exact_stock_identity_hint']
                                              for d in documents) else
                  'cached_without_exact_ticker_hint' if documents else 'no_cached_earlier_instance')
        rows.append({'symbol': item['symbol'], 'first_training_session': item['first_training_session'],
                     'status': status, 'documents': documents})
    if not all(checksum(Path(p)) == value for p, value in protected.items()):
        raise ValueError('protected source bytes changed')
    result = {'schema': 'task-cached-stock-type-audit-v1', 'records': rows,
              'status_counts': dict(Counter(r['status'] for r in rows)),
              'protected_sha256': protected, 'protected_sources_unchanged': True,
              'historical_metadata_admitted': False, 'validity_interval_inferred': False,
              'model_inputs_enabled': False, 'network_requests': 0,
              'colab_allocated': False, 'training_performed': False}
    write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print(result['status_counts'])


if __name__ == '__main__':
    main()
