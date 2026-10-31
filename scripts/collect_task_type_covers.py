"""One bounded original-cover collection: first three cached unresolved stock cases."""
import argparse
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.download import identity
from rl_stocks.data.providers.sec.filing_index import primary_html
from rl_stocks.data.providers.sec.html_source import acquire_filing_html
from rl_stocks.inference.bundle import checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    claim = Path('.runtime/task-type-cover-collection-claim-v1.json')
    if args.output.exists() or claim.exists():
        raise ValueError('collection already claimed; no automatic retry or overwrite')
    source = Path('results/task-conditioning-2026-10-31/cached-stock-evidence.json')
    ledger_path = Path('artifacts/filings/original-candidate-ledger.json')
    evidence = json.loads(source.read_text())
    ledger = json.loads(ledger_path.read_text())
    by_accession = {r['accession']: r for r in ledger['records']}
    selected = sorted((r for r in evidence['records'] if
                       r['status'] == 'cached_without_exact_ticker_hint'),
                      key=lambda r: r['symbol'])[:3]
    cases = []
    for row in selected:
        record = by_accession[row['documents'][0]['accession']]
        cases.append({'symbol': row['symbol'], 'cik': record['cik'],
                      'accession': record['accession'], 'form': record['form'],
                      'first_training_session': row['first_training_session'],
                      'status': 'not_attempted'})
    protected = {str(p): checksum(p) for p in
                 (source, ledger_path, Path('.runtime/usage-ledger.json'),
                  Path('scripts/collect_task_type_covers.py'),
                  Path('src/rl_stocks/data/providers/sec/html_source.py'),
                  Path('src/rl_stocks/data/providers/sec/filing_index.py'))}
    user_agent = identity()  # Private; never serialized or printed.
    result = {'schema': 'task-original-cover-collection-v1', 'cases': cases,
              'protected_sha256': protected, 'maximum_acquisition_operations': 6,
              'acquisition_operations_attempted': 0, 'automatic_retry': False,
              'historical_metadata_admitted': False, 'training_performed': False,
              'colab_allocated': False, 'model_inputs_enabled': False}
    write_once(claim, (json.dumps(result, indent=2) + '\n').encode())
    failure = None
    try:
        for case in cases:
            base = (f"https://www.sec.gov/Archives/edgar/data/{case['cik']}/"
                    f"{case['accession'].replace('-', '')}/")
            case['status'] = 'index_started'
            result['acquisition_operations_attempted'] += 1
            index, index_receipt, cached = acquire_filing_html(
                'data/raw/sec-filings', url=base + case['accession'] + '-index.htm',
                cik=case['cik'], accession=case['accession'], user_agent=user_agent)
            case['index'] = {'path': str(index), 'receipt': index_receipt, 'cached': cached}
            url = primary_html(index, index_receipt['sha256'], cik=case['cik'],
                               accession=case['accession'], form=case['form'])
            case['status'] = 'primary_started'
            result['acquisition_operations_attempted'] += 1
            primary, receipt, cached = acquire_filing_html(
                'data/raw/sec-filings', url=url, cik=case['cik'],
                accession=case['accession'], user_agent=user_agent)
            case.update(status='retrieved', primary={'path': str(primary), 'receipt': receipt,
                                                    'cached': cached})
    except BaseException as error:
        failure = error
        result.update(status='stopped', error_type=type(error).__name__, error=str(error))
    finally:
        result.setdefault('status', 'complete')
        result['protected_sources_unchanged'] = all(checksum(Path(p)) == value
                                                   for p, value in protected.items())
        write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
        print(result['status'], [(r['symbol'], r['status']) for r in cases])
    if failure is not None:
        raise RuntimeError('original-cover collection stopped; inspect retained receipt, no retry') from None
    if not result['protected_sources_unchanged']:
        raise ValueError('protected source bytes changed')


if __name__ == '__main__':
    main()
