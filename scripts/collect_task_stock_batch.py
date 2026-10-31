"""Bounded non-repeating batches from the fixed historical stock master selection."""
import argparse
import json
from pathlib import Path
import re

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.download import identity
from rl_stocks.data.providers.sec.filing_index import primary_html
from rl_stocks.data.providers.sec.html_source import acquire_filing_html
from rl_stocks.data.providers.sec.lineage import conservative_availability
from rl_stocks.inference.bundle import checksum
from review_task_type_covers import normalized_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selection_path = Path('results/task-conditioning-2026-10-31/stock-master-selection.json')
    selection = json.loads(selection_path.read_text())
    if selection['schema'] != 'task-stock-master-selection-v1':
        raise ValueError('fixed master source selection required')
    pool = [r for r in selection['records'] if r['collection_needed']]
    if len(pool) != 69 or not 0 <= args.batch < 9:
        raise ValueError('fixed69-source/nine-batch scope required')
    cases = [dict(r, status='not_attempted') for r in pool[args.batch * 8:(args.batch + 1) * 8]]
    claim = Path(f'.runtime/task-stock-master-batch-{args.batch}-claim-v1.json')
    stop = Path('.runtime/task-stock-master-collection-stop-v1.json')
    if args.output.exists() or claim.exists() or stop.exists():
        raise ValueError('source batch already claimed or collection stopped; no automatic retry')
    for path, value in selection['protected_sha256'].items():
        if checksum(path) != value:
            raise ValueError('original source selection input changed')
    protected = {str(p): checksum(p) for p in
                 (selection_path, Path(__file__), Path('.runtime/usage-ledger.json'),
                  Path('src/rl_stocks/data/providers/sec/html_source.py'),
                  Path('src/rl_stocks/data/providers/sec/filing_index.py'),
                  Path('src/rl_stocks/data/providers/sec/lineage.py'),
                  Path('scripts/review_task_type_covers.py'))}
    result = {'schema': 'task-stock-master-batch-v1', 'batch': args.batch,
              'cases': cases, 'selection_sha256': checksum(selection_path),
              'protected_sha256': protected, 'maximum_acquisition_operations': len(cases) * 2,
              'acquisition_operations_attempted': 0, 'automatic_retry': False,
              'historical_metadata_admitted': False, 'training_performed': False,
              'colab_allocated': False, 'model_inputs_enabled': False}
    ua = identity()
    write_once(claim, (json.dumps(result, indent=2) + '\n').encode())
    failure = None
    try:
        for case in cases:
            row = case['master_record']
            base = (f"https://www.sec.gov/Archives/edgar/data/{row['filing_directory_cik']}/"
                    f"{row['accession'].replace('-', '')}/")
            case['status'] = 'index_started'
            result['acquisition_operations_attempted'] += 1
            path, receipt, cached = acquire_filing_html(
                'data/raw/sec-filings', url=base + row['accession'] + '-index.htm',
                cik=row['filing_directory_cik'], accession=row['accession'], user_agent=ua)
            case['index'] = {'path': str(path), 'receipt': receipt, 'cached': cached}
            text = normalized_text(path)
            filed = re.findall(r'Filing Date (\d{4}-\d{2}-\d{2})', text)
            accepted = re.findall(r'Accepted (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', text)
            if len(filed) != 1 or len(accepted) != 1 or filed[0] != row['filed']:
                raise ValueError('original/master publication metadata mismatch')
            availability = conservative_availability(filed[0], accepted[0])
            if availability['available_time'][:10] > case['first_training_session']:
                raise ValueError('original publication metadata too late for initial training row')
            case['publication'] = {'filed': filed[0], 'accepted_source': accepted[0],
                                   'availability': availability}
            url = primary_html(path, receipt['sha256'], cik=row['filing_directory_cik'],
                               accession=row['accession'], form=row['form'])
            case['status'] = 'primary_started'
            result['acquisition_operations_attempted'] += 1
            path, receipt, cached = acquire_filing_html(
                'data/raw/sec-filings', url=url, cik=row['filing_directory_cik'],
                accession=row['accession'], user_agent=ua)
            case.update(status='retrieved', primary={'path': str(path), 'receipt': receipt,
                                                    'cached': cached})
    except BaseException as error:
        failure = error
        result.update(status='stopped', error_type=type(error).__name__, error=str(error))
        write_once(stop, (json.dumps({'batch': args.batch, 'claim': str(claim),
                                      'error_type': type(error).__name__, 'error': str(error),
                                      'automatic_retry': False}, indent=2) + '\n').encode())
    finally:
        result.setdefault('status', 'complete')
        result['protected_sources_unchanged'] = all(checksum(Path(p)) == value
                                                   for p, value in protected.items())
        write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
        print(result['status'], [(r['symbol'], r['status']) for r in cases])
    if failure is not None:
        raise RuntimeError('stock source collection stopped; retained evidence, no automatic retry') from None
    if not result['protected_sources_unchanged']:
        raise ValueError('protected sources changed')


if __name__ == '__main__':
    main()
