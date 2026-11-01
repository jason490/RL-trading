"""One fixed fund registration to corroborate the QQQQ/QQQ identifier bridge."""
import argparse
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.download import identity
from rl_stocks.data.providers.sec.filing_index import primary_html
from rl_stocks.data.providers.sec.html_source import acquire_filing_html
from rl_stocks.data.providers.sec.master import read_master
from rl_stocks.inference.bundle import checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    claim = Path('.runtime/task-qqq-alias-source-claim-v1.json')
    if args.output.exists() or claim.exists():
        raise ValueError('QQQ alias source already claimed; no automatic retry')
    sha = '1f3c48461e68aebfa6a19ba256d0b069fc8ab01e26fe8c3f93876afbb3bd2b82'
    archive = Path('data/raw/sec-masters') / sha / 'archive.zip'
    rows = [r for r in read_master(archive, sha, {'1067839'}, forms={'485BPOS'})['records']
            if r['accession'] == '0001104659-12-005092' and r['filed'] == '2012-01-30']
    if len(rows) != 1:
        raise ValueError('exact selected original fund registration required')
    row = rows[0]
    protected = {str(p): checksum(p) for p in
                 (Path(__file__), Path('.runtime/usage-ledger.json'), archive,
                  Path('src/rl_stocks/data/providers/sec/filing_index.py'),
                  Path('src/rl_stocks/data/providers/sec/html_source.py'))}
    result = {'schema': 'task-qqq-alias-collection-v1', 'master_record': row,
              'master_sha256': sha, 'maximum_acquisition_operations': 2,
              'acquisition_operations_attempted': 0, 'automatic_retry': False,
              'protected_sha256': protected, 'historical_metadata_admitted': False,
              'training_performed': False, 'colab_allocated': False,
              'purpose': 'Identifier/type-invariance review only; never backdate the2012 publication'}
    ua = identity()
    write_once(claim, (json.dumps(result, indent=2) + '\n').encode())
    failure = None
    try:
        base = ('https://www.sec.gov/Archives/edgar/data/1067839/'
                + row['accession'].replace('-', '') + '/')
        result['acquisition_operations_attempted'] += 1
        path, receipt, cached = acquire_filing_html('data/raw/sec-filings',
            url=base + row['accession'] + '-index.htm', cik='1067839',
            accession=row['accession'], user_agent=ua)
        result['index'] = {'path': str(path), 'receipt': receipt, 'cached': cached}
        url = primary_html(path, receipt['sha256'], cik='1067839',
                           accession=row['accession'], form='485BPOS')
        result['acquisition_operations_attempted'] += 1
        path, receipt, cached = acquire_filing_html('data/raw/sec-filings', url=url,
            cik='1067839', accession=row['accession'], user_agent=ua)
        result.update(status='complete', primary={'path': str(path), 'receipt': receipt, 'cached': cached})
    except BaseException as error:
        failure = error
        result.update(status='stopped', error_type=type(error).__name__, error=str(error))
    finally:
        result['protected_sources_unchanged'] = all(checksum(path) == value
                                                   for path, value in protected.items())
        write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
        print(result['status'])
    if failure is not None:
        raise RuntimeError('QQQ source collection stopped; no automatic retry') from None
    if not result['protected_sources_unchanged']:
        raise ValueError('protected source changed')


if __name__ == '__main__':
    main()
