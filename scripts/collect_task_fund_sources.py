"""Fixed four-fund historical document collection from immutable master indexes."""
import argparse
import json
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.download import identity
from rl_stocks.data.providers.sec.filing_index import primary_html
from rl_stocks.data.providers.sec.html_source import acquire_filing_html
from rl_stocks.data.providers.sec.master import read_master
from rl_stocks.inference.bundle import checksum

SOURCES = (
    ('SPY', '884394', '0000950123-10-005702', '485BPOS'),
    ('QQQ', '1067839', '0001104659-10-002985', '485BPOS'),
    ('DIA', '1041130', '0000950123-10-018164', '485BPOS'),
    ('IWM', '19617', '0000891092-09-004052', '424B2'),
)
ARCHIVES = ('53a29a28cf0876933843344df7663038269b0843e259f8af4892b2156ff1e913',
            '4d86c5196cc0c55fdbfd34eced3ea10b2a554ae80e17df5a1f1a4fba46b01670')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    claim = Path('.runtime/task-type-fund-source-claim-v1.json')
    if args.output.exists() or claim.exists():
        raise ValueError('fund collection already claimed; no automatic retry')
    protected = {str(p): checksum(p) for p in
                 (Path(__file__), Path('.runtime/usage-ledger.json'),
                  Path('src/rl_stocks/data/providers/sec/master.py'),
                  Path('src/rl_stocks/data/providers/sec/html_source.py'),
                  Path('src/rl_stocks/data/providers/sec/filing_index.py'))}
    catalog = []
    for sha in ARCHIVES:
        archive = Path('data/raw/sec-masters') / sha / 'archive.zip'
        protected[str(archive)] = sha
        catalog.extend(read_master(archive, sha, {r[1] for r in SOURCES},
                                   forms={'485BPOS', '424B2'})['records'])
    cases = []
    for symbol, cik, accession, form in SOURCES:
        candidates = [r for r in catalog if r['cik'] == cik and r['accession'] == accession
                      and r['form'] == form and r['filed'] < '2010-03-16']
        if len(candidates) != 1:
            raise ValueError('exact pre-training master row required')
        cases.append({'symbol': symbol, 'master_record': candidates[0], 'status': 'not_attempted',
                      'source_is_fund_registrant': symbol != 'IWM'})
    ua = identity()
    result = {'schema': 'task-fund-source-collection-v1', 'cases': cases,
              'protected_sha256': protected, 'maximum_acquisition_operations': 8,
              'acquisition_operations_attempted': 0, 'historical_metadata_admitted': False,
              'automatic_retry': False, 'colab_allocated': False, 'training_performed': False}
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
    finally:
        result.setdefault('status', 'complete')
        result['protected_sources_unchanged'] = all(checksum(Path(p)) == value
                                                   for p, value in protected.items())
        write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
        print(result['status'], [(r['symbol'], r['status']) for r in cases])
    if failure is not None:
        raise RuntimeError('fund collection stopped; retained evidence, no automatic retry') from None
    if not result['protected_sources_unchanged']:
        raise ValueError('protected source changed')


if __name__ == '__main__':
    main()
