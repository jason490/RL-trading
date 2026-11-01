"""Fixed ABBV spinoff and DIS predecessor primary-source collection, no fallback."""
import argparse
import io
import json
from pathlib import Path
import re
from zipfile import ZipFile

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.download import identity
from rl_stocks.data.providers.sec.filing_index import primary_html
from rl_stocks.data.providers.sec.html_source import acquire_filing_html
from rl_stocks.data.providers.sec.lineage import conservative_availability
from rl_stocks.inference.bundle import checksum
from review_task_type_covers import normalized_text

ROOT = Path('results/task-conditioning-2026-10-31')


def verify_candidates(selection):
    """Verify the two exact selected master rows, including the nonfinancial8-K."""
    matches = []
    for name, sha in selection['master_archive_sha256'].items():
        path = Path(name)
        if path.stat().st_size > 10_000_000 or checksum(path) != sha:
            raise ValueError('bounded immutable master archive required')
        with ZipFile(path) as archive:
            if archive.namelist() != ['master.idx'] or archive.getinfo('master.idx').file_size > 100_000_000:
                raise ValueError('single bounded master index required')
            with archive.open('master.idx') as source:
                for line in io.TextIOWrapper(source, encoding='latin1'):
                    fields = line.strip().split('|')
                    if len(fields) != 5:
                        continue
                    for row in selection['records']:
                        if fields == [row['candidate_cik'], fields[1], row['form'],
                                      row['filed'], row['submission_path']]:
                            matches.append({'symbol': row['symbol'], 'company_name': fields[1],
                                            'master_archive': name, 'master_sha256': sha})
    if sorted(r['symbol'] for r in matches) != ['ABBV', 'DIS']:
        raise ValueError('one exact official master row per fixed exception required')
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    claim = Path('.runtime/task-type-exception-source-claim-v1.json')
    if args.output.exists() or claim.exists():
        raise ValueError('exception source collection already claimed; no automatic retry')
    selection_path = ROOT / 'missing-stock-source-candidates.json'
    type_path = ROOT / 'type-evidence.json'
    selection = json.loads(selection_path.read_text())
    if [(r['symbol'], r['accession']) for r in selection['records']] != [
            ('ABBV', '0001047469-13-000017'), ('DIS', '0001193125-09-245848')]:
        raise ValueError('fixed original spinoff/predecessor selections required')
    master_proof = verify_candidates(selection)
    sessions = {r['symbol']: r['first_training_session'] for r in json.loads(type_path.read_text())['records']}
    protected = {str(p): checksum(p) for p in
                 (selection_path, type_path, Path(__file__), Path('.runtime/usage-ledger.json'),
                  Path('src/rl_stocks/data/providers/sec/filing_index.py'),
                  Path('src/rl_stocks/data/providers/sec/html_source.py'),
                  Path('src/rl_stocks/data/providers/sec/lineage.py'),
                  Path('scripts/review_task_type_covers.py'))}
    protected.update(selection['master_archive_sha256'])
    result = {'schema': 'task-exception-source-collection-v1', 'cases': [
        {**r, 'first_training_session': sessions[r['symbol']], 'status': 'not_attempted'}
        for r in selection['records']], 'master_proof': master_proof,
        'protected_sha256': protected, 'maximum_acquisition_operations': 4,
        'acquisition_operations_attempted': 0, 'automatic_retry': False,
        'historical_identity_verified': False, 'historical_metadata_admitted': False,
        'training_performed': False, 'colab_allocated': False}
    ua = identity()
    write_once(claim, (json.dumps(result, indent=2) + '\n').encode())
    failure = None
    try:
        for case in result['cases']:
            cik, accession = case['candidate_cik'], case['accession']
            base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/"
            case['status'] = 'index_started'
            result['acquisition_operations_attempted'] += 1
            path, receipt, cached = acquire_filing_html('data/raw/sec-filings',
                url=base + accession + '-index.htm', cik=cik, accession=accession, user_agent=ua)
            case['index'] = {'path': str(path), 'receipt': receipt, 'cached': cached}
            text = normalized_text(path)
            filed = re.findall(r'Filing Date (\d{4}-\d{2}-\d{2})', text)
            accepted = re.findall(r'Accepted (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', text)
            if len(filed) != 1 or len(accepted) != 1 or filed[0] != case['filed']:
                raise ValueError('original/master publication mismatch')
            availability = conservative_availability(filed[0], accepted[0])
            if availability['available_time'][:10] > case['first_training_session']:
                raise ValueError('exception source too late for initial training session')
            case['publication'] = {'filed': filed[0], 'accepted_source': accepted[0],
                                   'availability': availability}
            url = primary_html(path, receipt['sha256'], cik=cik, accession=accession, form=case['form'])
            case['status'] = 'primary_started'
            result['acquisition_operations_attempted'] += 1
            path, receipt, cached = acquire_filing_html('data/raw/sec-filings', url=url,
                cik=cik, accession=accession, user_agent=ua)
            case.update(status='retrieved', primary={'path': str(path), 'receipt': receipt,
                                                    'cached': cached})
    except BaseException as error:
        failure = error
        result.update(status='stopped', error_type=type(error).__name__, error=str(error))
    finally:
        result.setdefault('status', 'complete')
        result['protected_sources_unchanged'] = all(checksum(path) == sha for path, sha in protected.items())
        write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
        print(result['status'], [(r['symbol'], r['status']) for r in result['cases']])
    if failure is not None:
        raise RuntimeError('exception source collection stopped; no automatic retry') from None
    if not result['protected_sources_unchanged']:
        raise ValueError('protected source changed')


if __name__ == '__main__':
    main()
