"""Verify combined original/resumed source coverage without rewriting failed attempts."""
import argparse
from collections import Counter
import json
import re
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.filing_index import primary_html_joint
from rl_stocks.data.providers.sec.html_stock_type import inspect_html_stock
from rl_stocks.inference.bundle import checksum
from rl_stocks.data.providers.sec.lineage import conservative_availability
from review_task_type_covers import normalized_text

ROOT = Path('results/task-conditioning-2026-10-31')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh combined source verification output required')
    selection_path = ROOT / 'stock-master-selection.json'
    resume_path = ROOT / 'stock-reviewed-resume-result-v2.json'
    selection = json.loads(selection_path.read_text())
    resumed = json.loads(resume_path.read_text())
    if (resumed['status'] != 'complete' or resumed['completed_new_primaries'] != 34
            or resumed['selection_sha256'] != checksum(selection_path)):
        raise ValueError('complete original-plan reviewed resume required')
    if not all(checksum(path) == sha for path, sha in resumed['protected_sha256'].items()):
        raise ValueError('reviewed resume protected source changed')
    protected = {str(p): checksum(p) for p in
                 (selection_path, resume_path, Path(__file__),
                  Path('src/rl_stocks/data/providers/sec/filing_index.py'),
                  Path('src/rl_stocks/data/providers/sec/html_stock_type.py'),
                  Path('src/rl_stocks/data/providers/sec/lineage.py'),
                  Path('scripts/review_task_type_covers.py'),
                  Path('.runtime/task-stock-master-collection-stop-v1.json'),
                  Path('.runtime/usage-ledger.json'))}
    cases = {}
    for batch in range(5):
        path = ROOT / f'stock-master-batch-{batch}.json'
        value = json.loads(path.read_text())
        if (value['status'] != ('stopped' if batch == 4 else 'complete')
                or value['selection_sha256'] != checksum(selection_path)):
            raise ValueError('original complete/failed evidence required')
        protected[str(path)] = checksum(path)
        for case in value['cases']:
            if case['status'] == 'retrieved':
                if case['symbol'] in cases:
                    raise ValueError('duplicate original source symbol')
                cases[case['symbol']] = case
    if len(cases) != 35:
        raise ValueError('retained35-source prefix required')
    for case in resumed['cases']:
        if case['status'] != 'retrieved' or case['symbol'] in cases:
            raise ValueError('new complete non-overlapping source cases required')
        cases[case['symbol']] = case
    expected = {r['symbol']: r for r in selection['records'] if r['collection_needed']}
    if len(cases) != 69 or set(cases) != set(expected):
        raise ValueError('complete exact original69-source selection required')
    archives = {}
    for name, sha in selection['protected_sha256'].items():
        path = Path(name)
        if path.parent.parent.name == 'sec-masters' and path.name.startswith('acquisition-'):
            if checksum(path) != sha:
                raise ValueError('original master receipt changed')
            receipt = json.loads(path.read_text())
            archive = path.parent / 'archive.zip'
            protected[str(path)], protected[str(archive)] = sha, receipt['archive_sha256']
            archives[receipt['quarter']] = {'path': str(archive), 'sha256': receipt['archive_sha256']}
    records = []
    for symbol in sorted(cases):
        case = cases[symbol]
        row = expected[symbol]['master_record']
        if case['master_record'] != row:
            raise ValueError('original selected filing changed')
        for name in ('index', 'primary'):
            item = case[name]
            if (Path(item['path']).stat().st_size != item['receipt']['bytes']
                    or item['receipt']['bytes'] > 20_000_000
                    or checksum(item['path']) != item['receipt']['sha256']):
                raise ValueError('retained original HTML receipt mismatch')
            protected[item['path']] = item['receipt']['sha256']
        day = row['filed']
        archive = archives[day[:4] + 'q' + str((int(day[5:7]) - 1) // 3 + 1)]
        resolution = primary_html_joint(case['index']['path'], case['index']['receipt']['sha256'],
            cik=row['filing_directory_cik'], accession=row['accession'], form=row['form'],
            master_archive=archive['path'], master_sha256=archive['sha256'])
        if resolution['url'] != case['primary']['receipt']['source_url']:
            raise ValueError('primary endpoint differs from original index/master authority')
        text = normalized_text(case['index']['path'])
        filed = re.findall(r'Filing Date (\d{4}-\d{2}-\d{2})', text)
        accepted = re.findall(r'Accepted (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', text)
        if len(filed) != 1 or len(accepted) != 1 or filed[0] != row['filed']:
            raise ValueError('original/master publication metadata mismatch')
        publication = {'filed': filed[0], 'accepted_source': accepted[0],
                       'availability': conservative_availability(filed[0], accepted[0])}
        if publication != case['publication']:
            raise ValueError('recorded publication differs from original index')
        if publication['availability']['available_time'][:10] > expected[symbol]['first_training_session']:
            raise ValueError('source publication too late for initial training observation')
        hints = inspect_html_stock(case['primary']['path'], case['primary']['receipt']['sha256'], symbol=symbol)
        records.append({'symbol': symbol, 'accession': row['accession'],
                        'source_url': resolution['url'], 'publication': case['publication'],
                        'resolution': resolution, 'hints': hints})
        if len(records) % 16 == 0:
            print('Verified source documents', len(records), 'of69', flush=True)
    if not all(checksum(path) == sha for path, sha in protected.items()):
        raise ValueError('protected sources changed')
    counts = Counter({'documents': len(records)})
    for row in records:
        counts['common_equity_cover_hints'] += row['hints']['common_equity_cover_hint']
        counts['explicit_ticker_clause_hints'] += row['hints']['explicit_ticker_clause_hint']
        counts['cross_directory_joint_filings'] += row['resolution']['cross_directory']
    result = {'schema': 'task-stock-combined-source-audit-v1', 'records': records,
              'counts': dict(counts), 'collection_scope_complete': True,
              'old_stop_preserved': True, 'original_failed_batch_status': 'stopped',
              'protected_sha256': protected, 'protected_sources_unchanged': True,
              'historical_metadata_admitted': False, 'network_requests': 0,
              'training_performed': False, 'colab_allocated': False}
    write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print(dict(counts))


if __name__ == '__main__':
    main()
