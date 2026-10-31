"""Explicit reviewed source resume; original stop and every prior receipt stay immutable."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.download import identity
from rl_stocks.data.providers.sec.filing_index import primary_html_joint
from rl_stocks.data.providers.sec.html_source import acquire_filing_html
from rl_stocks.data.providers.sec.lineage import conservative_availability
from rl_stocks.inference.bundle import checksum
from review_task_type_covers import normalized_text

ROOT = Path('results/task-conditioning-2026-10-31')


def prepare():
    selection_path = ROOT / 'stock-master-selection.json'
    selection = json.loads(selection_path.read_text())
    pool = [r for r in selection['records'] if r['collection_needed']]
    if len(pool) != 69:
        raise ValueError('original69-source plan required')
    stop_path = Path('.runtime/task-stock-master-collection-stop-v1.json')
    stop = json.loads(stop_path.read_text())
    if stop['batch'] != 4 or stop['error'] != 'filing endpoint differs from declared identity':
        raise ValueError('review covers only the diagnosed joint-filer stop')
    protected = {str(p): checksum(p) for p in
                 (selection_path, stop_path, Path('.runtime/usage-ledger.json'), Path(__file__),
                  ROOT / 'nee-joint-resolution.json',
                  Path('src/rl_stocks/data/providers/sec/filing_index.py'),
                  Path('src/rl_stocks/data/providers/sec/html_source.py'),
                  Path('src/rl_stocks/data/providers/sec/master.py'),
                  Path('src/rl_stocks/data/providers/sec/lineage.py'),
                  Path('scripts/review_task_type_covers.py'))}
    done = set()
    previous_index = {}
    for batch in range(5):
        source = ROOT / f'stock-master-batch-{batch}.json'
        claim = Path(f'.runtime/task-stock-master-batch-{batch}-claim-v1.json')
        protected[str(source)], protected[str(claim)] = checksum(source), checksum(claim)
        value = json.loads(source.read_text())
        if (value['selection_sha256'] != checksum(selection_path)
                or value['status'] != ('stopped' if batch == 4 else 'complete')):
            raise ValueError('original complete/stopped sequence required')
        for case in value['cases']:
            expected = next(r for r in pool if r['symbol'] == case['symbol'])
            if case['master_record'] != expected['master_record']:
                raise ValueError('original master source changed')
            for name in ('index', 'primary'):
                if name in case:
                    item = case[name]
                    if checksum(item['path']) != item['receipt']['sha256']:
                        raise ValueError('retained original source changed')
                    protected[item['path']] = item['receipt']['sha256']
            if case['status'] == 'retrieved':
                done.add(case['symbol'])
            elif 'index' in case:
                previous_index[case['symbol']] = case['index']
    if len(done) != 35 or set(previous_index) != {'NEE'}:
        raise ValueError('exact retained35-primary prefix and NEE index required')
    archives = {}
    for receipt_path in Path('data/raw/sec-masters').glob('*/acquisition-*.json'):
        if str(receipt_path) not in selection['protected_sha256']:
            continue
        if checksum(receipt_path) != selection['protected_sha256'][str(receipt_path)]:
            raise ValueError('original master receipt changed')
        receipt = json.loads(receipt_path.read_text())
        path = receipt_path.parent / 'archive.zip'
        if checksum(path) != receipt['archive_sha256']:
            raise ValueError('original master archive changed')
        protected[str(receipt_path)], protected[str(path)] = checksum(receipt_path), receipt['archive_sha256']
        archives[receipt['quarter']] = {'path': str(path), 'sha256': receipt['archive_sha256']}
    cases = [{**deepcopy(r), 'original_batch': i // 8, 'status': 'not_attempted'}
             for i, r in enumerate(pool) if r['symbol'] not in done]
    if len(cases) != 34 or cases[0]['symbol'] != 'NEE':
        raise ValueError('only original34 unfinished source cases may resume')
    for case in cases:
        day = case['master_record']['filed']
        quarter = day[:4] + 'q' + str((int(day[5:7]) - 1) // 3 + 1)
        case['master_archive'] = archives[quarter]
    index = previous_index['NEE']
    nee = cases[0]
    proof = primary_html_joint(index['path'], index['receipt']['sha256'],
        cik=nee['master_record']['filing_directory_cik'],
        accession=nee['master_record']['accession'], form=nee['master_record']['form'],
        master_archive=nee['master_archive']['path'], master_sha256=nee['master_archive']['sha256'])
    if proof != json.loads((ROOT / 'nee-joint-resolution.json').read_text())['resolution']:
        raise ValueError('actual reviewed joint-filer resolution changed')
    if not all(checksum(p) == sha for p, sha in protected.items()):
        raise ValueError('protected source changed during preflight')
    return {'schema': 'task-stock-reviewed-resume-plan-v2', 'cases': cases,
            'protected_sha256': protected, 'selection_sha256': checksum(selection_path),
            'retained_primary_count': 35, 'remaining_primary_count': 34,
            'maximum_acquisition_operations': 68, 'old_stop_preserved': True,
            'automatic_retry': False, 'historical_metadata_admitted': False,
            'training_performed': False, 'colab_allocated': False}


def progress(path, result):
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def execute(plan_path, output):
    plan = json.loads(plan_path.read_text())
    if plan != prepare():
        raise ValueError('reviewed concrete resume plan differs from current preflight')
    claim = Path('.runtime/task-stock-reviewed-resume-v2-claim.json')
    stop = Path('.runtime/task-stock-reviewed-resume-v2-stop.json')
    checkpoint = output.with_name(output.name + '.progress.json')
    if any(p.exists() for p in (output, checkpoint, claim, stop)):
        raise ValueError('reviewed resume already claimed/stopped; no automatic retry')
    result = {**deepcopy(plan), 'schema': 'task-stock-reviewed-resume-result-v2',
              'plan_sha256': checksum(plan_path), 'acquisition_operations_attempted': 0,
              'status': 'running', 'completed_new_primaries': 0}
    ua = identity()
    write_once(claim, (json.dumps({'plan_sha256': checksum(plan_path), 'old_stop_preserved': True,
                                  'automatic_retry': False}, indent=2) + '\n').encode())
    output.parent.mkdir(parents=True, exist_ok=True)
    failure = None
    try:
        for case in result['cases']:
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
                raise ValueError('source unavailable at first training session')
            case['publication'] = {'filed': filed[0], 'accepted_source': accepted[0],
                                   'availability': availability}
            resolution = primary_html_joint(path, receipt['sha256'],
                cik=row['filing_directory_cik'], accession=row['accession'], form=row['form'],
                master_archive=case['master_archive']['path'],
                master_sha256=case['master_archive']['sha256'])
            case['resolution'] = resolution
            case['status'] = 'primary_started'
            result['acquisition_operations_attempted'] += 1
            path, receipt, cached = acquire_filing_html(
                'data/raw/sec-filings', url=resolution['url'], cik=resolution['directory_cik'],
                accession=row['accession'], user_agent=ua)
            case.update(status='retrieved', primary={'path': str(path), 'receipt': receipt,
                                                    'cached': cached})
            result['completed_new_primaries'] += 1
            progress(checkpoint, result)
            print(case['symbol'], 'retrieved; original batch', case['original_batch'], flush=True)
        result['status'] = 'complete'
    except BaseException as error:
        failure = error
        result.update(status='stopped', error_type=type(error).__name__, error=str(error))
        write_once(stop, (json.dumps({'plan_sha256': checksum(plan_path),
                                      'error_type': type(error).__name__, 'error': str(error),
                                      'old_stop_preserved': True, 'automatic_retry': False}, indent=2) + '\n').encode())
    finally:
        result['protected_sources_unchanged'] = all(checksum(p) == sha
                                                   for p, sha in plan['protected_sha256'].items())
        progress(checkpoint, result)
        write_once(output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    if failure is not None:
        raise RuntimeError('reviewed source resume stopped; no automatic retry') from None
    if not result['protected_sources_unchanged']:
        raise ValueError('protected source changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.output is not None or args.plan.exists():
            raise ValueError('new plan path and no execution output required')
        result = prepare()
        write_once(args.plan, (json.dumps(result, indent=2) + '\n').encode())
        print('Prepared34 unfinished cases; retained35 primaries and original stop')
    else:
        if args.output is None:
            raise ValueError('explicit fresh execution output required')
        execute(args.plan, args.output)


if __name__ == '__main__':
    main()
