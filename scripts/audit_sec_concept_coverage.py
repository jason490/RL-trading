"""Bounded parallel scan of cached NUM metadata, without fitting or performance scoring."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.concepts import scan_archive


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.state.exists():
        raise ValueError('new output/journal required; no automatic restart')
    collection_raw, ledger_raw = args.collection.read_bytes(), args.ledger.read_bytes()
    collection, ledger = json.loads(collection_raw), json.loads(ledger_raw)
    protocol_raw = Path('configs/experiments/filing_concept_audit.json').read_bytes()
    coverage_raw = Path('configs/experiments/filing_coverage_audit.json').read_bytes()
    quarters = [f'{y}q{q}' for y in range(2009, 2024) for q in range(1, 5) if (y, q) >= (2009, 2)]
    if (collection['status'] != 'downloaded_only' or collection['planned_quarters'] != quarters
            or [r['quarter'] for r in collection['completed']] != quarters
            or collection['protocol_sha256'] != hashlib.sha256(coverage_raw).hexdigest()
            or ledger['schema'] != 'sec-original-candidate-ledger-v1'):
        raise ValueError('fixed complete financial source collection and candidate ledger required')
    args.output.mkdir(parents=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = dict(status='running', started_epoch=time.time(), completed=[], current_workers=2,
                 protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(),
                 collection_sha256=hashlib.sha256(collection_raw).hexdigest(),
                 ledger_sha256=hashlib.sha256(ledger_raw).hexdigest(),
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 scanner_sha256=hashlib.sha256(Path('src/rl_stocks/data/providers/sec/concepts.py').read_bytes()).hexdigest(),
                 training_performed=False, colab_allocated=False)
    save(args.state, state)
    pool = ProcessPoolExecutor(max_workers=2)
    futures = {}
    try:
        for item in collection['completed']:
            targets = [r for r in ledger['records'] if r['archive_quarter'] == item['quarter']]
            if any(r['archive_sha256'] != item['archive_sha256'] for r in targets):
                raise ValueError('target source lineage mismatch')
            futures[pool.submit(scan_archive, item, targets)] = item['quarter']
        for future in as_completed(futures):
            report = future.result()
            quarter = futures[future]
            write_once(args.output / (quarter + '.json'), (json.dumps(report, indent=2) + '\n').encode())
            state['completed'].append(quarter)
            save(args.state, state)
            print(f"Inspected {quarter}: {len(state['completed'])}/59, {len(report['records'])} candidates", flush=True)
        state['status'] = 'complete'
    except BaseException as error:
        state['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        state['error_type'] = type(error).__name__
        raise
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        state['workers_stopped'] = True
        state['updated_epoch'] = time.time()
        save(args.state, state)


if __name__ == '__main__':
    main()
