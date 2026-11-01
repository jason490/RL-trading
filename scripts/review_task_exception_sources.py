"""Check the fixed spinoff/predecessor equity and ticker observations."""
import argparse
import json
from pathlib import Path
import re

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.lineage import conservative_availability
from rl_stocks.inference.bundle import checksum
from review_task_type_covers import normalized_text

ANCHORS = {
    'ABBV': ['outstanding stock of AbbVie',
             'independent public company trading under the symbol "ABBV" on the New York Stock Exchange.'],
    'DIS': ['common stock is listed on the New York Stock Exchange under the ticker symbol “DIS”.'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh exception review output required')
    source = Path('results/task-conditioning-2026-10-31/exception-source-collection.json')
    collected = json.loads(source.read_text())
    if collected['status'] != 'complete' or {r['symbol'] for r in collected['cases']} != set(ANCHORS):
        raise ValueError('complete fixed exception collection required')
    protected = {str(p): checksum(p) for p in
                 (source, Path(__file__), Path('scripts/review_task_type_covers.py'),
                  Path('src/rl_stocks/data/providers/sec/lineage.py'), Path('.runtime/usage-ledger.json'))}
    records = []
    for case in collected['cases']:
        for name in ('index', 'primary'):
            item = case[name]
            if checksum(item['path']) != item['receipt']['sha256']:
                raise ValueError('collected exception source changed')
            protected[item['path']] = item['receipt']['sha256']
        text = normalized_text(case['primary']['path'])
        if any(anchor not in text for anchor in ANCHORS[case['symbol']]):
            raise ValueError('reviewed equity/ticker statement absent')
        index = normalized_text(case['index']['path'])
        filed = re.findall(r'Filing Date (\d{4}-\d{2}-\d{2})', index)
        accepted = re.findall(r'Accepted (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', index)
        if len(filed) != 1 or len(accepted) != 1 or filed[0] != case['filed']:
            raise ValueError('original/master publication mismatch')
        availability = conservative_availability(filed[0], accepted[0])
        if availability['available_time'][:10] > case['first_training_session']:
            raise ValueError('observation unavailable at initial training session')
        records.append({'symbol': case['symbol'], 'source_registrant_cik': case['candidate_cik'],
                        'accession': case['accession'], 'source_url': case['primary']['receipt']['source_url'],
                        'source_sha256': case['primary']['receipt']['sha256'],
                        'anchors': ANCHORS[case['symbol']], 'publication': availability,
                        'dated_corporate_equity_observation_supported': True,
                        'explicit_canonical_ticker_supported': True,
                        'historical_price_link_verified': False,
                        'historical_metadata_admitted': False, 'validity_interval_inferred': False})
    if not all(checksum(path) == sha for path, sha in protected.items()):
        raise ValueError('protected source changed')
    result = {'schema': 'task-exception-source-review-v1', 'records': records,
              'protected_sha256': protected, 'protected_sources_unchanged': True,
              'historical_metadata_admitted': False, 'network_requests': 0,
              'training_performed': False, 'colab_allocated': False}
    write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print('Verified two dated equity/ticker observations; full metadata remains unadmitted')


if __name__ == '__main__':
    main()
