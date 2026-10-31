"""Verify fixed historical fund statements without admitting a full history mapping."""
import argparse
import json
from pathlib import Path
import re

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.lineage import conservative_availability
from rl_stocks.inference.bundle import checksum
from review_task_type_covers import normalized_text

ANCHORS = {
    'SPY': ['SPDR S&P 500 ETF TRUST', 'Units trade on NYSE Arca, Inc. under the symbol “SPY”.'],
    'QQQ': ['is a unit investment trust', 'under the symbol "QQQQ."'],
    'DIA': ['SPDR DOW JONES INDUSTRIAL AVERAGE ETF TRUST',
            'Units trade on NYSE Arca, Inc. under the symbol “DIA”.'],
    'IWM': ['Russell 2000 Index Fund (“IWM”)', 'Index Fund is an exchange-traded fund'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh fund review output required')
    source = Path('results/task-conditioning-2026-10-31/fund-source-collection.json')
    evidence_path = Path('results/task-conditioning-2026-10-31/type-evidence.json')
    collected = json.loads(source.read_text())
    first_sessions = {r['symbol']: r['first_training_session']
                      for r in json.loads(evidence_path.read_text())['records']}
    if collected['status'] != 'complete' or {c['symbol'] for c in collected['cases']} != set(ANCHORS):
        raise ValueError('original complete four-source collection required')
    protected = {str(p): checksum(p) for p in
                 (source, evidence_path, Path(__file__), Path('scripts/review_task_type_covers.py'),
                  Path('.runtime/usage-ledger.json'))}
    records = []
    for case in collected['cases']:
        for name in ('index', 'primary'):
            item = case[name]
            if checksum(item['path']) != item['receipt']['sha256']:
                raise ValueError('collected fund source changed')
            protected[item['path']] = item['receipt']['sha256']
        text = normalized_text(case['primary']['path'])
        if any(anchor not in text for anchor in ANCHORS[case['symbol']]):
            raise ValueError('reviewed fund source anchor absent; no substitution')
        index = normalized_text(case['index']['path'])
        filed = re.findall(r'Filing Date (\d{4}-\d{2}-\d{2})', index)
        accepted = re.findall(r'Accepted (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', index)
        if len(filed) != 1 or len(accepted) != 1 or filed[0] != case['master_record']['filed']:
            raise ValueError('original/master publication metadata mismatch')
        availability = conservative_availability(filed[0], accepted[0])
        if availability['available_time'][:10] > first_sessions[case['symbol']]:
            raise ValueError('fund observation unavailable by initial training session')
        records.append({'symbol': case['symbol'], 'filed': filed[0],
                        'accepted_source': accepted[0], 'availability': availability,
                        'source_url': case['primary']['receipt']['source_url'],
                        'source_sha256': case['primary']['receipt']['sha256'],
                        'source_registrant_cik': case['master_record']['cik'],
                        'source_is_fund_registrant': case['source_is_fund_registrant'],
                        'historical_symbol_in_source': 'QQQQ' if case['symbol'] == 'QQQ' else case['symbol'],
                        'anchors': ANCHORS[case['symbol']],
                        'dated_fund_type_observation_supported': True,
                        'historical_price_link_verified': False,
                        'historical_metadata_admitted': False, 'validity_interval_inferred': False})
    if not all(checksum(Path(p)) == value for p, value in protected.items()):
        raise ValueError('protected source changed')
    result = {'schema': 'task-fund-source-review-v1', 'records': records,
              'protected_sha256': protected, 'protected_sources_unchanged': True,
              'historical_metadata_admitted': False, 'network_requests': 0,
              'colab_allocated': False, 'training_performed': False}
    write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print('Four dated fund-type observations; no automatic history mapping or metadata admission')


if __name__ == '__main__':
    main()
