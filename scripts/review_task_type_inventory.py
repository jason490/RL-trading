"""Full initial coarse-type source ledger; never automatically enables model inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.lineage import conservative_availability
from rl_stocks.inference.bundle import checksum
from review_task_type_covers import normalized_text

ROOT = Path('results/task-conditioning-2026-10-31')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new source sufficiency ledger required')
    names = ('type-evidence.json', 'cached-stock-evidence.json', 'stock-source-combined-audit.json',
             'original-cover-review.json', 'original-cover-collection.json',
             'exception-source-review.json', 'fund-source-review.json', 'stock-master-selection.json')
    inputs = {name: json.loads((ROOT / name).read_text()) for name in names}
    protected = {str(p): checksum(p) for p in
                 [*(ROOT / name for name in names), Path(__file__),
                  Path('scripts/review_task_type_covers.py'),
                  Path('src/rl_stocks/data/providers/sec/lineage.py'), Path('.runtime/usage-ledger.json'),
                  Path('artifacts/filings/current-identity-discovery.json')]}
    discovery = json.loads(Path('artifacts/filings/current-identity-discovery.json').read_text())
    expected = {r['symbol']: r for r in inputs['type-evidence.json']['records']}
    selected = {r['symbol']: r for r in inputs['stock-master-selection.json']['records']}
    records = {}

    def add(symbol, *, kind, sha, url, available, family, cik=None, literal=False, rule=None):
        if symbol in records or kind != expected[symbol]['kind']:
            raise ValueError('exact unique source kind required')
        if available['available_time'][:10] > expected[symbol]['first_training_session']:
            raise ValueError('initial source fact unavailable before training')
        current = {r['cik'] for r in discovery['candidates'].get(symbol, [])}
        records[symbol] = {'symbol': symbol, 'kind_supported_by_source': kind,
            'source_sha256': sha, 'source_url': url, 'source_family': family,
            'source_registrant_cik': cik, 'publication': available,
            'initial_type_fact_supported': True, 'source_rule': rule,
            'explicit_canonical_ticker_observation': literal,
            'same_cik_in_current_discovery': cik in current if cik is not None else False,
            'current_discovery_is_not_historical_evidence': True,
            'first_training_session': expected[symbol]['first_training_session'],
            'historical_price_link_verified': False, 'historical_metadata_admitted': False}

    for row in inputs['cached-stock-evidence.json']['records']:
        if row['status'] != 'exact_document_hint':
            continue
        document = next(d for d in row['documents'] if d['evidence']['document_exact_stock_identity_hint'])
        evidence = document['evidence']
        path = Path('data/raw/sec-xbrl') / evidence['source_sha256'] / 'instance.xml'
        if checksum(path) != evidence['source_sha256']:
            raise ValueError('original instance changed')
        protected[str(path)] = evidence['source_sha256']
        available = conservative_availability(evidence['filed'], document['accepted'])
        add(row['symbol'], kind='stock', sha=evidence['source_sha256'], url=document['source_url'],
            available=available, family='exact_original_instance', cik=evidence['expected_cik'],
            literal=True, rule='issuer-qualified ticker/form/common-share facts')

    for row in inputs['stock-source-combined-audit.json']['records']:
        sha = row['hints']['source_sha256']
        path = Path('data/raw/sec-filings') / sha / 'filing.htm'
        if checksum(path) != sha:
            raise ValueError('original primary changed')
        protected[str(path)] = sha
        text = normalized_text(path)
        match = re.search(r'\b(?:common\s+(?:stock|shares)|ordinary\s+shares|capital\s+stock)\b',
                          text[:14000], re.I)
        if match is None:
            raise ValueError('corporate-equity front matter needs explicit further review')
        add(row['symbol'], kind='stock', sha=sha, url=row['source_url'],
            available=row['publication']['availability'], family='selected_original_primary',
            cik=selected[row['symbol']]['master_record']['cik'],
            literal=row['hints']['explicit_ticker_clause_hint'],
            rule={'source_concept': 'corporate_equity', 'front_matter_offset': match.start()})

    cover_cases = {r['symbol']: r for r in inputs['original-cover-collection.json']['cases']}
    for row in inputs['original-cover-review.json']['records']:
        add(row['symbol'], kind='stock', sha=row['source_sha256'], url=row['source_url'],
            available=row['availability'], family='reviewed_original_cover',
            cik=cover_cases[row['symbol']]['cik'], literal=row['exact_ticker_clause_supported'],
            rule='fixed original registered common-equity observation')
    for row in inputs['exception-source-review.json']['records']:
        add(row['symbol'], kind='stock', sha=row['source_sha256'], url=row['source_url'],
            available=row['publication'], family='reviewed_exception_primary',
            cik=row['source_registrant_cik'], literal=row['explicit_canonical_ticker_supported'],
            rule='dated distributed/registered corporate equity and own ticker statement')
    for row in inputs['fund-source-review.json']['records']:
        add(row['symbol'], kind='etf', sha=row['source_sha256'], url=row['source_url'],
            available=row['availability'], family='reviewed_fund_observation',
            cik=row['source_registrant_cik'], literal=row['historical_symbol_in_source'] == row['symbol'],
            rule={'fund_issuer_source': row['source_is_fund_registrant'],
                  'symbol_in_source': row['historical_symbol_in_source']})
    if set(records) != set(expected) or len(records) != 111:
        raise ValueError('exact111 source facts required; no snapshot fallback')
    for row in records.values():
        if row['source_family'] not in ('exact_original_instance', 'selected_original_primary'):
            path = Path('data/raw/sec-filings') / row['source_sha256'] / 'filing.htm'
            protected[str(path)] = row['source_sha256']
    if not all(checksum(path) == sha for path, sha in protected.items()):
        raise ValueError('protected source changed')
    counts = {'instruments': len(records), 'stocks': sum(r['kind_supported_by_source'] == 'stock'
              for r in records.values()), 'etfs': sum(r['kind_supported_by_source'] == 'etf'
              for r in records.values()), 'literal_canonical_ticker_observations': sum(
              r['explicit_canonical_ticker_observation'] for r in records.values())}
    result = {'schema': 'task-initial-type-source-ledger-v1', 'counts': counts,
        'source_family_counts': dict(Counter(r['source_family'] for r in records.values())),
        'records': [records[s] for s in sorted(records)], 'protected_sha256': protected,
        'protected_sources_unchanged': True, 'initial_source_fact_coverage_complete': True,
        'scope': 'Initial coarse stock/ETF type facts; not historical issuer/share-class/price continuity',
        'association_review_required': True, 'historical_metadata_admitted': False,
        'source_fact_is_not_a_validity_interval': True, 'network_requests': 0,
        'model_inputs_enabled': False, 'training_performed': False, 'colab_allocated': False}
    write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print('Initial source facts:', counts, '; association review/admission remain separate')


if __name__ == '__main__':
    main()
