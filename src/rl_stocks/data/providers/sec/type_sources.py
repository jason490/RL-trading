"""Deterministic primary-source discovery; no identity admission or price joins."""
from collections import defaultdict

from rl_stocks.data.providers.sec.lineage import conservative_availability


def select_stock_sources(type_rows, candidate_ledger, master_rows, covered_symbols=()):
    candidates = defaultdict(set)
    for row in candidate_ledger:
        for symbol in row['discovery_symbols']:
            candidates[symbol].add(row['cik'])
    availability = {}
    results = []
    for item in sorted(type_rows, key=lambda r: r['symbol']):
        if item['kind'] != 'stock':
            continue
        eligible = []
        for row in master_rows:
            if (row['cik'] not in candidates[item['symbol']]
                    or row['form'] not in ('10-K', '10-Q')
                    or row['filed'] >= item['first_training_session']):
                continue
            # Selection hint only. Original index acceptance metadata must agree later.
            if row['filed'] not in availability:
                availability[row['filed']] = conservative_availability(
                    row['filed'], row['filed'] + ' 00:00:00')['available_time'][:10]
            if availability[row['filed']] <= item['first_training_session']:
                eligible.append(row)
        annual = [r for r in eligible if r['form'] == '10-K']
        pool = annual or eligible
        selected = max(pool, key=lambda r: (r['filed'], r['accession'], r['cik'])) if pool else None
        results.append({'symbol': item['symbol'],
                        'first_training_session': item['first_training_session'],
                        'candidate_ciks': sorted(candidates[item['symbol']]),
                        'eligible_ciks': sorted({r['cik'] for r in eligible}),
                        'master_record': selected,
                        'already_has_document_observation': item['symbol'] in covered_symbols,
                        'collection_needed': selected is not None and item['symbol'] not in covered_symbols,
                        'historical_identity_verified': False,
                        'historical_metadata_admitted': False})
    return results
