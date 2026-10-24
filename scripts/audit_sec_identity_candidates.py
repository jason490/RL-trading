"""Current-CIK discovery versus archived submissions; never historical verification."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from rl_stocks.core.files import write_once

FORMS = {'10-K', '10-Q', '10-K/A', '10-Q/A'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--discovery', type=Path, required=True)
    parser.add_argument('--structural', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    discovery_raw, structural_raw = args.discovery.read_bytes(), args.structural.read_bytes()
    discovery, structural = json.loads(discovery_raw), json.loads(structural_raw)
    if (discovery.get('schema') != 'sec-current-identity-discovery-v1'
            or discovery.get('discovery_only') is not True
            or discovery.get('historical_mapping_verified') is not False):
        raise ValueError('explicit unverified current-identity discovery required')
    if (structural.get('schema') != 'sec-archive-structural-audit-v3'
            or structural.get('causal_ready') is not False):
        raise ValueError('unverified source structural report required')
    by_cik = {}
    for record in structural['submissions']:
        if record['form'] in FORMS:
            by_cik.setdefault(record['cik'], []).append(record)
    per_symbol = {}
    reverse = {}
    for symbol, candidates in discovery['candidates'].items():
        for candidate in candidates:
            reverse.setdefault(candidate['cik'], []).append(symbol)
    for symbol, candidates in sorted(discovery['candidates'].items()):
        records = [r for candidate in candidates for r in by_cik.get(candidate['cik'], [])]
        per_symbol[symbol] = dict(
            candidate_ciks=[c['cik'] for c in candidates],
            matching_filing_count=len(records),
            forms=dict(Counter(r['form'] for r in records)),
            reporting_years=dict(Counter(r['period'][:4] for r in records)),
            accessions=sorted({r['adsh'] for r in records}),
            ambiguous_current_candidates=len(candidates) != 1,
            shared_cik=any(len(set(reverse[c['cik']])) > 1 for c in candidates),
            historical_identity_verified=False)
    result = dict(
        schema='sec-candidate-archive-coverage-v1', quarter=structural['quarter'],
        discovery_sha256=hashlib.sha256(discovery_raw).hexdigest(),
        structural_sha256=hashlib.sha256(structural_raw).hexdigest(),
        current_identity_source_sha256=discovery['source_sha256'],
        archive_sha256=structural['archive_sha256'], symbol_count=len(per_symbol),
        candidate_present_count=sum(r['matching_filing_count'] > 0 for r in per_symbol.values()),
        candidate_absent=sorted(s for s, r in per_symbol.items() if not r['matching_filing_count']),
        per_symbol=per_symbol, historical_mapping_verified=False,
        model_inputs_enabled=False, training_performed=False,
        interpretation='Current CIK matches are discovery hints only. Counts do not prove historical ticker/security continuity or complete quarterly coverage.')
    write_once(args.output, (json.dumps(result, indent=2, sort_keys=True) + '\n').encode())
    print('Candidate matches in archive:', result['candidate_present_count'], '/', len(per_symbol))
    print('Current candidates absent:', result['candidate_absent'])
    print('Historical mapping remains unverified; no model inputs created.')


if __name__ == '__main__':
    main()
