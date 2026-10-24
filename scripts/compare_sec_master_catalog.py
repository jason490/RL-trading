"""Compare complete quarterly master metadata with financial-table candidates."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.master import read_master


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--discovery', type=Path, required=True)
    parser.add_argument('--financial-ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    state_raw, discovery_raw, financial_raw = [p.read_bytes() for p in
                                               (args.state, args.discovery, args.financial_ledger)]
    state, discovery, financial = map(json.loads, (state_raw, discovery_raw, financial_raw))
    protocol = Path('configs/experiments/filing_master_audit.json').read_bytes()
    scope_raw = Path('configs/filings/cohort_discovery_v1.json').read_bytes()
    scope = json.loads(scope_raw)
    quarters = [f'{y}q{q}' for y in range(2009, 2024) for q in range(1, 5) if (y, q) >= (2009, 2)]
    if (state.get('source_kind') != 'master' or state.get('status') != 'downloaded_only'
            or state.get('protocol_sha256') != hashlib.sha256(protocol).hexdigest()
            or state['planned_quarters'] != quarters
            or [r['quarter'] for r in state['completed']] != quarters):
        raise ValueError('complete fixed master collection required')
    if (discovery.get('discovery_only') is not True
            or discovery['source_sha256'] != scope['current_candidate_source_sha256']
            or financial['specification_sha256'] != hashlib.sha256(scope_raw).hexdigest()):
        raise ValueError('fixed candidate sources required')
    ciks = {c['cik'] for values in discovery['candidates'].values() for c in values}
    ciks.update(r['cik'] for r in scope['additional_candidates'])
    catalog, archive_counts, conflicts = {}, {}, []
    source_rows = 0
    for item in state['completed']:
        if Path(item['path']).stat().st_size != item['bytes']:
            raise ValueError('master journal source size mismatch')
        report = read_master(item['path'], item['archive_sha256'], ciks)
        source_rows += report['source_rows']
        archive_counts[item['quarter']] = len(report['records'])
        for record in report['records']:
            key = (record['cik'], record['accession'])
            if key in catalog:
                previous = {k: v for k, v in catalog[key].items() if not k.startswith('master_')}
                if previous != record:
                    conflicts.append(dict(cik=key[0], accession=key[1], quarter=item['quarter']))
                continue
            catalog[key] = dict(record, master_quarter=item['quarter'], master_sha256=item['archive_sha256'])
    financial_map = {(r['cik'], r['accession']): r for r in financial['records']}
    if len(financial_map) != len(financial['records']):
        raise ValueError('unique financial issuer/accession pairs required')
    master_keys, financial_keys = set(catalog), set(financial_map)
    matches = master_keys & financial_keys
    differences = []
    for key in sorted(matches):
        master, source = catalog[key], financial_map[key]
        label = source['archive_filed']
        source_date = label[:4] + '-' + label[4:6] + '-' + label[6:8]
        if master['form'] != source['form'] or master['filed'] != source_date:
            differences.append(dict(cik=key[0], accession=key[1], master_form=master['form'],
                                    financial_form=source['form'], master_filed=master['filed'],
                                    financial_filed=source_date))
    multiplicity = Counter(r['accession'] for r in catalog.values())
    summary = dict(schema='sec-master-financial-catalog-comparison-v1',
        source_quarters=len(quarters), source_master_rows=source_rows,
        candidate_cik_count=len(ciks), master_issuer_accession_pairs=len(master_keys),
        master_unique_accessions=len(multiplicity), financial_pairs=len(financial_keys),
        matched_pairs=len(matches), master_only_pairs=len(master_keys - financial_keys),
        financial_only_pairs=len(financial_keys - master_keys),
        joint_accessions=sum(n > 1 for n in multiplicity.values()),
        metadata_differences=differences, duplicate_master_conflicts=conflicts,
        master_only=[catalog[key] for key in sorted(master_keys - financial_keys)],
        financial_only=[financial_map[key] for key in sorted(financial_keys - master_keys)],
        master_quarter_candidate_counts=archive_counts,
        collection_compressed_bytes=state['acquired_bytes'],
        collection_elapsed_seconds=state['updated_utc_epoch'] - state['started_utc_epoch'],
        collection_state_sha256=hashlib.sha256(state_raw).hexdigest(),
        financial_ledger_sha256=hashlib.sha256(financial_raw).hexdigest(),
        discovery_sha256=hashlib.sha256(discovery_raw).hexdigest(),
        original_statements_collected=False, historical_vintages_verified=False,
        historical_mapping_verified=False, financial_features_enabled=False,
        training_performed=False,
        interpretation='Complete registered-quarter index comparison for candidate CIKs only. Joint registrants remain separate; current master indexes may reflect post-acceptance corrections. No stock/issuer or financial-fact approval.')
    output = dict(summary, records=[catalog[key] for key in sorted(catalog)])
    write_once(args.output, (json.dumps(output, indent=2) + '\n').encode())
    write_once(args.summary, (json.dumps(summary, indent=2) + '\n').encode())
    print({k: summary[k] for k in ['master_issuer_accession_pairs', 'master_unique_accessions',
          'financial_pairs', 'matched_pairs', 'master_only_pairs', 'financial_only_pairs', 'joint_accessions']})
    print('Catalog comparison only; financial/price readiness remains unverified.')


if __name__ == '__main__':
    main()
