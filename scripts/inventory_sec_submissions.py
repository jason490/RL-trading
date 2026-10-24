"""Checksum-verified chronological SUB inventory; no financial amounts or inference."""
import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

from rl_stocks.core.files import write_once

FORMS = {'10-K', '10-Q', '10-K/A', '10-Q/A'}
FIELDS = ('adsh', 'cik', 'form', 'period', 'filed', 'accepted')
QUARTERS = [f'{year}q{quarter}' for year in range(2009, 2024)
            for quarter in range(1, 5) if (year, quarter) >= (2009, 2)]


def inventory(state):
    if state.get('schema') != 'sec-quarter-collection-v1' or state.get('status') != 'downloaded_only':
        raise ValueError('terminal complete collection journal required')
    protocol = Path('configs/experiments/filing_coverage_audit.json').read_bytes()
    if state.get('protocol_sha256') != hashlib.sha256(protocol).hexdigest():
        raise ValueError('collection protocol identity mismatch')
    quarters = state['planned_quarters']
    if quarters != QUARTERS or [r['quarter'] for r in state['completed']] != quarters:
        raise ValueError('complete unique ordered source range required')
    global_counts, seen, by_cik, archive_counts = Counter(), {}, {}, {}
    conflicts = []
    for item in state['completed']:
        path = Path(item['path'])
        with path.open('rb') as source:
            sha = hashlib.file_digest(source, 'sha256').hexdigest()
        if sha != item['archive_sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError('collected source checksum/size mismatch')
        counts = Counter()
        with ZipFile(path) as z:
            if z.namelist().count('sub.txt') != 1:
                raise ValueError('exactly one SUB table required')
            if z.getinfo('sub.txt').file_size > 30_000_000:
                raise ValueError('SUB table byte limit exceeded')
            with z.open('sub.txt') as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig'), delimiter='\t')
                fields = reader.fieldnames or []
                if len(fields) != len(set(fields)) or not set(FIELDS) <= set(fields):
                    raise ValueError('SUB schema mismatch')
                local = set()
                for r in reader:
                    counts['source_rows'] += 1
                    if None in r or any(r[k] is None for k in FIELDS):
                        raise ValueError('malformed SUB row')
                    if not re.fullmatch(r'\d{10}-\d{2}-\d{6}', r['adsh']):
                        raise ValueError('invalid SUB accession')
                    if r['adsh'] in local:
                        raise ValueError('duplicate SUB accession within archive')
                    local.add(r['adsh'])
                    if not re.fullmatch(r'[1-9][0-9]{0,9}', r['cik']):
                        raise ValueError('invalid SUB CIK')
                    for key in ('period', 'filed'):
                        if key == 'period' and not r[key] and r['form'] not in FORMS:
                            counts['nonperiodic_forms_without_period'] += 1
                            continue
                        if not re.fullmatch(r'[0-9]{8}', r[key]):
                            raise ValueError('invalid SUB date format')
                        datetime.strptime(r[key], '%Y%m%d')
                    if not re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?', r['accepted']):
                        raise ValueError('invalid SUB acceptance format')
                    accepted = datetime.fromisoformat(r['accepted'])
                    if r['filed'] >= '20240101' or r['period'] >= '20240101' or accepted.year >= 2024:
                        counts['retired_SUB_rows_excluded'] += 1
                        continue
                    digest = hashlib.sha256(json.dumps({k: r[k] for k in FIELDS}, sort_keys=True).encode()).hexdigest()
                    prior = seen.get(r['adsh'])
                    if prior is not None:
                        counts['cross_archive_duplicate_accessions'] += 1
                        if prior != digest:
                            counts['cross_archive_conflicts'] += 1
                            if len(conflicts) < 30:
                                conflicts.append(dict(adsh=r['adsh'], quarter=item['quarter']))
                        continue
                    seen[r['adsh']] = digest
                    counts['unique_nonretired_submissions'] += 1
                    if r['form'] not in FORMS:
                        continue
                    counts['qualifying_form_submissions'] += 1
                    entry = by_cik.setdefault(r['cik'], dict(names=set(), filing_year_forms=Counter(),
                                                            period_year_forms=Counter()))
                    entry['names'].add(r.get('name', ''))
                    entry['filing_year_forms'][r['filed'][:4] + ':' + r['form']] += 1
                    entry['period_year_forms'][r['period'][:4] + ':' + r['form']] += 1
        archive_counts[item['quarter']] = dict(counts)
        global_counts.update(counts)
    for entry in by_cik.values():
        entry['names'] = sorted(entry['names'])
        entry['filing_year_forms'] = dict(sorted(entry['filing_year_forms'].items()))
        entry['period_year_forms'] = dict(sorted(entry['period_year_forms'].items()))
    return dict(schema='sec-chronological-sub-inventory-v1', scope='SUB metadata only; no NUM inspection',
                source_quarters=quarters, archives=archive_counts, counts=dict(global_counts),
                cik_count=len(by_cik), by_cik=by_cik, conflict_examples=conflicts,
                all_source_hashes_verified=True, financial_amounts_read=False,
                historical_mapping_verified=False, publication_verified=False,
                causal_ready=False, training_performed=False, financial_features_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.state.read_bytes()
    result = inventory(json.loads(raw))
    result['collection_journal_sha256'] = hashlib.sha256(raw).hexdigest()
    write_once(args.output, (json.dumps(result, indent=2, sort_keys=True) + '\n').encode())
    print('SUB inventory:', result['counts'], 'issuers:', result['cik_count'])
    print('Metadata only; historical identity/publication and financial readiness remain unverified.')


if __name__ == '__main__':
    main()
