"""Build complete source-candidate accession ledger without price joins or fetching."""
import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from rl_stocks.core.files import write_once


def build(state, discovery, specification):
    if state.get('status') != 'downloaded_only' or state.get('schema') != 'sec-quarter-collection-v1':
        raise ValueError('terminal source collection required')
    expected = [f'{y}q{q}' for y in range(2009, 2024) for q in range(1, 5) if (y, q) >= (2009, 2)]
    if state['planned_quarters'] != expected or [r['quarter'] for r in state['completed']] != expected:
        raise ValueError('complete preregistered source range required')
    if (discovery.get('discovery_only') is not True
            or discovery.get('historical_mapping_verified') is not False
            or discovery['source_sha256'] != specification['current_candidate_source_sha256']):
        raise ValueError('fixed unverified current discovery source required')
    by_cik = {}
    for symbol, candidates in discovery['candidates'].items():
        for candidate in candidates:
            by_cik.setdefault(candidate['cik'], set()).add(symbol)
    for candidate in specification['additional_candidates']:
        if candidate['symbol'] not in discovery['candidates']:
            raise ValueError('additional candidate outside research stocks')
        by_cik.setdefault(candidate['cik'], set()).add(candidate['symbol'])
    if any(not re.fullmatch(r'[1-9][0-9]{0,9}', c) for c in by_cik):
        raise ValueError('valid candidate issuer IDs required')
    ledger, seen = [], set()
    for item in state['completed']:
        path = Path(item['path'])
        with path.open('rb') as source:
            sha = hashlib.file_digest(source, 'sha256').hexdigest()
        if sha != item['archive_sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError('source archive hash/size mismatch')
        with ZipFile(path) as z:
            if z.namelist().count('sub.txt') != 1 or z.getinfo('sub.txt').file_size > 30_000_000:
                raise ValueError('bounded unique SUB table required')
            with z.open('sub.txt') as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig'), delimiter='\t')
                for row in reader:
                    if row['cik'] not in by_cik or row['form'] not in specification['forms']:
                        continue
                    accession, instance = row['adsh'], row['instance']
                    if not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession) or accession in seen:
                        raise ValueError('unique valid candidate accession required')
                    if not re.fullmatch(r'[A-Za-z0-9_.-]+\.xml', instance) or '..' in instance:
                        raise ValueError('safe original instance filename required')
                    seen.add(accession)
                    ledger.append(dict(accession=accession, cik=row['cik'], form=row['form'],
                        discovery_symbols=sorted(by_cik[row['cik']]), archive_quarter=item['quarter'],
                        archive_sha256=sha, archive_period_label=row['period'],
                        archive_filed=row['filed'], archive_accepted=row['accepted'],
                        archive_fiscal_year=row.get('fy'), archive_fiscal_period=row.get('fp'),
                        instance_filename=instance,
                        source_url=f"https://www.sec.gov/Archives/edgar/data/{row['cik']}/{accession.replace('-', '')}/{instance}",
                        original_period_verified=False, historical_price_link_verified=False))
    ledger.sort(key=lambda r: (r['archive_filed'], r['accession']))
    return dict(schema='sec-original-candidate-ledger-v1', source_dataset=specification['source_dataset'],
                forms=specification['forms'], stock_count=len(discovery['candidates']),
                candidate_cik_count=len(by_cik), accession_count=len(ledger),
                filing_year_counts=dict(sorted(Counter(r['archive_filed'][:4] for r in ledger).items())),
                records=ledger, source_archives_verified=True, discovery_only=True,
                original_instances_collected=False, historical_mapping_verified=False,
                financial_features_enabled=False, training_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--discovery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    specification_path = Path('configs/filings/cohort_discovery_v1.json')
    specification_raw = specification_path.read_bytes()
    result = build(json.loads(args.state.read_text()), json.loads(args.discovery.read_text()),
                   json.loads(specification_raw))
    result['specification_sha256'] = hashlib.sha256(specification_raw).hexdigest()
    write_once(args.output, (json.dumps(result, indent=2) + '\n').encode())
    print('Original candidates:', result['accession_count'], 'CIKs:', result['candidate_cik_count'])
    print('Filing years:', result['filing_year_counts'])
    print('Discovery only; no historical price mapping or financial input approval.')


if __name__ == '__main__':
    main()
