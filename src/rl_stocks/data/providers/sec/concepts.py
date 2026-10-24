"""Financial-table tag presence for source triage; never financial arithmetic."""
import csv
import hashlib
import io
import time
from pathlib import Path
from zipfile import ZipFile

from rl_stocks.features.fundamentals.snapshot import REVENUE

ALTERNATIVES = {'OperatingIncomeLoss', 'ProfitLoss', 'SalesRevenueGoodsNet', 'SalesRevenueServicesNet',
                'RevenuesNetOfInterestExpense', 'NetIncomeLossAvailableToCommonStockholdersBasic'}
TRACKED = REVENUE | ALTERNATIVES | {'NetIncomeLoss', 'Assets', 'Liabilities'}


def scan_archive(item, targets):
    started = time.monotonic()
    path = Path(item['path'])
    with path.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != item['archive_sha256']:
            raise ValueError('concept audit source checksum mismatch')
    if path.stat().st_size != item['bytes']:
        raise ValueError('concept audit source size mismatch')
    target_map = {r['accession']: r for r in targets}
    if len(target_map) != len(targets):
        raise ValueError('unique accession targets required')
    accumulated = {key: dict(instants=set(), flows={}) for key in target_map}
    source_rows = 0
    with ZipFile(path) as z:
        if z.namelist().count('num.txt') != 1 or z.getinfo('num.txt').file_size > 2_000_000_000:
            raise ValueError('bounded unique NUM table required')
        with z.open('num.txt') as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig'), delimiter='\t')
            fields = reader.fieldnames or []
            required = {'adsh', 'tag', 'version', 'ddate', 'qtrs', 'uom', 'coreg', 'segments', 'value'}
            if len(fields) != len(set(fields)) or not required <= set(fields):
                raise ValueError('explicit segment-aware NUM schema required')
            for row in reader:
                source_rows += 1
                if source_rows % 10000 == 0 and time.monotonic() - started > 300:
                    raise TimeoutError('bounded concept audit archive deadline exceeded')
                accession = row['adsh']
                if accession not in target_map or row['tag'] not in TRACKED:
                    continue
                if (row['ddate'] != target_map[accession]['archive_period_label']
                        or not row['version'].startswith('us-gaap/') or row['uom'] != 'USD'
                        or row['coreg'] or row['segments'] or not row['value'].strip()):
                    continue
                qtrs = row['qtrs']
                if not qtrs.isascii() or not qtrs.isdigit():
                    raise ValueError('valid source quarter count required')
                qtrs = int(qtrs)
                state = accumulated[accession]
                if qtrs == 0 and row['tag'] in {'Assets', 'Liabilities'}:
                    state['instants'].add(row['tag'])
                elif qtrs in {1, 2, 3, 4}:
                    state['flows'].setdefault(qtrs, set()).add(row['tag'])
    records = []
    for accession, target in sorted(target_map.items()):
        state = accumulated[accession]
        allowed = {4} if target['form'] in {'10-K', '10-K/A'} else {1, 2, 3}
        flows = {q: tags for q, tags in state['flows'].items() if q in allowed}
        common = [q for q, tags in flows.items() if tags & REVENUE and 'NetIncomeLoss' in tags]
        tags = set().union(*flows.values()) if flows else set()
        records.append(dict(accession=accession, cik=target['cik'], form=target['form'],
                            filing_year=target['archive_filed'][:4],
                            current_total_revenue_present=bool(tags & REVENUE),
                            current_net_income_present='NetIncomeLoss' in tags,
                            current_assets_present='Assets' in state['instants'],
                            current_liabilities_present='Liabilities' in state['instants'],
                            same_flow_qtrs=sorted(common),
                            structural_presence_pass=bool(common) and state['instants'] == {'Assets', 'Liabilities'},
                            unsupported_current_labels=sorted(tags & ALTERNATIVES)))
    return dict(quarter=item['quarter'], archive_sha256=item['archive_sha256'], source_num_rows=source_rows,
                elapsed_seconds=time.monotonic() - started, records=records,
                financial_ratios_computed=False, financial_features_enabled=False)
