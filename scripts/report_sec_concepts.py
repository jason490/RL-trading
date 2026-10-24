"""Summarize source-table tag presence, without promoting financial feature readiness."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import plotly.graph_objects as go

from rl_stocks.core.files import write_once

FIELDS = ('current_total_revenue_present', 'current_net_income_present',
          'current_assets_present', 'current_liabilities_present', 'structural_presence_pass')
LABELS = ('Supported revenue', 'NetIncomeLoss', 'Assets', 'Liabilities', 'All four / same flow group')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--master-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--html', type=Path, required=True)
    args = parser.parse_args()
    state_raw = args.state.read_bytes()
    state = json.loads(state_raw)
    ledger_raw = args.ledger.read_bytes()
    ledger = json.loads(ledger_raw)
    master = json.loads(args.master_summary.read_text())
    quarters = [f'{y}q{q}' for y in range(2009, 2024) for q in range(1, 5) if (y, q) >= (2009, 2)]
    if (state['status'] != 'complete' or state.get('workers_stopped') is not True
            or sorted(state['completed']) != sorted(quarters)
            or state['ledger_sha256'] != hashlib.sha256(ledger_raw).hexdigest()
            or master['financial_ledger_sha256'] != state['ledger_sha256']):
        raise ValueError('complete stopped audit with matched candidate/index sources required')
    target = {(r['cik'], r['accession']): r for r in ledger['records']}
    rows, sources, total_num = [], {}, 0
    for quarter in quarters:
        path = args.reports / (quarter + '.json')
        raw = path.read_bytes()
        report = json.loads(raw)
        if report['quarter'] != quarter or report['financial_features_enabled'] is not False:
            raise ValueError('source-only quarterly report required')
        if any(target[(r['cik'], r['accession'])]['archive_sha256'] != report['archive_sha256']
               for r in report['records']):
            raise ValueError('quarter source identity mismatch')
        rows.extend(report['records'])
        sources[quarter] = hashlib.sha256(raw).hexdigest()
        total_num += report['source_num_rows']
    if len(rows) != len(target) or {(r['cik'], r['accession']) for r in rows} != set(target):
        raise ValueError('exact candidate coverage required; no missing/duplicate records')
    by_year = defaultdict(Counter)
    counts = Counter()
    for r in rows:
        bucket = by_year[r['filing_year']]
        for aggregate in (counts, bucket):
            aggregate['candidate_pairs'] += 1
            for field in FIELDS:
                aggregate[field] += int(r[field])
            for name in r['unsupported_current_labels']:
                aggregate['unsupported_' + name] += 1
    rows.sort(key=lambda r: (r['filing_year'], r['cik'], r['accession']))
    summary = dict(schema='sec-concept-presence-summary-v1', assessed_financial_pairs=len(rows),
        unassessed_master_pairs=master['master_only_pairs'], master_issuer_accession_pairs=master['master_issuer_accession_pairs'],
        source_num_rows=total_num, counts=dict(counts),
        filing_year_counts={y: dict(c) for y, c in sorted(by_year.items())},
        quarterly_report_sha256=sources, audit_state_sha256=hashlib.sha256(state_raw).hexdigest(),
        protocol_sha256=state['protocol_sha256'], scanner_sha256=state['scanner_sha256'],
        audit_script_sha256=state['script_sha256'],
        elapsed_seconds=state['updated_epoch'] - state['started_epoch'], workers_stopped=True,
        amounts_retained=False, financial_ratios_computed=False, arithmetic_readiness_assessed=False,
        historical_mapping_verified=False, financial_features_enabled=False,
        normalization_fitted=False, training_performed=False, colab_allocated=False,
        interpretation='Financial-table current-period tag presence only. Original filings may add omitted facts; exact contexts, values, comparables, identity and availability remain unverified. No price coverage or model-return inference.')
    output = dict(summary, records=rows)
    write_once(args.output, (json.dumps(output, indent=2, sort_keys=True) + '\n').encode())
    write_once(args.summary, (json.dumps(summary, indent=2, sort_keys=True) + '\n').encode())
    years = sorted(by_year)
    percentages = [[100 * by_year[y][f] / by_year[y]['candidate_pairs'] for y in years] for f in FIELDS]
    fig = go.Figure(go.Heatmap(x=years, y=LABELS, z=percentages, zmin=0, zmax=100,
                              colorscale='Viridis', colorbar=dict(title='Presence %'),
                              hovertemplate='%{y}, %{x}: %{z:.1f}%<extra></extra>'))
    fig.update_layout(title='Candidate financial-table tag presence — NOT feature readiness',
                      xaxis_title='Filing year (2009 starts Q2)', template='plotly_white', height=500)
    write_once(args.html, fig.to_html(full_html=True, include_plotlyjs=True,
                                    div_id='sec-concept-presence').encode())
    print('Concept presence:', dict(counts))
    print('Unassessed master pairs:', master['master_only_pairs'], 'No arithmetic/price/model readiness claim.')


if __name__ == '__main__':
    main()
