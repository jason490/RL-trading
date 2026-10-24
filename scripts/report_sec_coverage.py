"""Visualize current-CIK candidate filing counts without certifying historical identity."""
import argparse
import hashlib
import json
from pathlib import Path

import plotly.graph_objects as go

from rl_stocks.core.files import write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--discovery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    raw = args.inventory.read_bytes()
    inventory = json.loads(raw)
    discovery_raw = args.discovery.read_bytes()
    discovery = json.loads(discovery_raw)
    if (inventory.get('schema') != 'sec-chronological-sub-inventory-v1'
            or inventory.get('all_source_hashes_verified') is not True
            or inventory.get('historical_mapping_verified') is not False
            or discovery.get('discovery_only') is not True):
        raise ValueError('verified source inventory and explicitly unverified identity discovery required')
    years = list(range(2009, 2024))
    symbols = sorted(discovery['candidates'])
    rows, ambiguous = [], []
    for symbol in symbols:
        candidates = discovery['candidates'][symbol]
        if len(candidates) != 1:
            ambiguous.append(symbol)
            rows.append([None] * len(years))
            continue
        item = inventory['by_cik'].get(candidates[0]['cik'], {})
        counts = item.get('filing_year_forms', {})
        rows.append([sum(value for key, value in counts.items() if key.startswith(str(year) + ':'))
                     for year in years])
    fig = go.Figure(go.Heatmap(x=years, y=symbols, z=rows, colorscale='Viridis',
                              colorbar=dict(title='Filing count'),
                              hovertemplate='%{y}, %{x}: %{z} accessions<extra></extra>'))
    fig.update_layout(title='Current-CIK candidate filing counts — historical identity UNVERIFIED',
                      height=max(800, 15 * len(symbols)), xaxis_title='Filing year (2009 starts Q2)',
                      yaxis_title='Current research ticker', template='plotly_white',
                      annotations=[dict(text='Counts are SUB metadata, not NUM completeness, listing history, or model readiness.',
                                        x=0, y=1.03, xref='paper', yref='paper', showarrow=False)])
    write_once(args.output, fig.to_html(include_plotlyjs=True, full_html=True,
                                      div_id='sec-candidate-coverage').encode())
    summary = dict(schema='sec-candidate-year-counts-v1', inventory_sha256=hashlib.sha256(raw).hexdigest(),
                   discovery_sha256=hashlib.sha256(discovery_raw).hexdigest(),
                   years=years, symbols=symbols, filing_counts=rows, ambiguous_current_candidates=ambiguous,
                   historical_mapping_verified=False, financial_features_enabled=False,
                   interpretation='Counts use current candidate CIK only. Zeros may reflect issuer changes, filing timing, listing history or archive gaps; no performance or readiness inference.')
    write_once(args.summary, (json.dumps(summary, indent=2) + '\n').encode())
    print('Candidate/year count report saved; historical identities remain unverified.')


if __name__ == '__main__':
    main()
