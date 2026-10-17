"""Validation-only matched training-universe comparison and compact progress plots."""
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.reporting.study import verified_sources
from rl_stocks.training.curves import latest_points


def verify_panels(data):
    panels = {arm: read_dataset(value['path']) for arm, value in data['datasets'].items()}
    a, b = panels['control'], panels['expanded']
    if a['manifest']['snapshot'] != b['manifest']['snapshot'] or a['names'] != b['names']:
        raise ValueError('unmatched source snapshot or feature panel')
    validation = lambda panel: {k: v for k, v in panel['episodes'].items() if v['split'] == 'validation'}
    if validation(a) != validation(b):
        raise ValueError('unmatched validation episodes')
    if any(e['split'] == 'test' for panel in panels.values() for e in panel['episodes'].values()):
        raise ValueError('retired test episodes must be absent')
    return panels


def summarize_universe(runs, protocol, data, approved_sources):
    panels = verify_panels(data)
    labels = {value['id']: arm for arm, value in data['datasets'].items()}
    records, seen, assumptions = [], set(), None
    expected = {(arm, seed) for arm in protocol['arms'] for seed in protocol['seeds']}
    for run in runs:
        arm = labels.get(run['dataset'])
        key = (arm, run['seed'])
        if key not in expected or key in seen:
            raise ValueError('unexpected or duplicate universe cell')
        if (run['status'] != 'complete' or run['algorithm'] != protocol['algorithm']
                or run['device'] != protocol['device'] or run['actual_steps'] != protocol['steps']
                or run['requested_steps'] != protocol['steps'] or run.get('smoke')
                or run['source_hash'] not in approved_sources or run['git'].get('dirty') != ''
                or run['reward_config'] != {'kind': 'net_log_equity', 'scale': 1.0}):
            raise ValueError('unmatched training conditions')
        current = (run['simulation'], run['versions'], run['selection_metric'],
                   run['resolved_parameters'], approved_sources[run['source_hash']],
                   digest({k: v['baselines'] for k, v in run['selected_metrics'].items()}))
        if assumptions is None:
            assumptions = current
        elif assumptions != current:
            raise ValueError('unmatched costs, parameters, dependencies or baselines')
        metrics = run['selected_metrics']
        required = {k for k, e in panels[arm]['episodes'].items() if e['split'] == 'validation'}
        if set(metrics) != required:
            raise ValueError('scored validation panel differs from declaration')
        history = run['validation_checkpoints']
        points = latest_points(history)
        if ([p['steps'] for p in points] != list(range(10000, protocol['steps'] + 1, 10000))
                or points[-1]['phase'] != 'post_update'
                or not all(math.isfinite(p['score']) for p in history)
                or not math.isclose(max(p['score'] for p in history), run['selected_score'], abs_tol=1e-9)
                or not math.isclose(mean(m['policy']['net_return'] for m in metrics.values()),
                                    run['selected_score'], abs_tol=1e-9)):
            raise ValueError('inconsistent checkpoint trajectory')
        records.append({'arm': arm, 'seed': run['seed'], 'bundle_id': run['bundle_id'],
            'dataset': run['dataset'], 'selected_step': run['selected_step'],
            'selected_return_pct': 100 * run['selected_score'],
            'final_return_pct': 100 * points[-1]['score'],
            'selected_drawdown_pct': 100 * mean(m['policy']['max_drawdown'] for m in metrics.values()),
            'final_drawdown_pct': points[-1]['mean_max_drawdown_pct'],
            'buy_hold_return_pct': 100 * mean(m['baselines']['buy_hold']['net_return'] for m in metrics.values()),
            'points': [{k: v for k, v in p.items() if k != 'symbols'} for p in points]})
        seen.add(key)
    families, grouped = [], {}
    fields = ('selected_return_pct', 'final_return_pct', 'selected_drawdown_pct',
              'final_drawdown_pct', 'buy_hold_return_pct')
    for arm in protocol['arms']:
        rows = sorted((r for r in records if r['arm'] == arm), key=lambda r: r['seed'])
        if [r['seed'] for r in rows] != protocol['seeds']:
            continue
        grouped[arm] = rows
        families.append({'arm': arm, **{'mean_' + field: mean(r[field] for r in rows) for field in fields},
            'selected_seed_std_pp': stdev(r['selected_return_pct'] for r in rows),
            'final_seed_std_pp': stdev(r['final_return_pct'] for r in rows)})
    comparison = None
    if seen == expected:
        pairs = list(zip(grouped['control'], grouped['expanded']))
        selected = [b['selected_return_pct'] - a['selected_return_pct'] for a, b in pairs]
        final = [b['final_return_pct'] - a['final_return_pct'] for a, b in pairs]
        selected_dd = mean(a['selected_drawdown_pct'] - b['selected_drawdown_pct'] for a, b in pairs)
        final_dd = mean(a['final_drawdown_pct'] - b['final_drawdown_pct'] for a, b in pairs)
        gate = protocol['gate']
        comparison = {'paired_selected_gains_pp': selected, 'paired_final_gains_pp': final,
            'mean_selected_gain_pp': mean(selected), 'mean_final_gain_pp': mean(final),
            'selected_drawdown_worsening_pp': selected_dd, 'final_drawdown_worsening_pp': final_dd,
            'passed': (mean(selected) >= gate['minimum_mean_selected_gain_pp']
                       and mean(final) >= gate['minimum_mean_final_gain_pp']
                       and sum(v > 0 for v in final) >= gate['minimum_paired_final_positive_seeds']
                       and selected_dd <= gate['maximum_selected_drawdown_worsening_pp']
                       and final_dd <= gate['maximum_final_drawdown_worsening_pp'])}
    return {'complete': seen == expected, 'completed_cells': len(seen), 'families': families,
            'comparison': comparison, 'runs': records, 'protocol': protocol,
            'verified_source_packages': approved_sources, 'test_evaluated': False, 'paper_eligible': False,
            'interpretation': 'Retrospective-universe development validation on a common panel; '
                              'equal means of separate stock accounts, not independent test performance.'}


def report_universe(roots, protocol, data, destination, repository):
    runs = [json.loads(path.read_text()) for root in roots for path in (root / 'runs').glob('*/metadata.json')]
    sources = verified_sources(runs, repository)
    value = summarize_universe(runs, protocol, data, sources)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig = make_subplots(rows=3, cols=1, subplot_titles=(
        'Selected and final validation return', 'Validation net return every 10k steps',
        'Incremental change in percentage points'))
    for label, field in (('Selected', 'selected_return_pct'), ('Final', 'final_return_pct')):
        fig.add_trace(go.Bar(name=label, x=[f['arm'] for f in value['families']],
            y=[f['mean_' + field] for f in value['families']]), row=1, col=1)
    for run in value['runs']:
        points = run['points']
        name = run['arm'] + ' seed ' + str(run['seed'])
        fig.add_trace(go.Scatter(name=name, x=[p['steps'] for p in points],
            y=[100 * p['score'] for p in points], mode='lines+markers'), row=2, col=1)
        fig.add_trace(go.Scatter(name=name + ' change', x=[p['steps'] for p in points],
            y=[p['change_pp'] for p in points], mode='lines+markers'), row=3, col=1)
    fig.update_layout(height=1100, title='Training universe expansion — development validation only')
    fig.write_html(path, include_plotlyjs=True)
    rows = [{k: v for k, v in r.items() if k != 'points'} for r in value['runs']]
    if rows:
        with path.with_suffix('.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return value
