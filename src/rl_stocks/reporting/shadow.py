"""Forward journal summaries; fixed review gates only after the full horizon."""
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
from statistics import mean, stdev
from rl_stocks.inference.shadow.journal import transaction, records
from rl_stocks.inference.shadow.protocol import read_protocol


def drawdown(equities, initial):
    peak = initial
    worst = 0.
    for value in equities:
        peak = max(peak, value)
        worst = min(worst, value/peak-1)
    return 100*worst


def report(root, destination):
    root = Path(root)
    protocol = read_protocol(root)
    with transaction(root/'journal.sqlite3') as db:
        journal = records(db)
    initial = protocol['simulation']['initial_equity']
    rows = defaultdict(list)
    benchmark = defaultdict(list)
    for day in journal:
        for row in day['settlements']:
            rows[(row['arm'], row['symbol'])].append(row)
        for row in day['baselines']:
            benchmark[row['symbol']].append(row)
    summaries = []
    curves = []
    for arm in protocol['arms']:
        returns, dd, costs = [], [], []
        for symbol in protocol['symbols']:
            history = rows[(arm['name'], symbol)]
            returns.append(100*(history[-1]['equity']/initial-1) if history else 0.)
            dd.append(drawdown([r['equity'] for r in history], initial))
            costs.append(history[-1]['costs'] if history else 0.)
        summaries.append({'arm': arm['name'], 'family': arm['family'], 'seed': arm['seed'],
                          'return_pct': mean(returns), 'mean_max_drawdown_pct': mean(dd),
                          'mean_costs': mean(costs)})
        for day in journal:
            subset = [r for r in day['settlements'] if r['arm'] == arm['name']]
            if subset:
                curves.append({'arm': arm['name'], 'session': day['session'],
                               'return_pct': mean(100*(r['equity']/initial-1) for r in subset)})
    base_return = mean(100*(v[-1]['equity']/initial-1) for v in benchmark.values()) if benchmark else 0.
    base_dd = mean(drawdown([r['equity'] for r in v], initial) for v in benchmark.values()) if benchmark else 0.
    settled = sum(bool(r['settlements']) for r in journal)
    families = []
    for family in sorted({a['family'] for a in protocol['arms']}):
        group = [r for r in summaries if r['family'] == family]
        excess = [r['return_pct']-base_return for r in group]
        ready = settled >= protocol['review_sessions']
        passed = (len(group) >= 3 and mean(excess) > protocol['review']['minimum_mean_excess_return_pp']
                  and sum(v > 0 for v in excess) >= protocol['review']['minimum_positive_seeds']
                  and base_dd-mean(r['mean_max_drawdown_pct'] for r in group)
                  <= protocol['review']['maximum_mean_drawdown_worsening_pp']) if ready else None
        families.append({'family': family, 'mean_return_pct': mean(r['return_pct'] for r in group),
                         'seed_std_pp': stdev(r['return_pct'] for r in group) if len(group) > 1 else None,
                         'mean_excess_return_pp': mean(excess), 'research_review_passed': passed})
    value = {'campaign': root.name, 'mode': 'shadow-no-orders', 'registered_at': protocol['registered_at'],
             'settled_sessions': settled, 'review_sessions': protocol['review_sessions'],
             'review_due': settled >= protocol['review_sessions'], 'paper_eligible': False,
             'buy_hold_return_pct': base_return, 'buy_hold_mean_max_drawdown_pct': base_dd,
             'cash_return_pct': 0., 'arms': summaries, 'families': families, 'curves': curves,
             'interpretation': protocol['interpretation'], 'records': journal}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix('.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    with destination.with_suffix('.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['arm', 'session', 'return_pct'])
        writer.writeheader()
        writer.writerows(curves)
    import plotly.graph_objects as go
    figure = go.Figure()
    for arm in protocol['arms']:
        points = [r for r in curves if r['arm'] == arm['name']]
        figure.add_trace(go.Scatter(x=[r['session'] for r in points], y=[r['return_pct'] for r in points],
                                   name=arm['name']))
    dates, base_values = [], []
    for day in journal:
        if day['baselines']:
            dates.append(day['session'])
            base_values.append(mean(100*(r['equity']/initial-1) for r in day['baselines']))
    figure.add_trace(go.Scatter(x=dates, y=base_values, name='Buy and hold'))
    figure.add_trace(go.Scatter(x=dates, y=[0.]*len(dates), name='Cash'))
    figure.update_layout(title='Forward shadow returns — separate stock accounts',
                         xaxis_title='Session', yaxis_title='Net return (%)')
    text = '<!doctype html><meta charset="utf-8"><title>Forward shadow tracking</title>'
    text += '<h1>Forward shadow tracking</h1><p>No broker orders. '+str(settled)+' settled sessions; '
    text += 'review after '+str(protocol['review_sessions'])+'. No gate before the horizon.</p>'
    text += '<p>'+html.escape(protocol['interpretation'])+'</p>'
    text += figure.to_html(full_html=False, include_plotlyjs=True)
    text += '<pre>'+html.escape(json.dumps({k: v for k, v in value.items()
                                           if k not in ('records', 'curves')}, indent=2))+'</pre>'
    destination.write_text(text)
    return value
