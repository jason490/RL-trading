"""Small append-only validation records and percentage-point learning charts."""
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean


def latest_points(history):
    """Retain the last evaluation at each step (post-update supersedes on-step)."""
    points={}
    for row in history:points[row['steps']]=row
    return [points[step] for step in sorted(points)]


def curve_record(results,history,steps,phase,elapsed_seconds,training=None):
    from rl_stocks.evaluation.runner import selection_score
    score=selection_score(results)
    previous=[r for r in latest_points(history) if r['steps']<steps]
    prior=previous[-1]['score'] if previous else None
    anchors=[r for r in latest_points(history) if r['steps']==50000]
    anchor=score if steps==50000 else anchors[-1]['score'] if anchors else None
    metrics={name:{key:result['metrics'][key] for key in
                  ('net_return','max_drawdown','sharpe','costs','turnover')}
             for name,result in results.items()}
    score_bh=mean(result['baselines']['buy_hold']['metrics']['net_return'] for result in results.values())
    row={'steps':int(steps),'phase':phase,'score':score,'return_pct':100*score,
         'change_pp':None if prior is None else 100*(score-prior),
         'equity_change_pct_from_previous':None if prior is None or prior<=-1 else 100*((1+score)/(1+prior)-1),
         'gain_since_50k_pp':None if anchor is None else 100*(score-anchor),
         'best_return_pct':100*max([score]+[r['score'] for r in history]),
         'buy_hold_return_pct':100*score_bh,'excess_return_pp':100*(score-score_bh),
         'mean_max_drawdown_pct':100*mean(v['max_drawdown'] for v in metrics.values()),
         'mean_costs':mean(v['costs'] for v in metrics.values()),
         'mean_turnover':mean(v['turnover'] for v in metrics.values()),
         'elapsed_seconds':elapsed_seconds,'symbols':metrics,'training':training or {}}
    if not math.isfinite(score):raise ValueError('nonfinite validation return')
    return row


def append_record(path,record):
    path=Path(path)
    with path.open('a') as stream:
        stream.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def render_curve(history,destination):
    destination=Path(destination);points=latest_points(history)
    columns=('steps','phase','return_pct','change_pp','equity_change_pct_from_previous',
        'gain_since_50k_pp','best_return_pct','buy_hold_return_pct','excess_return_pp',
        'mean_max_drawdown_pct','mean_costs','mean_turnover','elapsed_seconds')
    with destination.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(points)
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    figure=make_subplots(rows=4,cols=1,shared_xaxes=True,subplot_titles=(
        'Validation total return (%) — separate symbol episodes',
        'Incremental change (percentage points)', 'Average maximum drawdown (%)','Average modeled costs ($)'))
    x=[r['steps'] for r in points]
    for name,key,row in [('Current','return_pct',1),('Best so far','best_return_pct',1),
             ('Buy and hold','buy_hold_return_pct',1),('Since previous checkpoint','change_pp',2),
             ('Since 50k','gain_since_50k_pp',2),('Drawdown','mean_max_drawdown_pct',3),('Costs','mean_costs',4)]:
        figure.add_trace(go.Scatter(x=x,y=[r[key] for r in points],name=name,mode='lines+markers'),row=row,col=1)
    figure.update_layout(height=1100,title='Validation learning curve; no final-test evaluation')
    figure.update_xaxes(title_text='Training transitions',row=4,col=1)
    figure.write_html(str(destination),include_plotlyjs=True)
