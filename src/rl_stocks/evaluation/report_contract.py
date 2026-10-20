"""Bind saved development reports to immutable episodes and accounting."""
from rl_stocks.evaluation.runner import baselines
from rl_stocks.evaluation.metrics import metrics
from rl_stocks.evaluation.direction import directional_attribution


def validate_development_report(results,dataset,config):
    episodes={k:e for k,e in dataset['episodes'].items() if e['split']=='validation'}
    if not episodes or set(results)!=set(episodes):raise ValueError('exact immutable validation episodes required')
    for key,episode in episodes.items():
        bars=episode['bars'][1:];r=results[key];rows=r['rows']
        if (r['split']!='validation' or r['symbol']!=episode['symbol'] or not rows or len(rows)>len(bars)
            or any(not '2021-01-01'<=b['session']<='2023-12-31' for b in episode['bars'])
            or [x['session'] for x in rows]!=[b['session'] for b in bars[:len(rows)]]
            or any(x['price']!=b['close'] for x,b in zip(rows,bars))):
            raise ValueError('saved report dates/prices differ from immutable validation')
        previous=config.initial_equity
        for row in rows:
            if row['previous_equity']!=previous:raise ValueError('saved equity chain mismatch')
            previous=row['equity']
        if len(rows)!=len(bars) and previous>0:raise ValueError('unexplained early report cutoff')
        if r['metrics']!=metrics(rows,r['events'],config.initial_equity):raise ValueError('saved metrics disagree with path')
        if r['baselines']!=baselines(episode,config):raise ValueError('saved holding/cash baseline mismatch')
        if r.get('directional')!=directional_attribution(episode,rows,r['events'],config.initial_equity):raise ValueError('saved directional accounting mismatch')
    return dict(episodes=len(episodes),dates_prices_metrics_baselines_directional_verified=True)
