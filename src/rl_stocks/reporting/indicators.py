"""Paired input-family attribution against a fixed, provenance-verified control."""
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean,stdev
from rl_stocks.core.contracts import digest
from rl_stocks.training.curves import latest_points


def summarize_indicators(runs,protocol,source_proofs):
    if not source_proofs or len(set(source_proofs.values()))!=1:
        raise ValueError('matching learning-source proofs required')
    labels={identity:label for label,identity in protocol['datasets'].items()}
    groups=defaultdict(dict);assumptions=set();symbol_sets=set();benchmarks=set()
    for run in runs:
        if (run.get('status')!='complete' or run.get('smoke') or run['algorithm']!='sac'
                or run['device']!='cpu' or run['requested_steps']!=250000 or run['actual_steps']!=250000
                or run['reward_config']!={'kind':'net_log_equity','scale':1.}
                or run['git'].get('dirty')!='' or run['source_hash'] not in source_proofs):
            raise ValueError('complete clean CPU SAC 250k original-reward cells required')
        label=labels[run['dataset']];seed=run['seed']
        if seed not in protocol['seeds'] or seed in groups[label]:raise ValueError('unexpected or duplicate seed')
        assumptions.add(digest({key:run[key] for key in ('simulation','versions','resolved_parameters','selection_metric')}))
        points=latest_points(run.get('validation_checkpoints',run.get('points',[])))
        if not points or points[-1]['steps']!=250000 or points[-1]['phase']!='post_update':
            raise ValueError('completed post-update endpoint required')
        selected=list(run['selected_metrics'].values())
        symbol_sets.add(tuple(sorted(s['symbol'] for s in selected)))
        if not selected or len(symbol_sets)!=1:raise ValueError('unmatched symbol coverage')
        benchmarks.add(digest({s['symbol']:s['baselines']['buy_hold'] for s in selected}))
        if len(benchmarks)>1:raise ValueError('unmatched buy-and-hold benchmarks')
        selected_return=mean(s['policy']['net_return'] for s in selected)
        if abs(selected_return-run['selected_score'])>1e-10:raise ValueError('selected score mismatch')
        groups[label][seed]={
            'seed':seed,'bundle_id':run['bundle_id'],'selected_return_pct':100*selected_return,
            'final_return_pct':points[-1]['return_pct'],
            'selected_drawdown_pct':100*mean(s['policy']['max_drawdown'] for s in selected),
            'final_drawdown_pct':points[-1]['mean_max_drawdown_pct'],
            'selected_costs':mean(s['policy']['costs'] for s in selected),
            'selected_turnover':mean(s['policy']['turnover'] for s in selected),
            'buy_hold_return_pct':100*mean(s['baselines']['buy_hold']['net_return'] for s in selected),
            'points':[{k:v for k,v in p.items() if k!='symbols'} for p in points]}
    if len(assumptions)>1:raise ValueError('unmatched simulator, dependencies, algorithm settings or selection')
    complete_groups={label:[entries[s] for s in protocol['seeds']] for label,entries in groups.items()
                     if sorted(entries)==protocol['seeds']}
    families=[];contrasts=[];gate=protocol['feature_gate']
    for label,entries in sorted(complete_groups.items()):
        row={'family':label,'seeds':entries}
        for field in ('selected_return_pct','final_return_pct','selected_drawdown_pct',
                      'final_drawdown_pct','selected_costs','selected_turnover','buy_hold_return_pct'):
            row['mean_'+field]=mean(e[field] for e in entries)
        row['selected_seed_std_pp']=stdev(e['selected_return_pct'] for e in entries)
        if label!='baseline' and 'baseline' in complete_groups:
            controls=complete_groups['baseline']
            selected=[e['selected_return_pct']-b['selected_return_pct'] for e,b in zip(entries,controls)]
            final=[e['final_return_pct']-b['final_return_pct'] for e,b in zip(entries,controls)]
            selected_dd=mean(b['selected_drawdown_pct']-e['selected_drawdown_pct'] for e,b in zip(entries,controls))
            final_dd=mean(b['final_drawdown_pct']-e['final_drawdown_pct'] for e,b in zip(entries,controls))
            passed=(mean(selected)>=gate['mean_selected_gain_pp'] and sum(g>0 for g in selected)>=gate['paired_positive_seeds']
                and mean(final)>=gate['minimum_mean_final_gain_pp']
                and selected_dd<=gate['maximum_selected_drawdown_worsening_pp']
                and final_dd<=gate['maximum_final_drawdown_worsening_pp'])
            row['feature_gate']={'passed':passed,'paired_selected_gains_pp':selected,'mean_selected_gain_pp':mean(selected),
                'paired_final_gains_pp':final,'mean_final_gain_pp':mean(final),
                'selected_drawdown_worsening_pp':selected_dd,'final_drawdown_worsening_pp':final_dd}
        families.append(row)
    complete=set(complete_groups)==set(protocol['datasets'])
    if complete:
        weights={
            'supply_demand_without_sector':{'supply_demand':1,'baseline':-1},
            'supply_demand_with_sector':{'combined':1,'rotation':-1},
            'sector_without_zones':{'rotation':1,'baseline':-1},
            'sector_with_zones':{'combined':1,'supply_demand':-1},
            'interaction':{'combined':1,'supply_demand':-1,'rotation':-1,'baseline':1}}
        for name,terms in weights.items():
            gains=[sum(weight*complete_groups[label][i]['selected_return_pct'] for label,weight in terms.items())
                   for i in range(len(protocol['seeds']))]
            contrasts.append({'contrast':name,'paired_selected_gains_pp':gains,'mean_selected_gain_pp':mean(gains),
                              'seed_std_pp':stdev(gains)})
    return {'complete':complete,'received_cells':sum(len(g) for g in groups.values()),'expected_cells':12,
        'test_evaluated':False,'paper_eligible':False,'families':families,'contrasts':contrasts,
        'source_proofs':source_proofs,'protocol':protocol,
        'interpretation':'Paired validation-selected comparisons; seed variability is descriptive, not an out-of-sample or significance claim.'}


def report_indicators(runs,protocol,source_proofs,destination):
    value=summarize_indicators(runs,protocol,source_proofs)
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    path.with_suffix('.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    rows=[{k:v for k,v in f.items() if k not in ('seeds','feature_gate')} for f in value['families']]
    if rows:
        with path.with_suffix('.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=3,cols=1,subplot_titles=('Selected and final net validation returns (%)',
        'Current validation return by training step (%)','Maximum drawdown (%)'))
    labels=[f['family'] for f in value['families']]
    for field,name,row in [('mean_selected_return_pct','Selected',1),('mean_final_return_pct','Final',1),
                           ('mean_selected_drawdown_pct','Selected drawdown',3),('mean_final_drawdown_pct','Final drawdown',3)]:
        fig.add_trace(go.Bar(x=labels,y=[f[field] for f in value['families']],name=name),row=row,col=1)
    for family in value['families']:
        entries=family['seeds'];common=sorted(set.intersection(*(set(p['steps'] for p in e['points']) for e in entries)))
        points=[{p['steps']:p for p in e['points']} for e in entries]
        fig.add_trace(go.Scatter(x=common,y=[mean(p[s]['return_pct'] for p in points) for s in common],
            name=family['family'],mode='lines+markers'),row=2,col=1)
    fig.update_layout(height=1100,barmode='group',title='Matched 250k input-family comparison — validation only')
    fig.write_html(str(path),include_plotlyjs=True)
    return value
