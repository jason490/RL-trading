"""Paired validation-only algorithm stability gates and compact learning charts."""
import csv
import json
import math
from pathlib import Path
from statistics import mean,stdev

from rl_stocks.core.contracts import digest
from rl_stocks.training.curves import latest_points


def paired_gate(control,candidate,gate):
    if [r['seed'] for r in control]!=[r['seed'] for r in candidate] or len(control)!=3:
        raise ValueError('three paired seeds required')
    selected=[b['selected_return_pct']-a['selected_return_pct'] for a,b in zip(control,candidate)]
    final=[b['final_return_pct']-a['final_return_pct'] for a,b in zip(control,candidate)]
    selected_dd=mean(a['selected_drawdown_pct']-b['selected_drawdown_pct'] for a,b in zip(control,candidate))
    final_dd=mean(a['final_drawdown_pct']-b['final_drawdown_pct'] for a,b in zip(control,candidate))
    reduction=mean(selected)-mean(final)
    # A smaller selected-minus-final gap is better: final gains must exceed selected gains.
    reduction=-reduction
    passed=(mean(selected)>=gate['minimum_mean_selected_gain_pp'] and
        mean(final)>=gate['minimum_mean_final_gain_pp'] and
        sum(x>0 for x in final)>=gate['minimum_paired_final_positive_seeds'] and
        selected_dd<=gate['maximum_selected_drawdown_worsening_pp'] and
        final_dd<=gate['maximum_final_drawdown_worsening_pp'] and
        reduction>=gate['minimum_reduction_in_mean_selected_minus_final_gap_pp'])
    return {'passed':passed,'paired_selected_gains_pp':selected,'paired_final_gains_pp':final,
        'mean_selected_gain_pp':mean(selected),'mean_final_gain_pp':mean(final),
        'selected_drawdown_worsening_pp':selected_dd,'final_drawdown_worsening_pp':final_dd,
        'reduction_in_mean_selected_minus_final_gap_pp':reduction}


def summarize_stability(runs,protocol,approved_sources):
    if not approved_sources or len(set(approved_sources.values()))!=1:raise ValueError('one verified learning source required')
    from rl_stocks.rl.registry.factory import resolved_parameters
    records=[];seen=set();reference=None
    for run in runs:
        key=(run['algorithm'],run['seed'])
        if key in seen:raise ValueError('duplicate stability cell')
        if (key[0] not in protocol['algorithms'] or key[1] not in protocol['seeds'] or run.get('smoke') or
            run.get('status')!='complete' or run.get('device')!='cpu' or run.get('dataset')!=protocol['dataset'] or
            run.get('actual_steps')!=protocol['steps'] or run.get('requested_steps')!=protocol['steps'] or
            run.get('reward_config')!={'kind':'net_log_equity','scale':1.} or
            run.get('milestones')!=[protocol['steps']] or run.get('evaluation_every')!=10000 or
            not run.get('compact_checkpoints') or run['git'].get('dirty')!='' or
            run['source_hash'] not in approved_sources):raise ValueError('unmatched stability configuration')
        expected=resolved_parameters(key[0],False,{'n_steps':1000,'batch_size':50} if key[0]!='sac' else None)
        if run['resolved_parameters']!=expected:raise ValueError('unexpected algorithm settings')
        metrics=run['selected_metrics']
        assumptions=(run['simulation'],run['versions'],run['selection_metric'],
            digest({k:r['baselines']['buy_hold'] for k,r in metrics.items()}),sorted(metrics))
        if reference is None:reference=assumptions
        elif assumptions!=reference:raise ValueError('unmatched costs, dependencies, selection or benchmarks')
        history=run['validation_checkpoints'];points=latest_points(history)
        if ([p['steps'] for p in points]!=list(range(10000,protocol['steps']+1,10000)) or
            points[-1]['phase']!='post_update' or any(not math.isfinite(p['score']) for p in history) or
            not math.isclose(max(p['score'] for p in history),run['selected_score'],abs_tol=1e-9) or
            not math.isclose(mean(m['policy']['net_return'] for m in metrics.values()),run['selected_score'],abs_tol=1e-9)):
            raise ValueError('inconsistent learning curve or selection')
        selected_dd=100*mean(m['policy']['max_drawdown'] for m in metrics.values())
        final=points[-1]
        records.append({'algorithm':key[0],'seed':key[1],'bundle_id':run['bundle_id'],
            'selected_step':run['selected_step'],'selected_return_pct':100*run['selected_score'],
            'final_return_pct':100*final['score'],'selected_drawdown_pct':selected_dd,
            'final_drawdown_pct':final['mean_max_drawdown_pct'],
            'buy_hold_return_pct':100*mean(m['baselines']['buy_hold']['net_return'] for m in metrics.values()),
            'points':[{k:v for k,v in p.items() if k!='symbols'} for p in points]})
        seen.add(key)
    groups={};families=[]
    for algorithm in protocol['algorithms']:
        rows=sorted((r for r in records if r['algorithm']==algorithm),key=lambda r:r['seed'])
        if [r['seed'] for r in rows]!=protocol['seeds']:continue
        groups[algorithm]=rows
        families.append({'algorithm':algorithm,'mean_selected_return_pct':mean(r['selected_return_pct'] for r in rows),
            'mean_final_return_pct':mean(r['final_return_pct'] for r in rows),
            'selected_seed_std_pp':stdev(r['selected_return_pct'] for r in rows),
            'final_seed_std_pp':stdev(r['final_return_pct'] for r in rows),
            'mean_selected_drawdown_pct':mean(r['selected_drawdown_pct'] for r in rows),
            'mean_final_drawdown_pct':mean(r['final_drawdown_pct'] for r in rows),
            'mean_selected_minus_final_gap_pp':mean(r['selected_return_pct']-r['final_return_pct'] for r in rows)})
    comparisons={}
    for control,candidate in [('sac','ppo'),('sac','recurrent_ppo'),('ppo','recurrent_ppo')]:
        if control in groups and candidate in groups:
            comparisons[candidate+'_versus_'+control]=paired_gate(groups[control],groups[candidate],protocol['gate'])
    return {'protocol':protocol,'complete':len(seen)==len(protocol['algorithms'])*len(protocol['seeds']),
        'completed_cells':len(seen),'families':families,'comparisons':comparisons,'runs':records,
        'verified_source_packages':approved_sources,'test_evaluated':False,'paper_eligible':False,
        'interpretation':'Validation-selected policies and final checkpoints, equal means of separate stock episodes. Recurrent models add capacity; their effect is not a parameter-count-matched proof of memory alone. No independent test performance claim.'}


def report_stability(runs,protocol,approved_sources,destination):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    value=summarize_stability(runs,protocol,approved_sources)
    fig=make_subplots(rows=3,cols=1,subplot_titles=('Selected and final validation returns','Current validation return every 10k steps','Selected minus final return gap'))
    rows=value['families'];names=[r['algorithm'] for r in rows]
    for label,key,error in [('Selected','mean_selected_return_pct','selected_seed_std_pp'),('Final','mean_final_return_pct','final_seed_std_pp')]:
        fig.add_trace(go.Bar(name=label,x=names,y=[r[key] for r in rows],error_y={'type':'data','array':[r[error] for r in rows]}),row=1,col=1)
    for run in value['runs']:
        fig.add_trace(go.Scatter(name=run['algorithm']+' seed '+str(run['seed']),x=[p['steps'] for p in run['points']],y=[100*p['score'] for p in run['points']],mode='lines+markers'),row=2,col=1)
    fig.add_trace(go.Bar(name='Return gap',x=names,y=[r['mean_selected_minus_final_gap_pp'] for r in rows]),row=3,col=1)
    fig.update_layout(height=1100,barmode='group',title='CPU algorithm stability — validation only; no held-out claim')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    fig.write_html(destination,include_plotlyjs=True)
    destination.with_suffix('.json').write_text(json.dumps(value,indent=2)+'\n')
    with destination.with_suffix('.csv').open('w',newline='') as stream:
        fields=['algorithm','mean_selected_return_pct','mean_final_return_pct','selected_seed_std_pp','final_seed_std_pp','mean_selected_drawdown_pct','mean_final_drawdown_pct','mean_selected_minus_final_gap_pp']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return value
