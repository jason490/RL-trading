"""Validation-only learning gates and compact multi-seed progress reports."""
from collections import defaultdict
import csv,json
from pathlib import Path
from statistics import mean,stdev
from rl_stocks.training.curves import latest_points


def assess_learning(histories,gate):
    if len(histories)!=3:raise ValueError('three paired seeds required')
    ends=[latest_points(h) for h in histories]
    starts=[];finals=[];selected_starts=[];selected_finals=[]
    for history,points in zip(histories,ends):
        anchor=next((r for r in points if r['steps']==50000 and r['phase']=='post_update'),None)
        end=points[-1]
        if anchor is None or end['phase']!='post_update' or end['steps']<=50000:
            raise ValueError('completed post-update 50k and later milestones required')
        starts.append(anchor);finals.append(end)
        selected_starts.append(max(r['score'] for r in history if r['steps']<=50000))
        selected_finals.append(max(r['score'] for r in history))
    paired=[100*(b['score']-a['score']) for a,b in zip(starts,finals)]
    selected_paired=[100*(b-a) for a,b in zip(selected_starts,selected_finals)]
    worsening=mean(a['mean_max_drawdown_pct']-b['mean_max_drawdown_pct'] for a,b in zip(starts,finals))
    historical_gain=100*(mean(selected_finals)-gate['historical_best_selected_mean_return'])
    passed=(mean(paired)>=gate['mean_current_gain_pp'] and mean(selected_paired)>=gate['mean_selected_gain_pp']
        and sum(v>0 for v in paired)>=gate['paired_positive_seeds']
        and sum(v>0 for v in selected_paired)>=gate['paired_positive_seeds']
        and worsening<=gate['maximum_drawdown_worsening_pp']
        and historical_gain>=gate['minimum_selected_gain_over_historical_pp'])
    return {'passed':passed,'mean_current_gain_pp':mean(paired),'paired_current_gains_pp':paired,
        'mean_selected_gain_pp':mean(selected_paired),'paired_selected_gains_pp':selected_paired,
        'drawdown_worsening_pp':worsening,'historical_selected_gain_pp':historical_gain,
        'mean_selected_return':mean(selected_finals),'mean_final_return':mean(r['score'] for r in finals),
        'interpretation':'Validation learning gate only; no new out-of-sample claim.'}


def summarize_learning(runs,protocol,expected_families=None):
    labels={v:k for k,v in protocol['datasets'].items()};groups=defaultdict(list);seen=set()
    for run in runs:
        if run.get('smoke'):continue
        if (run['device']!='cpu' or run['algorithm']!='sac' or run['status']!='complete'
                or run['actual_steps']!=run['requested_steps']):
            raise ValueError('complete CPU SAC cells required')
        if run['git'].get('dirty')!='':raise ValueError('dirty source upload')
        label=labels[run['dataset']];key=(label,run['reward_config']['scale'],run['seed'])
        if key in seen:raise ValueError('duplicate learning cell')
        seen.add(key);groups[key[:2]].append(run)
    families=[]
    for (label,scale),entries in sorted(groups.items()):
        entries.sort(key=lambda r:r['seed'])
        if [r['seed'] for r in entries]!=protocol['seeds']:continue
        if len({r['requested_steps'] for r in entries})!=1:raise ValueError('unmatched training budgets')
        histories=[r['validation_checkpoints'] for r in entries]
        gate=assess_learning(histories,protocol['gate'])
        points=[{r['steps']:r for r in latest_points(h)} for h in histories]
        common=sorted(set.intersection(*(set(p) for p in points)));aggregated=[];previous=None;anchor=None
        for step in common:
            rows=[p[step] for p in points];value=mean(r['return_pct'] for r in rows)
            if step==50000:anchor=value
            aggregated.append({'family':label,'reward_scale':scale,'steps':step,'mean_return_pct':value,
                'seed_std_pct':stdev(r['return_pct'] for r in rows),
                'mean_best_return_pct':mean(r['best_return_pct'] for r in rows),
                'change_pp':None if previous is None else value-previous,
                'gain_since_50k_pp':None if anchor is None else value-anchor,
                'mean_max_drawdown_pct':mean(r['mean_max_drawdown_pct'] for r in rows),
                'mean_costs':mean(r['mean_costs'] for r in rows),
                'buy_hold_return_pct':mean(r['buy_hold_return_pct'] for r in rows)})
            previous=value
        families.append({'family':label,'reward_scale':scale,'gate':gate,'points':aggregated})
    expected=expected_families or list(protocol['datasets'])
    complete=bool(families) and set(f['family'] for f in families)==set(expected)
    return {'protocol':protocol,'complete':complete,'completed_cells':len(seen),'families':families,
        'improvement_gate_passed':complete and any(f['gate']['passed'] for f in families),
        'test_evaluated':False,'runs':[{**{k:v for k,v in r.items() if k!='validation_checkpoints'},
            'points':[{k:v for k,v in p.items() if k!='symbols'} for p in latest_points(r['validation_checkpoints'])]}
            for r in runs if not r.get('smoke')]}


def report_learning(runs,protocol,destination,expected_families=None):
    value=summarize_learning(runs,protocol,expected_families)
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    path.with_suffix('.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    points=[p for f in value['families'] for p in f['points']]
    if points:
        with path.with_suffix('.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=(
        'Validation net return (%) — current checkpoints', 'Incremental changes (percentage points)',
        'Average maximum drawdown (%)'))
    for family in value['families']:
        rows=family['points'];x=[r['steps'] for r in rows];name=family['family']+' / scale '+str(family['reward_scale'])
        for key,label,row in [('mean_return_pct','current',1),('mean_best_return_pct','selected so far',1),
                 ('change_pp','since previous',2),('gain_since_50k_pp','since 50k',2),('mean_max_drawdown_pct','drawdown',3)]:
            fig.add_trace(go.Scatter(x=x,y=[r[key] for r in rows],name=name+' / '+label,mode='lines+markers'),row=row,col=1)
        fig.add_trace(go.Scatter(x=x,y=[r['buy_hold_return_pct'] for r in rows],name=name+' / buy hold',line={'dash':'dot'}),row=1,col=1)
    for run in value['runs']:
        fig.add_trace(go.Scatter(x=[p['steps'] for p in run['points']],y=[p['return_pct'] for p in run['points']],
            name=run['dataset'][:8]+' seed '+str(run['seed']),visible='legendonly',mode='lines+markers'),row=1,col=1)
    fig.update_layout(height=1000,title='Controlled CPU learning study — validation only; no final-test evaluation')
    fig.write_html(str(path),include_plotlyjs=True)
    return value
