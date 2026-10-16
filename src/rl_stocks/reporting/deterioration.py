"""Describe saved validation paths without claiming a causal training diagnosis."""
import math
from statistics import mean

from rl_stocks.evaluation.behavior import summarize_behavior


def action_summary(results):
    actions=[row['raw_action'] for result in results.values() for row in result['rows']]
    if not actions or any(len(a)!=4 or any(not math.isfinite(x) or abs(x)>1.000001 for x in a) for a in actions):
        raise ValueError('finite four-component actions required')
    switches=decisions=0
    for result in results.values():
        signs=[(r['raw_action'][0]>0)-(r['raw_action'][0]<0) for r in result['rows']]
        switches+=sum(a*b<0 for a,b in zip(signs,signs[1:]))
        decisions+=max(0,len(signs)-1)
    return {'component_order':['exposure','stop','target_near','zone_width'],
        'mean_action':[mean(a[i] for a in actions) for i in range(4)],
        'saturation_fraction':[mean(abs(a[i])>=.95 for a in actions) for i in range(4)],
        'target_sign_switch_fraction':switches/decisions if decisions else 0.}


def yearly_returns(results):
    per_year={}
    for result in results.values():
        groups={}
        for row in result['rows']:groups.setdefault(row['session'][:4],[]).append(row)
        for year,rows in groups.items():
            per_year.setdefault(year,[]).append(rows[-1]['equity']/rows[0]['previous_equity']-1)
    return {year:100*mean(values) for year,values in sorted(per_year.items())}


def diagnose_pair(metadata,selected,final):
    if metadata.get('status')!='complete' or not selected or set(selected)!=set(final):
        raise ValueError('complete matched selected/final results required')
    if any('-validation-' not in key for key in selected):
        raise ValueError('validation episodes only')
    initial={result['rows'][0]['previous_equity'] for result in [*selected.values(),*final.values()]}
    if len(initial)!=1 or next(iter(initial))<=0:raise ValueError('matched initial equity required')
    initial=next(iter(initial))
    phases={name:{'behavior':summarize_behavior(results)['equal_weight_means'],
                  'actions':action_summary(results),'yearly_return_pct':yearly_returns(results)}
            for name,results in [('selected',selected),('final',final)]}
    before=phases['selected']['behavior'];after=phases['final']['behavior']
    net_change=100*(after['net_return']-before['net_return'])
    cost_change=100*(after['reported_costs']-before['reported_costs'])/initial
    curves=metadata.get('validation_checkpoints',[])
    selected_point=next((p for p in curves if p['steps']==metadata['selected_step'] and
        math.isclose(p['score'],metadata['selected_score'],abs_tol=1e-10)),None)
    endpoint=next((p for p in reversed(curves) if p['steps']==metadata['actual_steps'] and p['phase']=='post_update'),None)
    return {'algorithm':metadata['algorithm'],'seed':metadata['seed'],'dataset':metadata['dataset'],
        'selected_step':metadata['selected_step'],'final_step':metadata['actual_steps'],
        'phases':phases,'final_minus_selected_return_pp':net_change,
        'additional_costs_as_initial_equity_pp':cost_change,
        'change_in_non_cost_pnl_as_initial_equity_pp':net_change+cost_change,
        'symbols_deteriorating':sum(final[k]['metrics']['net_return']<selected[k]['metrics']['net_return'] for k in selected),
        'symbol_count':len(selected),'selected_training':selected_point.get('training',{}) if selected_point else {},
        'final_training':endpoint.get('training',{}) if endpoint else {},
        'interpretation':'Validation-selected versus final saved paths. Cost add-back is an accounting decomposition of those paths, not a cost-free backtest. Year returns carry the existing positions across year boundaries. Losses, entropy and saturation are diagnostics; they do not prove overfitting or establish a causal mechanism.'}
