"""Observed action/close-position relationships, never counterfactual P&L."""
from collections import Counter
import math
from statistics import mean

from rl_stocks.reporting.participation import episode_diagnosis

CATEGORIES = ('flat_request', 'same_direction', 'opposite_direction', 'flat_rejected',
              'flat_after_protection', 'flat_after_holding_limit', 'flat_approved_zero', 'flat_other')


def worst_drawdown_window(result):
    """Ex-post descriptive window only; never a feature or selection signal."""
    rows=result['rows']; peak=100000.; peak_index=-1; low=0.; window=(-1,-1,peak,peak)
    costs_by_day=Counter()
    for event in result['events']:costs_by_day[event['time'][:10]]+=event['fee']+event['impact']
    cumulative=[0.]; running=0.
    for i,row in enumerate(rows):
        running+=costs_by_day[row['session']]+row.get('borrow_cost',0.);cumulative.append(running)
        if row['equity']>peak:peak=row['equity'];peak_index=i
        dd=row['equity']/peak-1
        if dd<low:low=dd;window=(peak_index,i,peak,row['equity'])
    if not math.isclose(low,result['metrics']['max_drawdown'],abs_tol=1e-12,rel_tol=0):
        raise ValueError('drawdown window metric mismatch')
    start,end,peak_equity,trough_equity=window;subset=rows[start+1:end+1]
    cost=(cumulative[end+1]-cumulative[start+1])/peak_equity
    return dict(peak_session='initial' if start<0 else rows[start]['session'],
        trough_session='initial' if end<0 else rows[end]['session'],sessions=len(subset),
        max_drawdown=low,cost_fraction_of_peak=cost,noncost_loss_fraction=low+cost,
        mean_abs_end_exposure=mean(abs(r['exposure']) for r in subset) if subset else 0.,
        end_short_fraction=mean(r['quantity']<0 for r in subset) if subset else 0.)


def diagnose_episode(result):
    accounting=episode_diagnosis(result)  # Require causal saved dates and reconciled costs/P&L.
    rows=result['rows']; events={row['session']:set() for row in rows}
    for event in result['events']:events[event['time'][:10]].add(event['reason'])
    counts=Counter(); pnl=Counter(); requests=[]; policy_requests=[]; previous=100000.
    for row in rows:
        action=row['raw_action']
        if len(action)!=4 or any(not math.isfinite(v) or abs(v)>1 for v in action):
            raise ValueError('bounded finite four-component action required')
        policy=row.get('policy_action',action)
        if len(policy)!=4 or any(not math.isfinite(v) or abs(v)>1 for v in policy):
            raise ValueError('bounded finite original policy action required')
        policy_requests.append(float(policy[0]))
        requested=float(action[0]); quantity=row['quantity']
        if not math.isclose(row['previous_equity'],previous,abs_tol=1e-6,rel_tol=0):
            raise ValueError('saved equity chain mismatch')
        previous=row['equity']; requests.append(requested)
        if requested==0: category='flat_request'
        elif requested*quantity>0: category='same_direction'
        elif requested*quantity<0: category='opposite_direction'
        elif row.get('rejected'): category='flat_rejected'
        elif events[row['session']] & {'stop','target_near','target_far'}: category='flat_after_protection'
        elif 'holding_limit' in events[row['session']]: category='flat_after_holding_limit'
        elif row.get('approved_target_quantity')==0: category='flat_approved_zero'
        else: category='flat_other'
        counts[category]+=1
        pnl[category]+=(row['equity']-row['previous_equity'])/100000.
    if not math.isclose(sum(pnl.values()),result['metrics']['net_return'],abs_tol=1e-10,rel_tol=0):
        raise ValueError('action-category P&L reconciliation failed')
    return dict(symbol=result['symbol'],sessions=len(rows),worst_drawdown=worst_drawdown_window(result),
        net_return=accounting['net_return'],incremental_cost_drag_pp=accounting['incremental_cost_drag_pp'],
        noncost_excess_pp=accounting['saved_path_noncost_excess_pp'],
        long_return_contribution=accounting['long_return_contribution'],short_return_contribution=accounting['short_return_contribution'],
        policy_zero_fraction=mean(v==0 for v in policy_requests),
        mean_policy_long_output=mean(max(v,0) for v in policy_requests),
        mean_policy_short_output=mean(max(-v,0) for v in policy_requests),
        requested_long_fraction=mean(v>0 for v in requests),requested_short_fraction=mean(v<0 for v in requests),
        mean_requested_long_exposure=mean(max(v,0) for v in requests),
        mean_requested_short_exposure=mean(max(-v,0) for v in requests),
        mean_end_long_exposure=mean(max(r['exposure'],0) for r in rows),
        mean_end_short_exposure=mean(max(-r['exposure'],0) for r in rows),
        categories={k:dict(sessions=counts[k],fraction=counts[k]/len(rows),observed_return_contribution=pnl[k]) for k in CATEGORIES})


def diagnose_groups(results,stocks):
    records=[diagnose_episode(r) for r in results.values()]
    if len({r['symbol'] for r in records})!=len(records):raise ValueError('one episode per symbol required')
    groups={}
    for kind in ('stock','etf'):
        rows=[r for r in records if (r['symbol'] in stocks)==(kind=='stock')]
        if not rows:raise ValueError('both groups required')
        groups[kind]=dict(count=len(rows),means={k:mean(r[k] for r in rows) for k in
            ('net_return','incremental_cost_drag_pp','noncost_excess_pp','long_return_contribution','short_return_contribution',
             'policy_zero_fraction','mean_policy_long_output','mean_policy_short_output','requested_long_fraction','requested_short_fraction','mean_requested_long_exposure',
             'mean_requested_short_exposure','mean_end_long_exposure','mean_end_short_exposure')},
            worst_drawdown={k:mean(r['worst_drawdown'][k] for r in rows) for k in ('max_drawdown','cost_fraction_of_peak','noncost_loss_fraction','mean_abs_end_exposure','end_short_fraction')},
            categories={k:{field:mean(r['categories'][k][field] for r in rows) for field in
                ('fraction','observed_return_contribution')} for k in CATEGORIES})
    return dict(groups=groups,per_symbol=records,
        interpretation='Mutually exclusive same-session observations, not causal attribution or counterfactual returns. Request is decided on the prior completed bar; close position follows gaps, fills, protections and holding exits. Priority: flat request, same/opposite direction, then flat rejection/protection/holding/approved-zero/other. P&L includes the opening position and costs; it is not the isolated value of this action.')
