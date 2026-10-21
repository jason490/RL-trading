"""Observed action/close-position relationships, never counterfactual P&L."""
from collections import Counter
import math
from statistics import mean

from rl_stocks.reporting.participation import episode_diagnosis

CATEGORIES = ('flat_request', 'same_direction', 'opposite_direction', 'flat_rejected',
              'flat_after_protection', 'flat_after_holding_limit', 'flat_approved_zero', 'flat_other')


def diagnose_episode(result):
    episode_diagnosis(result)  # Require causal saved dates and reconciled costs/P&L.
    rows=result['rows']; events={row['session']:set() for row in rows}
    for event in result['events']:events[event['time'][:10]].add(event['reason'])
    counts=Counter(); pnl=Counter(); requests=[]; previous=100000.
    for row in rows:
        action=row['raw_action']
        if len(action)!=4 or any(not math.isfinite(v) or abs(v)>1 for v in action):
            raise ValueError('bounded finite four-component action required')
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
    return dict(symbol=result['symbol'],sessions=len(rows),
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
            ('requested_long_fraction','requested_short_fraction','mean_requested_long_exposure',
             'mean_requested_short_exposure','mean_end_long_exposure','mean_end_short_exposure')},
            categories={k:{field:mean(r['categories'][k][field] for r in rows) for field in
                ('fraction','observed_return_contribution')} for k in CATEGORIES})
    return dict(groups=groups,per_symbol=records,
        interpretation='Mutually exclusive same-session observations, not causal attribution or counterfactual returns. Request is decided on the prior completed bar; close position follows gaps, fills, protections and holding exits. Priority: flat request, same/opposite direction, then flat rejection/protection/holding/approved-zero/other. P&L includes the opening position and costs; it is not the isolated value of this action.')
