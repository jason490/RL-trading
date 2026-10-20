"""Saved-path participation/cost diagnostics; no policy or cost-free replay."""
from collections import Counter
import math
from statistics import mean
from rl_stocks.evaluation.behavior import behavior


def episode_diagnosis(result, initial=100000.):
    rows=result['rows'];events=result['events'];holding=result['baselines']['buy_hold']
    if (result['split']!='validation' or not rows
        or any(not '2021-01-01'<=r['session']<='2023-12-31' for r in rows)
        or [r['session'] for r in rows]!=sorted({r['session'] for r in rows})):
        raise ValueError('unique development-validation sessions required')
    if not math.isfinite(initial) or initial<=0:raise ValueError('positive initial equity required')
    if rows[0]['previous_equity']!=initial:raise ValueError('initial equity mismatch')
    if [r['session'] for r in rows]!=[r['session'] for r in holding['rows']][:len(rows)]:
        raise ValueError('holding session mismatch')
    b=behavior(result);side=result['directional']
    values=[b['reported_costs'],b['total_decomposed_costs'],side['long_costs']+side['short_costs']]
    if not all(math.isfinite(v) and v>=0 for v in values) or any(not math.isclose(v,values[0],abs_tol=1e-5,rel_tol=1e-9) for v in values):
        raise ValueError('cost reconciliation failed')
    pnl=side['long_net_pnl']+side['short_net_pnl']
    net=result['metrics']['net_return'];hold=holding['metrics']['net_return']
    if (not math.isclose(pnl,rows[-1]['equity']-initial,abs_tol=1e-5,rel_tol=1e-9)
        or not math.isclose(net,rows[-1]['equity']/initial-1,abs_tol=1e-10)):
        raise ValueError('side/equity/return reconciliation failed')
    sessions={r['session'] for r in rows}
    if any(e['time'][:10] not in sessions for e in events):raise ValueError('fill outside saved path')
    if any(not math.isfinite(r['exposure']) for r in rows):raise ValueError('nonfinite exposure')
    # Full holding baseline can outlast an early-terminated policy. This is a
    # documented different horizon; refuse it here rather than conceal it.
    if len(rows)!=len(holding['rows']):raise ValueError('complete matched paths required')
    hold_cost=holding['metrics']['costs'];cost_drag=100*(values[0]-hold_cost)/initial
    if not math.isfinite(hold_cost) or hold_cost<0:raise ValueError('invalid holding costs')
    noncost=100*(net+values[0]/initial-hold-hold_cost/initial)
    resizes=[e for e in events if e['reason'] in ('add','reduce') and (e['quantity']-e['delta'])*e['quantity']>0]
    small=[]
    for e in resizes:
        pretrade=e['equity']+e['fee']+e['impact']
        if pretrade<=0 or not math.isfinite(pretrade) or not math.isfinite(e['reference_price']) or e['reference_price']<=0:
            raise ValueError('invalid rebalance notional/equity')
        if abs(e['delta'])*e['reference_price']/pretrade<.05:small.append(e)
    return {'symbol':result['symbol'],**b,
        'buy_hold_return':hold,'net_excess_pp':100*(net-hold),
        'incremental_cost_drag_pp':cost_drag,'saved_path_noncost_excess_pp':noncost,
        'mean_signed_exposure':mean(r['exposure'] for r in rows),
        'mean_long_exposure':mean(max(r['exposure'],0.) for r in rows),
        'mean_short_abs_exposure':mean(max(-r['exposure'],0.) for r in rows),
        'long_return_contribution':side['long_net_pnl']/initial,
        'short_return_contribution':side['short_net_pnl']/initial,
        'long_touched_fraction':side['long_active_sessions']/len(rows),
        'short_touched_fraction':side['short_active_sessions']/len(rows),
        'discretionary_same_side_resize_fills':len(resizes),
        'same_side_resizes_below_5pct_equity':len(small),
        'saved_fees_impact_on_small_resizes_pp':100*sum(e['fee']+e['impact'] for e in small)/initial,
        'holding_limit_fills':sum(e['reason']=='holding_limit' for e in events)}


def diagnose_groups(results,stocks):
    if len({r['symbol'] for r in results.values()})!=len(results):raise ValueError('one episode per symbol required')
    records=[episode_diagnosis(r) for r in results.values()]
    groups={}
    keys=('net_return','buy_hold_return','net_excess_pp','incremental_cost_drag_pp','saved_path_noncost_excess_pp',
          'reported_costs','mean_abs_exposure','mean_signed_exposure','mean_long_exposure','mean_short_abs_exposure',
          'long_fraction','short_fraction','flat_fraction','long_touched_fraction','short_touched_fraction',
          'long_return_contribution','short_return_contribution','fills','discretionary_same_side_resize_fills',
          'holding_limit_fills','protective_reentries_within_one_session',
          'same_side_resizes_below_5pct_equity','saved_fees_impact_on_small_resizes_pp')
    for kind in ('stock','etf'):
        rows=[r for r in records if (r['symbol'] in stocks)==(kind=='stock')]
        if not rows:raise ValueError('both stock/ETF groups required')
        reasons=Counter()
        for r in rows:reasons.update(r['fill_reasons'])
        groups[kind]={'count':len(rows),'equal_weight_means':{k:mean(r[k] for r in rows) for k in keys},'total_fill_reasons':dict(reasons)}
    return {'groups':groups,'per_symbol':records,
        'interpretation':'Fixed saved paths only. Net excess equals non-cost excess minus incremental cost drag. Add-back is not a zero-cost replay or causal attribution. Exposure/fractions are end-of-session; touched fractions include intraday positions and may overlap. Future declining-stock cohorts never select policies.'}
