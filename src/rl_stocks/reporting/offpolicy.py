"""Saved off-policy path diagnostics without reconstructing missing dividends."""
import math
from statistics import mean
from rl_stocks.evaluation.behavior import behavior


def saved_episode(result,initial=100000.):
    rows=result['rows'];holding=result['baselines']['buy_hold'];sessions=[r['session'] for r in rows]
    if (result['split']!='validation' or not rows or sessions!=sorted(set(sessions))
        or any(not '2021-01-01'<=d<='2023-12-31' for d in sessions)
        or sessions!=[r['session'] for r in holding['rows']]):raise ValueError('complete matched development paths required')
    if not math.isfinite(initial) or initial<=0 or rows[0]['previous_equity']!=initial:raise ValueError('initial equity mismatch')
    if any(e['time'][:10] not in sessions for e in result['events']):raise ValueError('fill outside path')
    b=behavior(result);cost=b['reported_costs'];hold_cost=holding['metrics']['costs'];net=b['net_return'];hold=holding['metrics']['net_return']
    if (not all(math.isfinite(v) and v>=0 for v in (cost,hold_cost,b['total_decomposed_costs']))
        or not math.isclose(cost,b['total_decomposed_costs'],abs_tol=1e-5,rel_tol=1e-9)
        or not math.isclose(net,rows[-1]['equity']/initial-1,abs_tol=1e-10,rel_tol=0)):
        raise ValueError('saved cost/equity mismatch')
    if any(not math.isfinite(r['exposure']) for r in rows):raise ValueError('nonfinite exposure')
    drag=100*(cost-hold_cost)/initial;noncost=100*(net+cost/initial-hold-hold_cost/initial)
    if not math.isfinite(noncost):raise ValueError('finite matched holding required')
    held=[r for r in rows if r['quantity']]
    return dict(symbol=result['symbol'],**b,buy_hold_return=hold,net_excess_pp=100*(net-hold),
        incremental_cost_drag_pp=drag,saved_path_noncost_excess_pp=noncost,
        held_protection_fraction=mean(r.get('protection') is not None for r in held) if held else None,
        directional_pnl_reconstructed=False)
