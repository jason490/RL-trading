"""Exact saved-fill equity bridge; descriptive evaluation, never policy inputs."""
import math
from types import SimpleNamespace

COMPONENTS=('overnight','inherited_intraday','fill_to_close','dividends','fees','impact','borrow')


def holding_bridge(episode, saved, config):
    """Reconstruct only the declared holding baseline, never a policy or orders."""
    from rl_stocks.environment.accounting.portfolio import Portfolio
    p=Portfolio(cash=config.initial_equity);rows=[];previous=config.initial_equity
    for i,bar in enumerate(episode['bars'][1:]):
        p.corporate_action(SimpleNamespace(**bar))
        if i==0:
            shares=int(p.cash/(bar['open']*(1+(config.fee_bps+config.impact_bps)/10000)))
            p.trade(shares,bar['open'],bar['open_time'],'buy_hold',config.fee_bps,config.impact_bps)
        equity=p.equity(bar['close'])
        rows.append(dict(session=bar['session'],previous_equity=previous,equity=equity,quantity=p.quantity,borrow_cost=0.))
        previous=equity
    if len(rows)!=len(saved['rows']) or any(a['session']!=b['session'] or a['equity']!=b['equity'] for a,b in zip(rows,saved['rows'],strict=True)):
        raise ValueError('reconstructed holding path differs from saved baseline')
    return bridge(episode,rows,p.events,config.initial_equity)


def bridge(episode, rows, events, initial=100000.):
    """Bridge each close using corporate actions and actual ordered fills.

    Inherited intraday is a bookkeeping term; fill_to_close offsets inventory
    changes. Neither is the isolated causal value of an exit or a signal.
    """
    bars=episode['bars']
    if (episode['split']!='validation' or not rows or len(rows)!=len(bars)-1
            or not math.isfinite(initial) or initial<=0
            or any(not '2021-01-01'<=b['session']<='2023-12-31' for b in bars)
            or [r['session'] for r in rows]!=[b['session'] for b in bars[1:]]
            or [b['session'] for b in bars]!=sorted({b['session'] for b in bars})):
        raise ValueError('complete matched development path required')
    by_day={r['session']:[] for r in rows}
    for event in events:
        day=event['time'][:10]
        if day not in by_day:raise ValueError('fill outside matched path')
        by_day[day].append(event)
    quantity=0;previous=initial;records=[];max_error=0.
    totals={k:0. for k in COMPONENTS}
    for old,bar,row in zip(bars[:-1],bars[1:],rows,strict=True):
        prices=[old['close'],bar['open'],bar['close'],bar.get('split',1.)]
        if any(not math.isfinite(v) or v<=0 for v in prices):raise ValueError('positive finite prices/split required')
        if not math.isclose(row['previous_equity'],previous,rel_tol=0,abs_tol=1e-6):
            raise ValueError('saved equity chain mismatch')
        inherited=quantity;exact=quantity*bar.get('split',1.);quantity=math.trunc(exact)
        values=dict(overnight=inherited*(bar.get('split',1.)*bar['open']-old['close']),
            inherited_intraday=quantity*(bar['close']-bar['open']),fill_to_close=0.,
            dividends=quantity*bar.get('dividend',0.),fees=0.,impact=0.,borrow=-row.get('borrow_cost',0.))
        for e in by_day[bar['session']]:
            delta=e['delta'];ref=e['reference_price'];fill=e['price']
            if (not delta or e['quantity']!=quantity+delta
                    or any(not math.isfinite(v) for v in [ref,fill,e['fee'],e['impact']])
                    or min(ref,fill)<=0 or min(e['fee'],e['impact'])<0
                    or not math.isclose(delta*(fill-ref),e['impact'],rel_tol=1e-9,abs_tol=1e-6)):
                raise ValueError('fill quantity/price/impact mismatch')
            values['fill_to_close']+=delta*(bar['close']-ref)
            values['fees']-=e['fee'];values['impact']-=e['impact'];quantity=e['quantity']
        if quantity!=row['quantity']:raise ValueError('closing quantity mismatch')
        if any(not math.isfinite(v) for v in values.values()) or values['borrow']>0:
            raise ValueError('finite cash flows and nonnegative borrow required')
        change=row['equity']-previous;error=abs(sum(values.values())-change)
        max_error=max(max_error,error)
        if not math.isclose(sum(values.values()),change,rel_tol=1e-9,abs_tol=1e-5):
            raise ValueError('per-session cash-flow bridge mismatch')
        for k in COMPONENTS:totals[k]+=values[k]
        records.append(dict(session=bar['session'],inherited_side='long' if inherited>0 else 'short' if inherited<0 else 'flat',
            components=values,equity_change=change,reconciliation_error=error))
        previous=row['equity']
    if not math.isclose(sum(totals.values()),previous-initial,rel_tol=1e-9,abs_tol=1e-5):
        raise ValueError('terminal cash-flow bridge mismatch')
    return dict(components_pp={k:100*v/initial for k,v in totals.items()},
        net_return_pp=100*(previous/initial-1),max_session_error=max_error,rows=records)
