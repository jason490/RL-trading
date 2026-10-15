"""Compact trading-behavior diagnostics from existing simulation outputs."""
from collections import Counter
import math
from statistics import mean


PROTECTIVE_EXITS={'stop','target_near','target_far','holding_limit'}


def behavior(result):
    rows=result['rows'];events=result['events']
    if not rows:raise ValueError('nonempty simulation required')
    sessions={row['session']:i for i,row in enumerate(rows)}
    entries=closures=reentries=0;last_exit=None;last_reason=None
    for event in events:
        old=event['quantity']-event['delta']
        index=sessions.get(event['time'][:10])
        if old==0 and event['quantity']!=0:
            entries+=1
            if (last_reason in PROTECTIVE_EXITS and index is not None and
                    last_exit is not None and 0<=index-last_exit<=1):reentries+=1
        if old!=0 and event['quantity']==0:
            closures+=1;last_exit=index;last_reason=event['reason']
    fees=sum(e.get('fee',0.) for e in events)
    impact=sum(e.get('impact',0.) for e in events)
    borrow=sum(row.get('borrow_cost',0.) for row in rows)
    rewards=[math.log(row['equity']/row['previous_equity']) for row in rows
             if row.get('previous_equity',0)>0 and row['equity']>0]
    return {'sessions':len(rows),'fills':len(events),'entries':entries,'full_closures':closures,
        'protective_reentries_within_one_session':reentries,
        'fill_reasons':dict(Counter(e['reason'] for e in events)),
        'rejected_decisions':dict(Counter(r['rejected'] for r in rows if r.get('rejected'))),
        'risk_clipped_sessions':sum(bool(r.get('risk_clipped')) for r in rows),
        'fees':fees,'impact':impact,'borrow':borrow,'total_decomposed_costs':fees+impact+borrow,
        'reported_costs':result['metrics']['costs'],
        'mean_abs_exposure':mean(abs(r.get('exposure',0.)) for r in rows),
        'long_fraction':mean(r.get('quantity',0)>0 for r in rows),
        'short_fraction':mean(r.get('quantity',0)<0 for r in rows),
        'flat_fraction':mean(r.get('quantity',0)==0 for r in rows),
        'mean_abs_log_reward':mean(abs(v) for v in rewards) if rewards else 0.,
        'net_return':result['metrics']['net_return'],
        'max_drawdown':result['metrics']['max_drawdown'],
        'turnover':result['metrics']['turnover']}


def summarize_behavior(results):
    symbols={name:behavior(result) for name,result in results.items()}
    numeric=('fills','entries','full_closures','protective_reentries_within_one_session',
        'fees','impact','borrow','reported_costs','mean_abs_exposure','long_fraction',
        'short_fraction','flat_fraction','mean_abs_log_reward','net_return','max_drawdown','turnover')
    return {'symbols':symbols,'equal_weight_means':{
        key:mean(s[key] for s in symbols.values()) for key in numeric},
        'interpretation':'Separate symbol episodes; partial exits are individual fills, not round-trip trades.'}
