"""Frozen historical augmentation protocol and immutable dataset acceptance."""
import json
from pathlib import Path
from rl_stocks.core.contracts import digest

PROTOCOL_SHA256='27514a57ebbd86c3bdaa412e6216ed23a65cb085fe82062037e39ebc8f4ae59c'


def protocol():
    value=json.loads((Path(__file__).resolve().parents[4]/'configs/experiments/historical_augmentation.json').read_text())
    if digest(value)!=PROTOCOL_SHA256:raise ValueError('historical protocol changed')
    return value


def validate_payload(data,arm):
    p=protocol()
    if arm not in p['arms'] or digest(data)!=p['datasets'][arm]:raise ValueError('historical dataset identity mismatch')
    meta=data['manifest']['history_augmentation'];episodes=data['episodes'].values()
    if (meta['arm']!=arm or meta['start_sampling']!='full_window_uniform'
        or data['manifest']['included_splits']!=['train','validation']
        or data['manifest']['unverified_override'] is not False
        or any(e['split'] not in ('train','validation') or any(b['session']>'2023-12-31' for b in e['bars']) for e in episodes)):
        raise ValueError('historical dataset contract mismatch')
    return data


def validate_counts(counts,arm,smoke,steps):
    window=128 if smoke else 512;n=26 if arm=='original' else 30
    summary=dict(mode='full_window_uniform',window_transitions=window,episode_count=n,
        selection='Uniform episode, then uniform start permitting the entire window',future_outcomes_used=False,downtrend_counts_measured=False)
    if counts.get('mode')!=summary['mode'] or counts.get('downtrend_counts_measured') is not False:raise ValueError('historical sampling mode mismatch')
    groups=counts.get('history_groups',{})
    universe=json.loads((Path(__file__).resolve().parents[4]/'configs/universes/sector_research.json').read_text())
    allowed={i['symbol'] for i in universe['instruments'] if i['role']=='tradable'}
    for key,v in groups.items():
        symbol,period=key.split(':')
        if (symbol not in allowed or period not in ('earlier','recent') or period=='earlier' and (arm!='older_etf' or symbol not in ('SPY','QQQ','IWM','DIA'))
            or set(v)!={'starts','transitions'} or any(type(x) is not int or x<0 for x in v.values())
            or v['transitions']>window*v['starts']):raise ValueError('invalid historical sampling group')
    if (not groups or counts.get('transitions')!=steps or counts.get('starts',0)<=0
        or sum(v['transitions'] for v in groups.values())!=steps
        or sum(v['starts'] for v in groups.values())!=counts['starts']
        or counts.get('downtrend_starts')!=0 or counts.get('downtrend_transitions')!=0):raise ValueError('historical sampling totals mismatch')
    return summary
