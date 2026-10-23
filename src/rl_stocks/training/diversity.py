"""Audit cached training expansion and preserve original preprocessing/validation."""
from collections import Counter
from copy import deepcopy
from rl_stocks.core.contracts import digest

ORIGINAL = '672a21621091a56c1c7beb30557464e7edd8df38874494659bab2c83738e1bf9'
EXPANDED = 'c2e6c58c4d4b0a9853b5b37851c731f559f42e597e9eefcf77796d874db31398'
ETFS = ('SPY','QQQ','IWM','DIA')


def audit_sources(original, expanded):
    for data in (original, expanded):
        if (data['manifest']['included_splits'] != ['train','validation']
                or data['manifest']['unverified_override'] is not False
                or data['manifest']['purge_sessions'] != 20
                or data['manifest']['max_holding_sessions'] != 20):
            raise ValueError('verified purged training/development source required')
        for e in data['episodes'].values():
            bars=e['bars']; dates=[b['session'] for b in bars]
            if (e['split'] not in ('train','validation') or len(bars)!=len(e['features'])
                    or len(bars)!=len(e['atr']) or dates!=sorted(set(dates))
                    or any(b['symbol']!=e['symbol'] for b in bars)
                    or not dates or dates[-1]>('2020-12-31' if e['split']=='train' else '2023-12-31')
                    or e['split']=='validation' and dates[0]<'2021-01-01'):
                raise ValueError('aligned causal split rows required')
    if original['schema']!=expanded['schema'] or original['names']!=expanded['names']:
        raise ValueError('same market feature schema required')
    if (original['manifest']['evaluation_symbols']!=expanded['manifest']['evaluation_symbols']
            or original['manifest']['splits']!=expanded['manifest']['splits']):
        raise ValueError('same evaluation panel and split boundaries required')
    for key,e in original['episodes'].items():
        if expanded['episodes'].get(key)!=e:
            raise ValueError('original training and validation episodes must remain exact')
    original_validation={k:e for k,e in original['episodes'].items() if e['split']=='validation'}
    expanded_validation={k:e for k,e in expanded['episodes'].items() if e['split']=='validation'}
    if original_validation!=expanded_validation:
        raise ValueError('no additional or modified validation episodes allowed')
    train={}; sectors={}
    for arm,data in [('original',original),('expanded',expanded)]:
        episodes=[e for e in data['episodes'].values() if e['split']=='train']
        symbols=[e['symbol'] for e in episodes]
        if len(symbols)!=len(set(symbols)) or set(ETFS)-set(symbols):
            raise ValueError('one episode per symbol and four broad ETFs required')
        if any(len(e['bars'])<513 for e in episodes):
            raise ValueError('every symbol must support a complete 512-transition window')
        annotations={i['symbol']:i for i in data['manifest']['universe_config']['instruments']}
        if any(s not in annotations or annotations[s]['kind'] not in ('stock','broad_etf')
               or (s in ETFS)!=(annotations[s]['kind']=='broad_etf') for s in symbols):
            raise ValueError('explicit stock and broad ETF kinds required')
        train[arm]=dict(episodes=len(episodes),stocks=len(symbols)-4,etfs=4,
            rows=sum(len(e['bars']) for e in episodes),symbols=sorted(symbols),
            date_ranges={e['symbol']:dict(first=e['bars'][0]['session'],last=e['bars'][-1]['session'],rows=len(e['bars'])) for e in episodes})
        sectors[arm]=dict(Counter(annotations[s]['sector'] for s in symbols if s not in ETFS))
    if set(train['original']['symbols'])-set(train['expanded']['symbols']):
        raise ValueError('expanded training must contain original symbols')
    return dict(training=train,current_sector_annotation_counts=sectors,
        additional_training_symbols=sorted(set(train['expanded']['symbols'])-set(train['original']['symbols'])),
        original_episodes_exact=True,validation_exact=True,
        normalizers_equal=original['normalizer']==expanded['normalizer'],
        original_normalizer_sha256=digest(original['normalizer']),expanded_normalizer_sha256=digest(expanded['normalizer']),
        future_outcomes_used=False,historical_membership_verified=False,
        interpretation='Retrospective survivor expansion; current sector counts are annotations, not causal historical classifications. IPO onsets remain explicit; no missing history or delisting outcome is invented.')


def fixed_preprocessing_expansion(original, expanded):
    audit=audit_sources(original,expanded)
    result=deepcopy(expanded)
    result['normalizer']=deepcopy(original['normalizer'])
    result['manifest']['sac_diversity']=dict(version=1,source_original=digest(original),source_expanded=digest(expanded),
        normalization='Original immutable training-only preprocessing, unchanged; no expanded refit',
        normalizer_sha256=audit['original_normalizer_sha256'],
        validation='Exact original26-symbol development episodes; no test reconstructed',
        sampling='Planned fixed original stock/ETF probabilities; explicit kinds, full windows; no future-outcome labels')
    return result

PROTOCOL_SHA256='3fd2f8f484d40d4abec51524ebb4c4087d56924103622019107a8d2b6931e616'


def protocol():
    import json
    from pathlib import Path
    from rl_stocks.rl.registry.factory import resolved_parameters
    p=json.loads((Path(__file__).resolve().parents[3]/'configs/experiments/sac_diversity.json').read_text())
    if digest(p)!=PROTOCOL_SHA256:raise ValueError('SAC diversity protocol changed')
    params=resolved_parameters('sac');params['net_arch']=params.pop('policy_kwargs')['net_arch']
    if params!=p['parameters'] or digest(p['frozen_normalizer'])!=p['normalizer_sha256']:raise ValueError('SAC diversity parameters/preprocessing changed')
    return p


def validate_payload(data,arm):
    p=protocol()
    if arm not in p['arms'] or digest(data)!=p['datasets'][arm]:raise ValueError('diversity dataset identity mismatch')
    if (digest(data['normalizer'])!=p['normalizer_sha256']
        or data['manifest']['included_splits']!=['train','validation'] or data['manifest']['unverified_override'] is not False
        or sorted(e['symbol'] for e in data['episodes'].values() if e['split']=='train')!=p['training_symbols'][arm]
        or any(e['split'] not in ('train','validation') or any(b['session']>('2020-12-31' if e['split']=='train' else '2023-12-31') for b in e['bars']) for e in data['episodes'].values())):
        raise ValueError('diversity data/preprocessing contract mismatch')
    return data


def sampling_summary(arm,smoke):
    p=protocol();n=len(p['training_symbols'][arm])
    return dict(mode=p['start_sampling'],window_transitions=128 if smoke else 512,episode_count=n,
        selection='Fixed group probability, shuffled episode cycles within group, uniform full-window start',
        future_outcomes_used=False,downtrend_counts_measured=False,
        group_probability=dict(stock_numerator=22,etf_numerator=4,denominator=26),etf_symbols=list(ETFS),
        episode_cycle='Without replacement until group exhausted',group_episodes=dict(stock=n-4,etf=4))


def validate_counts(counts,arm,smoke,steps):
    p=protocol();summary=sampling_summary(arm,smoke);allowed=set(p['training_symbols'][arm]);groups=counts.get('history_groups',{})
    if (counts.get('mode')!=p['start_sampling'] or counts.get('downtrend_counts_measured') is not False
        or type(counts.get('transitions')) is not int or counts.get('transitions')!=steps or counts.get('downtrend_starts')!=0 or counts.get('downtrend_transitions')!=0
        or type(counts.get('starts')) is not int or counts['starts']<=0 or not groups):
        raise ValueError('diversity sampling totals mismatch')
    per_symbol={s:dict(starts=0,transitions=0) for s in allowed}
    for key,row in groups.items():
        symbol,period=key.split(':')
        if (symbol not in allowed or period!='recent' or set(row)!={'starts','transitions'}
            or any(type(x) is not int or x<0 for x in row.values()) or row['transitions']>summary['window_transitions']*row['starts']):
            raise ValueError('invalid diversity symbol sampling')
        per_symbol[symbol]=row
    if sum(r['starts'] for r in per_symbol.values())!=counts['starts'] or sum(r['transitions'] for r in per_symbol.values())!=steps:
        raise ValueError('diversity per-symbol totals mismatch')
    if 'year_counts' in counts:
        from rl_stocks.environment.sampling.years import validate_year_counts
        years=validate_year_counts(counts['year_counts'],counts['starts'],steps)
        if (set(years['per_symbol'])-allowed or any(
            {k:years['per_symbol'].get(s,{}).get(k,0) for k in ('starts','transitions')}!=row
            for s,row in per_symbol.items())):
            raise ValueError('year counts differ from symbol sampling')
    group_counts={}
    for kind in ('stock','etf'):
        rows=[r for s,r in per_symbol.items() if (s in ETFS)==(kind=='etf')]
        starts=[r['starts'] for r in rows]
        if max(starts)-min(starts)>1:raise ValueError('diversity group cycle imbalance')
        group_counts[kind]={k:sum(r[k] for r in rows) for k in ('starts','transitions')}
    if not smoke and any(r['starts']<=0 or r['transitions']<=0 for r in per_symbol.values()):
        raise ValueError('diversity screen did not actually cover all symbols')
    return dict(summary=summary,group_counts=group_counts,symbols_with_transitions=sum(r['transitions']>0 for r in per_symbol.values()),
                actual_etf_transition_fraction=group_counts['etf']['transitions']/steps)


def validate_initialization(record):
    import math,re
    if (not math.isclose(record.get('alpha',float('nan')),1.,rel_tol=1e-6)
        or record.get('target_entropy')!=-4. or set(record.get('network_sha256',{}))!={'actor','critic','critic_target'}
        or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in record['network_sha256'].values())):
        raise ValueError('diversity initialization mismatch')
    return record
