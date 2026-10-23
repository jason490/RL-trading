"""Training-row subsets preserving causal source features and common development."""
from copy import deepcopy
from datetime import date
from rl_stocks.core.contracts import digest
from rl_stocks.reporting.history_coverage import window_coverage


def recent_subset(source,start='2016-01-01',steps=512):
    if not isinstance(start,str) or date.fromisoformat(start).isoformat()!=start or not '2010-01-01'<=start<='2020-12-31':
        raise ValueError('training-only ISO cutoff required')
    if (source['manifest']['included_splits']!=['train','validation']
        or any(e['split'] not in ('train','validation') for e in source['episodes'].values())):
        raise ValueError('train/development-only source required')
    if 'training_history' in source['manifest']:raise ValueError('original unfiltered source required')
    result=deepcopy(source);counts={};symbols=set()
    for key,old in source['episodes'].items():
        n=len(old['bars'])
        if n!=len(old['features']) or n!=len(old['atr']):raise ValueError('aligned source arrays required')
        limit='2020-12-31' if old['split']=='train' else '2023-12-31'
        if any(b['session']>limit for b in old['bars']):raise ValueError('source split cutoff violated')
        if old['split']!='train':continue
        symbol=old['symbol']
        if symbol in symbols:raise ValueError('unique training symbol required')
        symbols.add(symbol)
        window_coverage([b['session'] for b in old['bars']],steps)
        indices=[i for i,b in enumerate(old['bars']) if b['session']>=start]
        if len(indices)<=steps:raise ValueError('every symbol needs a full recent window')
        first=indices[0];new=result['episodes'][key]
        for field in ['bars','features','atr']:new[field]=new[field][first:]
        counts[symbol]=dict(source_bars=n,retained_bars=len(indices),first_session=new['bars'][0]['session'],last_session=new['bars'][-1]['session'])
    if not symbols:raise ValueError('training episodes required')
    validation={k:e for k,e in source['episodes'].items() if e['split']=='validation'}
    if not validation:raise ValueError('common development episodes required')
    result['manifest']['training_history']=dict(version=1,start=start,window_transitions=steps,
        source_dataset=digest(source),source_normalizer_sha256=digest(source['normalizer']),
        development_sha256=digest(validation),training_symbols=sorted(symbols),coverage=counts,
        rule='Only training row suffixes retained; source bars/features/ATR and development exact; no provider request or refit',
        preprocessing='Original full-training-only normalization held constant across arms; not a recent-only fit',
        warmup='Inherited causal source indicators may use earlier pre-cutoff training bars; cutoff limits sampled rows, not historical indicator context')
    return result


def validate_subset(source,recent,start='2016-01-01',steps=512):
    expected=recent_subset(source,start,steps)
    if digest(recent)!=digest(expected):raise ValueError('exact immutable training suffix and common development required')
    return recent
