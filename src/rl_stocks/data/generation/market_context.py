"""Immutable compact-market payloads; original seven-field preprocessing stays frozen."""
from copy import deepcopy
from dataclasses import asdict
import math
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.features.registry.assemble import NAMES as BASE_NAMES
from rl_stocks.features.market.context import MarketContext,NAMES

SCHEMAS={mode:'daily-indicators-v1-market-'+mode+'-v1' for mode in ('masked','full')}


def market_inputs(base,context,symbol,session,decision_time,mode):
    if (mode not in SCHEMAS or len(base)!=7 or not all(math.isfinite(v) for v in base)):
        raise ValueError('finite baseline seven inputs and explicit market mode required')
    extra=[0.]*3
    if mode=='full':
        if not isinstance(context,MarketContext) or context.benchmark!='SPY':raise ValueError('causal SPY context required')
        extra=context.at(symbol,session,decision_time)
        if extra is None:raise ValueError('complete available same-session market context required')
    return list(base)+extra


def build_payload(source,context,mode):
    if (mode not in SCHEMAS or source['schema']!='daily-indicators-v1' or source['names']!=list(BASE_NAMES)
            or set(source['normalizer'])!={'names','means','scales'}
            or source['normalizer']['names']!=source['names']
            or source['manifest']['included_splits']!=['train','validation']
            or 'market_context' in source['manifest']):
        raise ValueError('original baseline train/development source required')
    if mode=='full' and (not isinstance(context,MarketContext) or context.benchmark!='SPY'):
        raise ValueError('causal SPY context required')
    Normalizer(tuple(source['names']),tuple(source['normalizer']['means']),tuple(source['normalizer']['scales']))
    raw={} if mode=='masked' else {(b.symbol,b.session):b for series in context.series.values() for b in series}
    result=deepcopy(source);extra_train=[];train_rows=0
    for key,e in result['episodes'].items():
        if (e['split'] not in ('train','validation') or len(e['bars'])!=len(e['features']) or len(e['bars'])!=len(e['atr'])):
            raise ValueError('aligned train/development source required')
        for i,(b,features) in enumerate(zip(e['bars'],e['features'],strict=True)):
            if (b['symbol']!=e['symbol'] or b['session']>('2020-12-31' if e['split']=='train' else '2023-12-31')
                    or e['split']=='validation' and b['session']<'2021-01-01'):
                raise ValueError('causal source split/symbol identity required')
            if mode=='full' and ((e['symbol'],b['session']) not in raw or asdict(raw[e['symbol'],b['session']])!=b):
                raise ValueError('exact raw/context/source bar identity required')
            row=market_inputs(features,context,e['symbol'],b['session'],b['available_time'],mode)
            e['features'][i]=row
            if e['split']=='train':
                train_rows+=1
                if mode=='full':extra_train.append(row[7:])
    if not train_rows:raise ValueError('training-only preprocessing rows required')
    fitted=Normalizer.fit(NAMES,extra_train) if mode=='full' else Normalizer(NAMES,(0.,)*3,(1.,)*3)
    result['schema']=SCHEMAS[mode];result['names']=source['names']+list(NAMES)
    result['normalizer']={key:source['normalizer'][key]+list(getattr(fitted,key)) for key in ('names','means','scales')}
    result['manifest']['market_context']=dict(version=1,mode=mode,benchmark='SPY',window=20,names=list(NAMES),
        source_dataset=digest(source),source_normalizer_sha256=digest(source['normalizer']),
        source_episodes_sha256=digest(source['episodes']),source_snapshot=source['manifest']['snapshot'],
        extra_normalizer_fit_rows=train_rows if mode=='full' else 0,
        preprocessing='Original seven statistics exact; full extra fields training-only fit, masked raw fields/means zero and scales one',
        availability='Exact21 same sessions; all contributing bars available by source decision; no known gaps/stale fill',
        preserved='All source bars/ATR/splits/original seven features and row keys exact; no test reconstructed')
    return result


def validate_payload_pair(source,payload,context,mode):
    if digest(payload)!=digest(build_payload(source,context,mode)):
        raise ValueError('exact causal market-context payload required')
    return payload


def validate_frozen_contract(manifest):
    contract=manifest['data_manifest'].get('market_context')
    if contract is None:
        if manifest['feature_schema'] in SCHEMAS.values():raise ValueError('market context contract missing')
        return None
    mode=contract.get('mode')
    if (mode not in SCHEMAS or type(contract.get('version')) is not int or contract['version']!=1
            or contract.get('benchmark')!='SPY' or type(contract.get('window')) is not int or contract['window']!=20
            or contract.get('names')!=list(NAMES) or manifest['feature_schema']!=SCHEMAS[mode]
            or manifest['names']!=list(BASE_NAMES)+list(NAMES) or manifest['normalizer']['names']!=manifest['names']):
        raise ValueError('market context bundle schema/availability contract mismatch')
    if digest({k:manifest['normalizer'][k][:7] for k in ('names','means','scales')})!=contract.get('source_normalizer_sha256'):
        raise ValueError('original market-context preprocessing lineage mismatch')
    if mode=='masked' and (manifest['normalizer']['means'][7:]!=[0.]*3 or manifest['normalizer']['scales'][7:]!=[1.]*3):
        raise ValueError('masked market context requires exact zero/identity preprocessing')
    return mode
