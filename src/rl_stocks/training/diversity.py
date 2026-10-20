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
