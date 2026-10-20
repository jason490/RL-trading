"""Separate chronological training episodes with unchanged base observations."""
from copy import deepcopy
from dataclasses import asdict
from rl_stocks.features.registry.assemble import assemble,NAMES

ETFS=('SPY','QQQ','IWM','DIA')


def prepare_arms(base,older_bars,history_manifest):
    if base['names']!=list(NAMES) or any(e['split'] not in ('train','validation') for e in base['episodes'].values()):
        raise ValueError('baseline indicators and train/validation-only base required')
    train=[(key,e) for key,e in base['episodes'].items() if e['split']=='train']
    if len(train)!=26 or len({e['symbol'] for _,e in train})!=26:raise ValueError('one original training episode per declared symbol required')
    if any(not ('2010-01-01'<=b['session']<='2020-12-31') for _,e in train for b in e['bars']):raise ValueError('frozen recent training dates required')
    if {b.symbol for b in older_bars}!=set(ETFS) or any(not ('2001-01-01'<=b.session<'2010-01-01') for b in older_bars):
        raise ValueError('fixed earlier ETF-only history required')
    from rl_stocks.data.validation.bars import validate
    validate(older_bars)
    older={}
    for symbol in ETFS:
        bars=[b for b in older_bars if b.symbol==symbol];rows,names=assemble(bars)
        rows=rows[:-20]
        if names!=base['names'] or len(rows)<513 or any(b['bar']!=a['bar']+1 for a,b in zip(rows,rows[1:])):
            raise ValueError('complete independent warmup and fixed windows required')
        older[symbol]=dict(symbol=symbol,split='train',bars=[asdict(bars[r['bar']]) for r in rows],features=[r['values'] for r in rows],atr=[r['atr'] for r in rows])
    results={}
    for arm in ('original','recent_etf','older_etf'):
        payload=deepcopy(base)
        if arm!='original':
            for symbol in ETFS:
                source=next(e for _,e in train if e['symbol']==symbol) if arm=='recent_etf' else older[symbol]
                payload['episodes'][symbol+'-history-extra']=deepcopy(source)
        payload['manifest']['history_augmentation']=dict(version=1,arm=arm,source=history_manifest,
            normalizer='Unchanged original training-only parameters',start_sampling='full_window_uniform',
            expected_etf_episode_fraction=4/26 if arm=='original' else 8/30,
            extra_history_episode_fraction=0 if arm=='original' else 4/30,
            earlier_episode_policy='Separate feature warmup; discard final20 rows; no stitching into existing observations')
        results[arm]=payload
    return results
