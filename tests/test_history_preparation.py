from copy import deepcopy
from dataclasses import asdict,replace
from datetime import date
import pytest
from rl_stocks.core.contracts import Request
from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
from rl_stocks.data.generation.history import prepare_arms,ETFS
from rl_stocks.features.registry.assemble import NAMES


@pytest.fixture
def inputs():
    episode=dict(split='train',bars=[dict(session='2016-01-01')],features=[[0.]*7],atr=[1.])
    episodes={}
    for symbol in ETFS+tuple('STOCK'+str(i) for i in range(22)):
        episodes[symbol]=dict(symbol=symbol,**deepcopy(episode))
    episodes['validation']=dict(symbol='SPY',split='validation',bars=[],features=[],atr=[])
    base=dict(names=list(NAMES),schema='daily-indicators-v1',normalizer=dict(names=list(NAMES),means=[0.]*7,scales=[1.]*7),episodes=episodes,manifest={})
    bars=[replace(b,provider='yahoo',feed='yahoo',adjustments='split_adjusted_with_dividends') for b in SyntheticProvider().bars(Request(ETFS,date(2001,1,1),date(2004,1,1)))]
    return base,bars


def test_three_arms_preserve_base_and_normalizer_and_isolate_old_warmup(inputs):
    base,bars=inputs;before=deepcopy(base);out=prepare_arms(base,bars,{})
    assert base==before
    assert [len(out[k]['episodes']) for k in out]==[27,31,31]
    for d in out.values():
        assert d['normalizer']==base['normalizer']
        assert all(d['episodes'][k]==v for k,v in base['episodes'].items())
    for symbol in ETFS:
        assert out['recent_etf']['episodes'][symbol+'-history-extra']==base['episodes'][symbol]
        old=out['older_etf']['episodes'][symbol+'-history-extra']
        raw=[b for b in bars if b.symbol==symbol]
        assert old['bars'][-1]==asdict(raw[-21])
        assert old['bars'][0]['session']>raw[0].session
        assert len(old['bars'])==len(old['features'])==len(old['atr'])
    assert out['recent_etf']['manifest']['history_augmentation']['expected_etf_episode_fraction']==out['older_etf']['manifest']['history_augmentation']['expected_etf_episode_fraction']


def test_future_base_or_old_dates_rejected(inputs):
    base,bars=inputs;bad=deepcopy(base);bad['episodes']['SPY']['bars'][0]['session']='2021-01-01'
    with pytest.raises(ValueError):prepare_arms(bad,bars,{})
    with pytest.raises(ValueError):prepare_arms(base,[replace(b,session='2011-01-01') for b in bars],{})
    bad=deepcopy(base);bad['episodes']['validation']['split']='test'
    with pytest.raises(ValueError):prepare_arms(bad,bars,{})
