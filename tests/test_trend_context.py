from dataclasses import replace
from datetime import date
import math
import pytest
from rl_stocks.features.trend.context import trailing_context, NAMES
from rl_stocks.features.registry.assemble import assemble
from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
from rl_stocks.data.generation.datasets import generate, read_dataset, normalizer_from
from rl_stocks.data.storage.immutable import put
from rl_stocks.core.contracts import Request


def test_trailing_formulas_prefix_invariance_scale_and_rolling_high_expiry():
    prices=[100.+i*.1 for i in range(320)]
    full=trailing_context(prices)
    assert full[:252]==[None]*252
    assert full[252]==pytest.approx([math.log(prices[252]/prices[189]),math.log(prices[252]/prices[0]),0.])
    assert trailing_context(prices[:280])==full[:280]
    assert trailing_context(prices[:280]+[1.]*40)[:280]==full[:280]
    scaled=trailing_context([p*3 for p in prices]);assert scaled[252]==pytest.approx(full[252])
    falling=[1000.]+[100.]*252+[50.]*3
    result=trailing_context(falling)
    assert result[252][2]==0. # Old peak has left trailing252, despite annual negative return.
    assert result[253][2]==-.5
    for bad in [0.,-1.,float('nan'),float('inf')]:
        with pytest.raises(ValueError):trailing_context([100.,bad])


def test_assemble_matched_readiness_mask_and_forward_split_continuity():
    bars=SyntheticProvider().bars(Request(('SPY',),date(2018,1,1),date(2020,1,1)))
    legacy,names=assemble(bars);assert assemble(bars,trend_context='none')==(legacy,names)
    profiles={mode:assemble(bars,trend_context=mode) for mode in ['matched_baseline','masked','full']}
    base=profiles['matched_baseline'][0]
    for mode,(rows,fields) in profiles.items():
        assert [r['bar'] for r in rows]==list(range(252,len(bars)))
        assert [r['values'][:7] for r in rows]==[r['values'] for r in base]
        assert [r['available_time'] for r in rows]==[bars[r['bar']].available_time for r in rows]
    assert profiles['full'][1][-3:]==list(NAMES)
    assert all(r['values'][-3:]==[0.,0.,0.] for r in profiles['masked'][0])
    prefix,_=assemble(bars[:300],trend_context='full');assert prefix==[r for r in profiles['full'][0] if r['bar']<300]
    split=[replace(b,open=b.open/(2 if i>=270 else 1),high=b.high/(2 if i>=270 else 1),
        low=b.low/(2 if i>=270 else 1),close=b.close/(2 if i>=270 else 1),split=2. if i==270 else 1.) for i,b in enumerate(bars)]
    adjusted,_=assemble(split,trend_context='full')
    assert [r['values'][-3:] for r in adjusted]==[r['values'][-3:] for r in profiles['full'][0]]
    with pytest.raises(ValueError):assemble(bars,trend_context='future')


def test_dataset_profiles_keep_panels_train_only_fit_and_reset_after_gap(tmp_path):
    bars=SyntheticProvider().bars(Request(('SPY',),date(2018,1,1),date(2024,1,1)))
    provenance={'coverage':{'SPY':{'calendar_verified':True,'missing':[]}},'dividends_verified':True}
    universe={'instruments':[{'symbol':'SPY','role':'tradable'}]}
    snap=put(tmp_path/'raw',bars,provenance)
    options=dict(universe=universe,include_test=False)
    data={mode:read_dataset(generate(snap,tmp_path/'data',trend_context=mode,**options)) for mode in ['matched_baseline','masked','full']}
    for mode,d in data.items():
        assert d['manifest']['trend_context']['mode']==mode
        assert {e['split'] for e in d['episodes'].values()}=={'train','validation'}
        assert d['normalizer']['means'][:7]==data['full']['normalizer']['means'][:7]
        assert d['normalizer']['scales'][:7]==data['full']['normalizer']['scales'][:7]
        for key,e in d['episodes'].items():assert e['bars']==data['full']['episodes'][key]['bars']
    normalizer=normalizer_from(data['masked']['normalizer'])
    assert all(normalizer.transform(r,data['masked']['names'])[-3:]==[0.,0.,0.] for e in data['masked']['episodes'].values() for r in e['features'])
    changed=[replace(b,open=b.open*2,high=b.high*2,low=b.low*2,close=b.close*2) if b.session>'2020-12-31' else b for b in bars]
    other=put(tmp_path/'raw',changed,provenance)
    altered=read_dataset(generate(other,tmp_path/'data',trend_context='full',**options))
    assert altered['normalizer']==data['full']['normalizer']
    assert altered['episodes']['SPY-train-0']==data['full']['episodes']['SPY-train-0']
    missing=bars[500];gapbars=bars[:500]+bars[501:]
    gap=put(tmp_path/'raw',gapbars,{'coverage':{'SPY':{'calendar_verified':True,'missing':[missing.session]}},'dividends_verified':True})
    result=read_dataset(generate(gap,tmp_path/'data',trend_context='full',allow_unverified=True,**options))
    sessions={b['session'] for e in result['episodes'].values() for b in e['bars']}
    assert not sessions.intersection(b.session for b in gapbars[500:752])
