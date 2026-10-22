from dataclasses import replace
from datetime import date
import math
import pytest
from rl_stocks.features.price.sessions import session_context,NAMES
from rl_stocks.features.registry.assemble import assemble
from rl_stocks.core.contracts import Request
from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
from rl_stocks.data.storage.immutable import put
from rl_stocks.data.generation.datasets import generate,read_dataset,normalizer_from
from rl_stocks.data.generation.session_context import preserve_preprocessing


def bars(end=date(2020,1,1)):
    return SyntheticProvider().bars(Request(('SPY',),date(2018,1,1),end))


def test_exact_formula_ema_seed_prefix_and_scale():
    b=bars();v=session_context(b)
    assert v[:20]==[None]*20
    g=d=None
    for i in range(1,21):
        gap=math.log(b[i].open/b[i-1].close);day=math.log(b[i].close/b[i].open)
        g=gap if g is None else 2/21*gap+19/21*g
        d=day if d is None else 2/21*day+19/21*d
    assert v[20]==pytest.approx([gap,day,g,d])
    assert session_context(b[:80])==v[:80]
    other=[replace(x,open=x.open*3,close=x.close*3) for x in b]
    for x,y in zip(v[20:],session_context(other)[20:]):assert x==pytest.approx(y)
    for bad in [0.,-1.,float('nan')]:
        with pytest.raises(ValueError):session_context([replace(b[0],open=bad)])


def test_default_identity_mask_availability_split_and_information_witness():
    b=bars();default,names=assemble(b);assert assemble(b,session_mode='none')==(default,names)
    full,fields=assemble(b,session_mode='full');masked,mfields=assemble(b,session_mode='masked')
    assert fields==mfields and fields[-4:]==list(NAMES)
    assert [r['bar'] for r in default]==[r['bar'] for r in full]
    assert all(r['values'][:7]==d['values'] and r['available_time']==b[r['bar']].available_time for r,d in zip(full,default))
    assert all(r['values'][-4:]==[0.]*4 for r in masked)
    changed=list(b);i=100;changed[i]=replace(b[i],open=(b[i].high+b[i].low)/2)
    assert changed[i].open!=b[i].open
    assert assemble(changed)==(default,names)  # Baseline never observes this open change.
    assert assemble(changed,session_mode='full')[0][i-default[0]['bar']]['values'][-4:]!=full[i-default[0]['bar']]['values'][-4:]
    assert assemble(b[:120],session_mode='full')[0]==[r for r in full if r['bar']<120]
    split=[replace(x,open=x.open/(2 if j>=100 else 1),high=x.high/(2 if j>=100 else 1),low=x.low/(2 if j>=100 else 1),close=x.close/(2 if j>=100 else 1),split=2. if j==100 else 1.) for j,x in enumerate(b)]
    adjusted,split_names=assemble(split,session_mode='full')
    assert split_names==fields
    assert [r['values'][-4:] for r in adjusted]==[r['values'][-4:] for r in full]


def test_dataset_shared_panels_training_only_fit_masks_and_gap_reset(tmp_path):
    b=bars(date(2024,1,1));provenance={'coverage':{'SPY':{'calendar_verified':True,'missing':[]}},'dividends_verified':True}
    universe={'instruments':[{'symbol':'SPY','role':'tradable'}]};snap=put(tmp_path/'raw',b,provenance)
    options=dict(universe=universe,include_test=False)
    data={mode:read_dataset(generate(snap,tmp_path/'data',session_mode=mode,**options)) for mode in ['none','masked','full']}
    for mode,d in data.items():
        assert {e['split'] for e in d['episodes'].values()}=={'train','validation'}
        assert d['normalizer']['means'][:7]==data['none']['normalizer']['means']
        for k,e in d['episodes'].items():assert e['bars']==data['none']['episodes'][k]['bars']
    norm=normalizer_from(data['masked']['normalizer'])
    assert all(norm.transform(r,data['masked']['names'])[-4:]==[0.]*4 for e in data['masked']['episodes'].values() for r in e['features'])
    from copy import deepcopy
    frozen=deepcopy(data['none']);frozen['normalizer']['means'][0]+=1.
    prepared=preserve_preprocessing(frozen,data['full'],'full')
    assert prepared['normalizer']['means'][:7]==frozen['normalizer']['means']
    assert prepared['normalizer']['means'][-4:]==data['full']['normalizer']['means'][-4:]
    for corruption in ['panel','normalizer','features','split','mask']:
        candidate=deepcopy(data['masked'] if corruption=='mask' else data['full'])
        key=next(iter(candidate['episodes']));episode=candidate['episodes'][key]
        if corruption=='panel':episode['bars'][0]['close']+=1
        if corruption=='normalizer':candidate['normalizer']['means'][-1]+=1
        if corruption=='features':episode['features'][0][0]+=1
        if corruption=='split':episode['split']='test'
        if corruption=='mask':episode['features'][0][-1]=1.
        with pytest.raises(ValueError):preserve_preprocessing(frozen,candidate,'masked' if corruption=='mask' else 'full')
    future=[replace(x,open=x.open*2,high=x.high*2,low=x.low*2,close=x.close*2) if x.session>'2020-12-31' else x for x in b]
    altered=read_dataset(generate(put(tmp_path/'raw',future,provenance),tmp_path/'data',session_mode='full',**options))
    assert altered['normalizer']==data['full']['normalizer']
    missing=b[300].session;gap_b=[x for x in b if x.session!=missing]
    gap_prov={'coverage':{'SPY':{'calendar_verified':True,'missing':[missing]}},'dividends_verified':True}
    gap=read_dataset(generate(put(tmp_path/'raw',gap_b,gap_prov),tmp_path/'data',session_mode='full',allow_unverified=True,**options))
    observed={x['session'] for e in gap['episodes'].values() for x in e['bars']}
    assert not observed.intersection(x.session for x in b[301:321])
    assert 'session_context' not in data['none']['manifest']
    with pytest.raises(ValueError):generate(snap,tmp_path/'data',session_mode='future',**options)
