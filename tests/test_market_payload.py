from copy import deepcopy
from dataclasses import asdict,replace
import json
import numpy as np
import pytest
from test_market_context import series
from rl_stocks.features.registry.assemble import NAMES as BASE_NAMES
from rl_stocks.features.market.context import MarketContext,NAMES
from rl_stocks.data.generation.datasets import Normalizer,normalizer_from
from rl_stocks.data.generation.market_context import build_payload,validate_payload_pair,market_inputs,validate_frozen_contract
from rl_stocks.core.contracts import digest


def fixture():
    spy=series('SPY');own=series('AAPL',.02)
    bars=spy+own;context=MarketContext(bars,{'SPY':[],'AAPL':[]})
    source=dict(schema='daily-indicators-v1',names=list(BASE_NAMES),normalizer=dict(names=list(BASE_NAMES),means=[1.]*7,scales=[2.]*7),
        manifest=dict(included_splits=['train','validation'],snapshot='fixture',provenance={'capabilities':{'feed':'fixture'}}),episodes={})
    for symbol,rows in [('SPY',spy),('AAPL',own)]:
        chosen=rows[20:24]
        source['episodes'][symbol+'-train']=dict(symbol=symbol,split='train',bars=[asdict(b) for b in chosen],atr=[2.]*4,features=[[float(i)]*7 for i in range(4)])
    return source,context


def test_exact_source_rows_and_first_seven_statistics_preserved_and_masked_builder():
    source,ctx=fixture();before=deepcopy(source)
    for mode in ['full','masked']:
        payload=build_payload(source,ctx,mode);assert validate_payload_pair(source,payload,ctx,mode)==payload
        assert payload['names']==list(BASE_NAMES)+list(NAMES)
        for key,e in payload['episodes'].items():
            assert e['bars']==source['episodes'][key]['bars'] and e['atr']==source['episodes'][key]['atr']
            assert [r[:7] for r in e['features']]==source['episodes'][key]['features']
            if mode=='masked':assert all(r[7:]==[0.]*3 for r in e['features'])
        for k in ['names','means','scales']:assert payload['normalizer'][k][:7]==source['normalizer'][k]
        if mode=='masked':assert payload['normalizer']['means'][7:]==[0.]*3 and payload['normalizer']['scales'][7:]==[1.]*3
        assert payload['manifest']['market_context']['extra_normalizer_fit_rows']==(8 if mode=='full' else 0)
        payload['episodes']['SPY-train']['bars'][0]['close']=123.
        assert source==before
    assert market_inputs([1.]*7,None,'SPY','2020-01-01','2020-01-01T22:00:00+00:00','masked')==[1.]*7+[0.]*3
    with pytest.raises(ValueError):build_payload(source,None,'full')


def test_validation_values_do_not_enter_extra_preprocessing():
    source,ctx=fixture();full=build_payload(source,ctx,'full')
    altered=deepcopy(source);train=altered['episodes']['AAPL-train'];extra=deepcopy(train);extra['split']='validation'
    extra['bars']=[dict(b,session=b['session'].replace('2020','2021'),open_time=b['open_time'].replace('2020','2021'),event_time=b['event_time'].replace('2020','2021'),available_time=b['available_time'].replace('2020','2021')) for b in extra['bars']]
    altered['episodes']['AAPL-validation']=extra
    # Independent validation bars change market context but cannot enter training fit.
    raw=[]
    for rows in ctx.series.values():raw+=rows
    for symbol,rate in [('SPY',.05),('AAPL',-.02)]:
        raw+=[replace(b,session=b.session.replace('2020','2021'),open_time=b.open_time.replace('2020','2021'),event_time=b.event_time.replace('2020','2021'),available_time=b.available_time.replace('2020','2021')) for b in series(symbol,rate)]
    raw=sorted(raw,key=lambda b:(b.symbol,b.session));ctx2=MarketContext(raw,{'SPY':[],'AAPL':[]})
    # Align validation raw/source prices exactly, leaving its context intentionally distinct.
    lookup={(b.symbol,b.session):b for b in raw}
    extra['bars']=[asdict(lookup['AAPL',b['session']]) for b in extra['bars']]
    rebuilt=build_payload(altered,ctx2,'full')
    assert rebuilt['normalizer']==full['normalizer']


@pytest.mark.parametrize('kind',['test','future','bars','features','mask'])
def test_unsafe_source_or_payload_rejects(kind):
    source,ctx=fixture()
    if kind=='test':source['episodes']['SPY-train']['split']='test'
    if kind=='future':source['episodes']['SPY-train']['bars'][0]['session']='2024-01-01'
    if kind in ('test','future'):
        with pytest.raises(ValueError):build_payload(source,ctx,'full')
    else:
        payload=build_payload(source,ctx,'masked')
        if kind=='bars':payload['episodes']['SPY-train']['bars'][0]['close']=999.
        if kind=='features':payload['episodes']['SPY-train']['features'][0][0]=999.
        if kind=='mask':payload['episodes']['SPY-train']['features'][0][7]=1.
        with pytest.raises(ValueError):validate_payload_pair(source,payload,ctx,'masked')


@pytest.mark.parametrize('mode',['masked','full'])
def test_actual_frozen_mask_requires_zero_channels_and_preserves_recurrent_prediction(tmp_path,mode):
    import torch
    from sb3_contrib import RecurrentPPO
    from rl_stocks.environment.trading.env import TradingEnv
    from rl_stocks.environment.observations.state import observation_spec
    from rl_stocks.inference.bundle import freeze,FrozenPolicy
    source,ctx=fixture();data=build_payload(source,ctx,mode);torch.set_num_threads(1)
    env=TradingEnv(list(data['episodes'].values()),normalizer_from(data['normalizer']),data['names'],observation_profile='execution_masked')
    model=RecurrentPPO('MlpLstmPolicy',env,n_steps=8,batch_size=8,seed=11,policy_kwargs=dict(lstm_hidden_size=4,net_arch=[4]))
    model.save(tmp_path/'untrained');bundle=freeze(tmp_path/'untrained.zip',data,dict(algorithm='recurrent_ppo',observation=observation_spec('execution_masked')),tmp_path/'frozen')
    frozen=FrozenPolicy(bundle,data['schema'],'fixture');obs,_=env.reset(seed=11)
    assert obs.shape==(24,)
    if mode=='masked':assert np.all(obs[7:10]==0)
    initial=obs.copy();state=fstate=None;normal_before=digest(data['normalizer'])
    for i in range(2):
        expected,state=model.predict(obs,state=state,episode_start=np.array([i==0]),deterministic=True)
        previous=None if fstate is None else tuple(s.copy() for s in fstate)
        caller=fstate
        actual,fstate=frozen.predict_recurrent(obs,state=fstate,episode_start=i==0);np.testing.assert_array_equal(actual,expected)
        for a,b in zip(state,fstate,strict=True):np.testing.assert_array_equal(a,b)
        if previous is not None:
            for a,b in zip(previous,caller,strict=True):np.testing.assert_array_equal(a,b)
        if i==0:first_action=actual.copy()
        obs,*_=env.step(np.zeros(4))
    reset,_=frozen.predict_recurrent(initial,state=fstate,episode_start=True);np.testing.assert_array_equal(reset,first_action)
    assert digest(data['normalizer'])==normal_before and digest(asdict(frozen.normalizer))==normal_before
    assert model.num_timesteps==0 and not model.policy.optimizer.state
    bad=obs.copy();bad[8]=1.
    if mode=='masked':
        with pytest.raises(ValueError,match='must remain zero'):frozen.predict_recurrent(bad,state=None,episode_start=True)
    manifest=json.loads((bundle/'manifest.json').read_text());manifest['data_manifest']['market_context']['mode']='full' if mode=='masked' else 'masked'
    with pytest.raises(ValueError):validate_frozen_contract(manifest)
