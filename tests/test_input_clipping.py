from copy import deepcopy
from dataclasses import asdict,FrozenInstanceError
import pytest
from rl_stocks.data.generation.datasets import Normalizer,normalizer_from
from rl_stocks.data.generation.clipping import ClippedNormalizer,clipped_dataset,validate_clipped_dataset
from rl_stocks.features.registry.assemble import NAMES


def source():
    return dict(schema='daily-indicators-v1',names=list(NAMES),normalizer=asdict(Normalizer(NAMES,(0.,)*7,(1.,)*7)),
        manifest={'included_splits':['train','validation']},episodes={k:dict(symbol='SPY',split=k,bars=[dict(session=day)],features=[[20.,-10.,2.,3.,4.,5.,6.]],atr=[7.]) for k,day in [('train','2020-01-02'),('validation','2021-01-04')]})


def test_default_serialization_and_transforms_exact_new_clipping_opt_in_immutable():
    base=Normalizer(('x',),(1.,),(2.,));value=asdict(base);assert set(value)=={'names','means','scales'}
    assert type(normalizer_from(value)) is Normalizer and base.transform([21.],['x'])==[10.]
    clipped=normalizer_from(dict(value,clip_abs=5.));assert isinstance(clipped,ClippedNormalizer)
    assert clipped.transform([21.],['x'])==[5.] and clipped.transform([-21.],['x'])==[-5.]
    assert clipped.transform([3.],['x'])==[1.] and asdict(clipped)==dict(value,clip_abs=5.)
    with pytest.raises(FrozenInstanceError):clipped.clip_abs=3
    with pytest.raises(ValueError):ClippedNormalizer.fit(['x'],[[1.]])


@pytest.mark.parametrize('cap',[0,-1,True,float('nan'),float('inf'),None,'5'])
def test_invalid_clipping_bounds_rejected(cap):
    with pytest.raises(ValueError):normalizer_from(dict(names=['x'],means=[0.],scales=[1.],clip_abs=cap))


def test_unknown_normalizer_fields_not_silently_ignored():
    with pytest.raises(ValueError):normalizer_from(dict(names=['x'],means=[0.],scales=[1.],clip=5.))


def test_dataset_changes_only_versioned_transform_and_preserves_raw_sources():
    original=source();before=deepcopy(original);candidate=clipped_dataset(original)
    assert original==before and candidate['episodes']==original['episodes']
    assert candidate['normalizer']==dict(original['normalizer'],clip_abs=5.)
    assert candidate['schema']!=original['schema'] and validate_clipped_dataset(original,candidate)==candidate
    candidate['episodes']['train']['atr'][0]=999;assert original==before
    with pytest.raises(ValueError):validate_clipped_dataset(original,candidate)


@pytest.mark.parametrize('bad',['test','future','already','nan'])
def test_unsafe_dataset_sources_reject(bad):
    value=source()
    if bad=='test':value['episodes']['validation']['split']='test'
    if bad=='future':value['episodes']['train']['bars'][0]['session']='2024-01-02'
    if bad=='already':value['normalizer']['clip_abs']=5.
    if bad=='nan':value['episodes']['train']['features'][0][0]=float('nan')
    with pytest.raises(ValueError):clipped_dataset(value)


def test_clipping_changes_market_observation_not_fixed_action_accounting():
    import numpy as np
    from test_exposure_replay import fixture
    from rl_stocks.environment.trading.env import TradingEnv
    data=fixture()
    for e in data['episodes'].values():
        for row in e['features']:row[0]=100.
    def env(clip):
        n=deepcopy(data['normalizer'])
        if clip:n['clip_abs']=5.
        return TradingEnv(list(data['episodes'].values()),normalizer_from(n),data['names'])
    baseline,candidate=env(False),env(True);a,_=baseline.reset(seed=11);b,_=candidate.reset(seed=11)
    assert a[0]==100 and b[0]==5;np.testing.assert_array_equal(a[1:],b[1:])
    for _ in range(3):
        a=baseline.step([.8,1,1,1]);b=candidate.step([.8,1,1,1])
        assert a[1:4]==b[1:4]
        for key in ['equity','costs','quantity']:assert a[4][key]==b[4][key]
    assert baseline.sim.portfolio.events==candidate.sim.portfolio.events


def test_actual_untrained_frozen_recurrent_bundle_preserves_clipping_and_prediction(tmp_path):
    import numpy as np,torch
    from sb3_contrib import RecurrentPPO
    from test_exposure_replay import fixture
    from rl_stocks.environment.trading.env import TradingEnv
    from rl_stocks.inference.bundle import freeze,FrozenPolicy
    torch.set_num_threads(1);data=fixture();data['normalizer']['clip_abs']=5.
    for e in data['episodes'].values():
        for row in e['features']:row[0]=100.
    env=TradingEnv(list(data['episodes'].values()),normalizer_from(data['normalizer']),data['names'])
    obs,_=env.reset(seed=11)
    model=RecurrentPPO('MlpLstmPolicy',env,n_steps=8,batch_size=8,seed=11,device='cpu',policy_kwargs=dict(lstm_hidden_size=4,net_arch=[4]))
    model.save(tmp_path/'untrained')
    payload=dict(schema='fixture-zclip5-v1',names=data['names'],normalizer=asdict(env.normalizer),manifest={'provenance':{'capabilities':{'feed':'fixture'}}})
    bundle=freeze(tmp_path/'untrained.zip',payload,dict(algorithm='recurrent_ppo'),tmp_path/'frozen')
    frozen=FrozenPolicy(bundle,payload['schema'],'fixture');before=asdict(frozen.normalizer)
    assert frozen.normalizer.transform([100.],data['names'])==[5.]
    expected,expected_state=model.predict(obs,state=None,episode_start=np.array([True]),deterministic=True)
    actual,state=frozen.predict_recurrent(obs,state=None,episode_start=True)
    np.testing.assert_array_equal(actual,expected)
    for x,y in zip(state,expected_state):np.testing.assert_array_equal(x,y)
    assert asdict(frozen.normalizer)==before
    with pytest.raises(ValueError):FrozenPolicy(bundle,'fixture','fixture')
