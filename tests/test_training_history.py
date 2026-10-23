from copy import deepcopy
from datetime import date,timedelta
import pytest
from rl_stocks.data.generation.training_history import recent_subset,validate_subset


def source():
    bars=[{'session':str(date(2015,12,20)+timedelta(days=i))} for i in range(540)]
    train=dict(symbol='SPY',split='train',bars=bars,features=[[i] for i in range(540)],atr=list(range(540)))
    dev=dict(symbol='SPY',split='validation',bars=[{'session':'2021-01-04'}],features=[[5]],atr=[2])
    return dict(manifest={'included_splits':['train','validation']},normalizer={'names':['x'],'means':[3],'scales':[2]},
        names=['x'],schema='fixture',episodes={'train':train,'dev':dev})


def test_exact_suffix_frozen_normalizer_and_development_no_mutation():
    s=source();old=deepcopy(s);r=recent_subset(s)
    assert s==old and r['episodes']['dev']==s['episodes']['dev'] and r['normalizer']==s['normalizer']
    for key in ['bars','features','atr']:assert r['episodes']['train'][key]==s['episodes']['train'][key][12:]
    assert validate_subset(s,r)==r
    r['episodes']['train']['features'][0][0]=-1;assert s==old


@pytest.mark.parametrize('field',['features','atr','bars','normalizer','development'])
def test_changed_suffix_or_preprocessing_rejected(field):
    s=source();r=recent_subset(s)
    if field=='normalizer':r['normalizer']['means'][0]=99
    elif field=='development':r['episodes']['dev']['features'][0][0]=99
    elif field=='bars':r['episodes']['train']['bars'][0]['session']='2016-01-02'
    else:r['episodes']['train'][field][0]=99
    with pytest.raises(ValueError,match='exact immutable'):validate_subset(s,r)


@pytest.mark.parametrize('corruption',['test','future','short','alignment','order','duplicate'])
def test_unsafe_sources_reject(corruption):
    s=source();e=s['episodes']['train']
    if corruption=='test':s['episodes']['dev']['split']='test'
    if corruption=='future':e['bars'][-1]['session']='2021-01-01'
    if corruption=='short':
        for k in ['bars','features','atr']:e[k]=e[k][:520]
    if corruption=='alignment':e['atr'].pop()
    if corruption=='order':e['bars'][1]['session']=e['bars'][0]['session']
    if corruption=='duplicate':s['episodes']['extra']=deepcopy(e)
    with pytest.raises(ValueError):recent_subset(s)
