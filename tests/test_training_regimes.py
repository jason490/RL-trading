import pytest
from rl_stocks.reporting.training_regimes import summarize_training_regimes


def fixture():
    d = {'x':{'split':'train','rows':[
        {'session':f'2018-01-{i+2:02}', 'decision':{'reference':p,
         'decision_time':f'2018-01-{i+1:02}T21:00:00', 'approved':[-.5,0,0,0]}}
        for i,p in enumerate([100,80,60,90,100])]}}
    for i,r in enumerate(d['x']['rows']):r['price']=[80,60,90,100,110][i]
    return d


def test_reporting_cohorts():
    s=summarize_training_regimes(fixture(),horizon=1)
    assert s['all']['count']==3
    assert s['past_decline']['count']==2
    assert s['past_decline_future_decline']['count']==1
    assert s['past_decline_future_rebound']['count']==1
    assert s['past_decline']['short_request_fraction']==1


def test_reject_validation_and_bad_timing():
    d=fixture();d['x']['split']='validation'
    with pytest.raises(ValueError):summarize_training_regimes(d,1)
    d=fixture();d['x']['rows'][1]['decision']['decision_time']='2018-01-03'
    with pytest.raises(ValueError):summarize_training_regimes(d,1)


def test_rejected_decisions_excluded_without_breaking_price_horizon():
    d=fixture();d['x']['rows'][1]['decision']=None
    s=summarize_training_regimes(d,1)
    assert s['all']['count']==2
    assert s['past_decline']['count']==1
    assert s['past_decline_future_rebound']['count']==1
