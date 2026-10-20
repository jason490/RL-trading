from copy import deepcopy
from dataclasses import asdict,replace
from datetime import date
import math
import pytest
from rl_stocks.core.contracts import Request
from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
from rl_stocks.training.supervised.targets import completed_targets,fit_preprocessing


def dataset():
    bars=SyntheticProvider().bars(Request(('SPY',),date(2016,1,1),date(2021,1,1)))
    return {'names':['fixture'],'normalizer':{'unused':'pooled through2020'},
        'manifest':{'included_splits':['train','validation'],'unverified_override':False},
        'episodes':{'train':{'symbol':'SPY','split':'train','bars':[asdict(b) for b in bars],
            'features':[[b.close] for b in bars]},'validation':{'split':'validation','arbitrary':'unread'}}}


def test_targets_complete_before_fold_end_and_ignore_project_validation():
    d=dataset();r=completed_targets(d)
    assert all(x['label_available_time'][:10]<='2017-12-31' for x in r if x['fold']=='fit')
    assert all('2018-01-01'<=x['decision_session']<=x['exit_session']<='2020-12-31' for x in r if x['fold']=='probe')
    assert not any(x['decision_session']<'2018-01-01'<x['exit_session'] for x in r)
    changed=deepcopy(d);changed['episodes']['validation']={'split':'validation','features':[[float('nan')]],'future_label':'irrelevant'}
    changed['normalizer']={'arbitrary':'different'}
    assert completed_targets(changed)==r
    first=r[0];bars=d['episodes']['train']['bars'];assert first['target_log_total_return']==pytest.approx(math.log(bars[20]['close']/bars[1]['open']))
    before=fit_preprocessing(r,d['names']);modified=deepcopy(r)
    for row in modified:
        if row['fold']=='probe':row['features']=[1e9];row['target_log_total_return']=100.
    assert fit_preprocessing(modified,d['names'])==before


def test_entry_day_actions_excluded_later_split_and_dividends_accounted():
    d=dataset();bars=d['episodes']['train']['bars'];entry=bars[1]['open'];exit=bars[20]['close']
    bars[1].update(dividend=100.,split=10.)
    bars[2].update(split=2.,dividend=1.)
    r=completed_targets(d)
    assert r[0]['target_log_total_return']==pytest.approx(math.log((2*exit+2)/entry))


def test_no_targets_cross_episode_or_availability_boundaries_and_reject_test():
    d=dataset();original=completed_targets(d);cut=300
    split=deepcopy(d);e=split['episodes'].pop('train');a=deepcopy(e);b=deepcopy(e)
    for field in ['bars','features']:a[field]=e[field][:cut];b[field]=e[field][cut:]
    split['episodes'].update(first=a,second=b);rows=completed_targets(split)
    assert len(rows)==len(original)-20
    for mutation in [lambda v:v['episodes'].update(test={'split':'test'}),
        lambda v:v['episodes']['train']['bars'][0].update(available_time=v['episodes']['train']['bars'][1]['open_time']),
        lambda v:v['episodes']['train']['bars'][2].update(split=0.),
        lambda v:v['manifest'].update(unverified_override=True)]:
        bad=deepcopy(d);mutation(bad)
        with pytest.raises(ValueError):completed_targets(bad)


def test_future_bars_outside_horizon_cannot_change_an_earlier_label():
    d=dataset();before=completed_targets(d)[0]
    changed=deepcopy(d)
    for b in changed['episodes']['train']['bars'][21:]:
        b['open']*=2;b['close']*=2
    after=completed_targets(changed)[0]
    assert after==before
    delayed=deepcopy(d);delayed['episodes']['train']['bars'][-1]['available_time']='2021-01-04T21:15:00+00:00'
    assert len(completed_targets(delayed))==len(completed_targets(d))-1


def test_rejects_non_utc_labels_and_purges_delayed_utc_labels():
    d=dataset();bars=d['episodes']['train']['bars']
    i=next(i for i,b in enumerate(bars) if b['session']=='2017-12-29')
    bars[i]['available_time']='2017-12-31T23:30:00-05:00' # Jan1 UTC, before next fixture open.
    with pytest.raises(ValueError,match='UTC'):completed_targets(d)
    bars[i]['available_time']='2018-01-01T04:30:00+00:00'
    rows=completed_targets(d)
    assert not any(r['fold']=='fit' and r['exit_session']=='2017-12-29' for r in rows)
    invalid=[deepcopy(next(r for r in rows if r['fold']=='fit'))]
    invalid[0]['label_available_time']='2017-12-31T23:30:00-05:00'
    with pytest.raises(ValueError):fit_preprocessing(invalid,d['names'])
