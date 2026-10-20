from copy import deepcopy
import pytest
from rl_stocks.reporting.participation import episode_diagnosis,diagnose_groups


def fixture():
    return {'symbol':'SPY','split':'validation','rows':[{'session':'2021-01-04',
        'previous_equity':100000.,'equity':101000.,'quantity':0,'exposure':0.,'borrow_cost':5.}],
        'events':[{'time':'2021-01-04T14:30:00Z','quantity':-100,'delta':-100,'reason':'add','fee':10.,'impact':20.},
                  {'time':'2021-01-04T21:00:00Z','quantity':0,'delta':100,'reason':'stop','fee':10.,'impact':20.}],
        'metrics':{'net_return':.01,'costs':65.,'max_drawdown':0.,'turnover':2.},
        'baselines':{'buy_hold':{'rows':[{'session':'2021-01-04'}],
            'metrics':{'net_return':.02,'costs':30.}}},
        'directional':{'long_net_pnl':0.,'short_net_pnl':1000.,'long_costs':0.,'short_costs':65.,
            'long_active_sessions':0,'short_active_sessions':1}}


def test_saved_path_cost_identity_and_intraday_short_are_not_end_close_exposure():
    r=episode_diagnosis(fixture())
    assert r['net_excess_pp']==pytest.approx(r['saved_path_noncost_excess_pp']-r['incremental_cost_drag_pp'])
    assert r['incremental_cost_drag_pp']==pytest.approx(.035)
    assert r['short_return_contribution']==.01 and r['short_fraction']==0.
    assert r['short_touched_fraction']==1. and r['flat_fraction']==1.
    assert r['reported_costs']==65.
    stock=deepcopy(fixture());stock['symbol']='AAPL'
    report=diagnose_groups({'etf':fixture(),'stock':stock},{'AAPL'})
    assert report['groups']['stock']['count']==1
    assert report['groups']['etf']['total_fill_reasons']['stop']==1


def test_rejects_test_dates_unmatched_horizons_or_bad_accounting():
    for mutate in [lambda r:r.update(split='test'),lambda r:r['rows'][0].update(session='2024-01-02'),
        lambda r:r['directional'].update(short_costs=50.),lambda r:r['directional'].update(short_net_pnl=0.),
        lambda r:r['metrics'].update(net_return=.02),lambda r:r['baselines']['buy_hold']['rows'].append({'session':'2021-01-05'}),
        lambda r:r['events'][0].update(time='2021-01-05T14:30:00Z')]:
        r=fixture();mutate(r)
        with pytest.raises(ValueError):episode_diagnosis(r)


def test_small_resize_uses_pretrade_equity_excludes_protective_exits():
    r=fixture();r['events'].insert(1,{'time':'2021-01-04T15:00:00Z','quantity':-101,'delta':-1,
        'reason':'add','fee':.1,'impact':.2,'equity':100000.-.3,'reference_price':100.})
    r['events'][-1]['delta']=101
    r['metrics']['costs']+=.3;r['directional']['short_costs']+=.3
    d=episode_diagnosis(r)
    assert d['same_side_resizes_below_5pct_equity']==1
    assert d['saved_fees_impact_on_small_resizes_pp']==pytest.approx(.0003)
    assert d['discretionary_same_side_resize_fills']==1
