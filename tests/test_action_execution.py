from copy import deepcopy
import pytest
from test_participation import fixture
from rl_stocks.reporting.action_execution import diagnose_episode


def sample():
    r=fixture();r['rows'][0].update(raw_action=[-.5,.5,.5,.5],approved_target_quantity=-100)
    return r


def test_intraday_protection_is_not_a_flat_policy_request():
    x=diagnose_episode(sample());c=x['categories']
    assert c['flat_after_protection']['fraction']==1
    assert c['flat_after_protection']['observed_return_contribution']==pytest.approx(.01)
    assert c['flat_request']['fraction']==0
    assert sum(v['fraction'] for v in c.values())==1


def test_rejection_priority_and_zero_request_do_not_relabel_exits():
    r=sample();r['rows'][0]['rejected']='protective_fill_preempts_action'
    assert diagnose_episode(r)['categories']['flat_rejected']['fraction']==1
    r['rows'][0]['raw_action'][0]=0
    assert diagnose_episode(r)['categories']['flat_request']['fraction']==1


def test_invalid_action_and_broken_equity_chain_reject():
    for mutate in [lambda r:r['rows'][0]['raw_action'].__setitem__(0,float('nan')),
                   lambda r:r['rows'][0]['raw_action'].__setitem__(1,2.),
                   lambda r:r['rows'][0].update(previous_equity=99999.)]:
        r=deepcopy(sample());mutate(r)
        with pytest.raises(ValueError):diagnose_episode(r)


def test_worst_drawdown_window_reconciles_peak_trough_and_costs():
    from rl_stocks.reporting.action_execution import worst_drawdown_window
    r={'rows':[dict(session='2021-01-04',equity=110000.,exposure=.8,quantity=100),
               dict(session='2021-01-05',equity=99000.,exposure=-.7,quantity=-100)],
       'events':[dict(time='2021-01-04T14:30:00Z',fee=50.,impact=50.),
                 dict(time='2021-01-05T14:30:00Z',fee=50.,impact=50.)],
       'metrics':{'max_drawdown':-.1}}
    x=worst_drawdown_window(r)
    assert x['peak_session']=='2021-01-04' and x['trough_session']=='2021-01-05'
    assert x['cost_fraction_of_peak']==pytest.approx(100/110000)
    assert x['noncost_loss_fraction']==pytest.approx(-.1+100/110000)
    assert x['end_short_fraction']==1
    r['metrics']['max_drawdown']=-.2
    with pytest.raises(ValueError):worst_drawdown_window(r)


def test_original_policy_magnitude_is_distinct_from_issued_unit_request():
    r=sample();r['rows'][0].update(raw_action=[-1.,.5,.5,.5],policy_action=[-.2,.5,.5,.5])
    x=diagnose_episode(r)
    assert x['mean_policy_short_output']==.2 and x['mean_requested_short_exposure']==1
    assert x['policy_zero_fraction']==0
