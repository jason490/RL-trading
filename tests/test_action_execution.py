from copy import deepcopy
import pytest
from test_participation import fixture
from rl_stocks.reporting.action_execution import diagnose_episode


def sample():
    r=fixture();r['rows'][0].update(raw_action=[-.5,.5,.5,.5],approved_target_quantity=-100)
    r['baselines']['buy_hold']['rows'][0]['equity']=102000.
    return r


def test_exit_partition_reconciles_daily_dollar_excess_and_keeps_overlap():
    from rl_stocks.reporting.action_execution import exit_partition
    r=sample();r['events'][-1].update(reason='target_near',quantity=-50,delta=50)
    r['events'].append(dict(time='2021-01-04T21:00:00Z',reason='holding_limit',
        quantity=0,delta=50,fee=0.,impact=0.))
    x=exit_partition(r)
    assert x['flat/both']['sessions']==1
    assert x['flat/both']['policy_pp']==pytest.approx(1.)
    assert x['flat/both']['holding_pp']==pytest.approx(2.)
    assert sum(v['excess_pp'] for v in x.values())==pytest.approx(-1.)
    assert sum(v['sessions'] for v in x.values())==1


def test_exit_partition_uses_prior_quantity_not_request_or_closing_position():
    from rl_stocks.reporting.action_execution import exit_partition
    r=sample();r['rows']=[dict(session='2021-01-04',previous_equity=100000.,equity=101000.,
        quantity=100,exposure=.1,borrow_cost=0.),dict(session='2021-01-05',
        previous_equity=101000.,equity=103000.,quantity=0,exposure=0.,borrow_cost=0.)]
    r['events']=[dict(time='2021-01-04T14:30:00Z',reason='add',quantity=100,delta=100,fee=0.,impact=0.),
        dict(time='2021-01-05T21:00:00Z',reason='holding_limit',quantity=0,delta=-100,fee=0.,impact=0.)]
    r['metrics'].update(net_return=.03,costs=0.)
    r['directional'].update(long_net_pnl=3000.,short_net_pnl=0.,long_costs=0.,short_costs=0.)
    r['baselines']['buy_hold']=dict(rows=[dict(session='2021-01-04',equity=102000.),
        dict(session='2021-01-05',equity=104000.)],metrics=dict(net_return=.04,costs=0.))
    x=exit_partition(r)
    assert x['flat/neither']['sessions']==1 and x['flat/neither']['excess_pp']==pytest.approx(-1.)
    assert x['long/holding']['sessions']==1 and x['long/holding']['excess_pp']==pytest.approx(0.)


@pytest.mark.parametrize('corruption',['quantity','holding','equity'])
def test_exit_partition_rejects_unreconciled_sources(corruption):
    from rl_stocks.reporting.action_execution import exit_partition
    r=sample()
    if corruption=='quantity':r['events'][-1]['delta']=99
    if corruption=='holding':r['baselines']['buy_hold']['rows'][0]['equity']=103000.
    if corruption=='equity':r['rows'][0]['previous_equity']=99999.
    with pytest.raises(ValueError):exit_partition(r)


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


def test_fixed_training_period_is_explicit_and_default_stays_development_only():
    r=sample();r['split']='train'
    for row in r['rows']:row['session']='2018-01-02'
    for row in r['baselines']['buy_hold']['rows']:row['session']='2018-01-02'
    for event in r['events']:event['time']='2018-01-02'+event['time'][10:]
    with pytest.raises(ValueError,match='development-validation'):diagnose_episode(r)
    got=diagnose_episode(r,period='training-window')
    assert got['sessions']==1
    for row in r['rows']:row['session']='2024-01-02'
    with pytest.raises(ValueError,match='fixed training-window'):diagnose_episode(r,period='training-window')
    with pytest.raises(ValueError,match='declared diagnostic period'):diagnose_episode(r,period='test')
