from copy import deepcopy
from types import SimpleNamespace
import pytest
from rl_stocks.environment.accounting.portfolio import Portfolio
from rl_stocks.reporting.pnl_bridge import bridge


def fixture(short=False,split=1.):
    bars=[dict(session='2021-01-04',open=100.,close=101.,split=1.,dividend=0.),
          dict(session='2021-01-05',open=102.,close=105.,split=1.,dividend=0.),
          dict(session='2021-01-06',open=108./split,close=110./split,split=split,dividend=.2)]
    p=Portfolio();rows=[];previous=100000.;sign=-1 if short else 1
    for i,b in enumerate(bars[1:]):
        p.corporate_action(SimpleNamespace(**b))
        if i==0:p.trade(sign*101,b['open'],b['session']+'T14:30:00Z','add')
        else:
            # A partial target followed by reversal; ordered fills can share a timestamp.
            p.trade(sign*50,b['open']+1,b['session']+'T21:00:00Z','target_near')
            p.trade(-sign*20,b['close']-1,b['session']+'T14:30:00Z','reverse')
        cost=p.borrow(b['close']);equity=p.equity(b['close'])
        rows.append(dict(session=b['session'],previous_equity=previous,equity=equity,
                         quantity=p.quantity,borrow_cost=cost));previous=equity
    return dict(split='validation',bars=bars),rows,p.events


@pytest.mark.parametrize('short',[False,True])
@pytest.mark.parametrize('split',[1.,1.5,.5])
def test_split_dividend_partial_exit_reversal_borrow_and_gap_reconcile(short,split):
    e,r,v=fixture(short,split);x=bridge(e,r,v)
    assert x['max_session_error']<1e-8
    assert sum(x['components_pp'].values())==pytest.approx(x['net_return_pp'])
    assert x['rows'][0]['inherited_side']=='flat'
    assert x['rows'][1]['inherited_side']==('short' if short else 'long')
    assert x['components_pp']['overnight']==pytest.approx((-1 if short else 1)*101*3/1000)


@pytest.mark.parametrize('corruption',['quantity','impact','price','equity','date','borrow','baseline_length'])
def test_bridge_rejects_corrupt_or_non_development_paths(corruption):
    e,r,v=deepcopy(fixture())
    if corruption=='quantity':v[-1]['quantity']+=1
    if corruption=='impact':v[0]['impact']+=1
    if corruption=='price':e['bars'][1]['open']=float('nan')
    if corruption=='equity':r[-1]['equity']+=1
    if corruption=='date':e['split']='test'
    if corruption=='borrow':r[0]['borrow_cost']=-1
    if corruption=='baseline_length':r.pop()
    with pytest.raises(ValueError):bridge(e,r,v)


def test_holding_bridge_uses_original_initial_equity_and_matches_saved_path():
    from rl_stocks.reporting.pnl_bridge import holding_bridge
    from rl_stocks.environment.execution.simulator import SimulationConfig
    e,_,_=fixture()
    for b in e['bars']:b['open_time']=b['session']+'T14:30:00Z'
    saved=dict(rows=[dict(session='2021-01-05',equity=102940.),
                     dict(session='2021-01-06',equity=108036.)])
    c=SimulationConfig(fee_bps=0.,impact_bps=0.)
    x=holding_bridge(e,saved,c)
    assert x['net_return_pp']==pytest.approx(8.036)
    assert x['components_pp']['dividends']==pytest.approx(.196)
    saved['rows'][-1]['equity']+=1
    with pytest.raises(ValueError):holding_bridge(e,saved,c)


def test_overnight_partition_lags_exit_flags_and_preserves_overlap():
    from rl_stocks.reporting.pnl_bridge import overnight_partition
    rows=[dict(session=f'2021-01-0{i+4}',inherited_side=side,components=dict(overnight=v))
          for i,(side,v) in enumerate([('flat',0.),('long',100.),('flat',0.)])]
    policy=dict(rows=rows,components_pp=dict(overnight=.1))
    base=deepcopy(policy)
    for r in base['rows']:r['components']['overnight']=200.
    base['components_pp']['overnight']=.6
    events=[dict(time='2021-01-04T21:00:00Z',reason='target_near'),
            dict(time='2021-01-04T21:00:00Z',reason='holding_limit'),
            dict(time='2021-01-05T21:00:00Z',reason='stop')]
    x=overnight_partition(policy,base,events)
    assert x['flat/initial']['excess_pp']==pytest.approx(-.2)
    assert x['long/both']['excess_pp']==pytest.approx(-.1)
    assert x['flat/protection']['excess_pp']==pytest.approx(-.2)
    assert sum(v['sessions'] for v in x.values())==3
    assert sum(v['excess_pp'] for v in x.values())==pytest.approx(-.5)
    policy['components_pp']['overnight']=1.
    with pytest.raises(ValueError):overnight_partition(policy,base,events)
