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
