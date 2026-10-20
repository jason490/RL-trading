from copy import deepcopy
import pytest
from rl_stocks.reporting.history import screen_gate
from rl_stocks.training.history.spec import protocol


def cells():
    out={}
    for arm,value in [('original',.1),('recent_etf',.11),('older_etf',.2)]:
        phase=dict(score=value,mean_max_drawdown=-.1,groups={g:dict(net_return=value,excess_return=value-.2,median_excess_return=value-.2) for g in ('stock','etf')})
        out[arm]=dict(selected=deepcopy(phase),final=deepcopy(phase))
    return out


def test_pass_requires_both_controls_and_both_absolute_phases():
    assert screen_gate(cells(),protocol())['passed']
    for mutate in [lambda c:c['recent_etf']['final'].update(score=.21),lambda c:c['older_etf']['selected']['groups']['etf'].update(excess_return=-.03),lambda c:c['older_etf']['final']['groups']['stock'].update(median_excess_return=-.03),lambda c:c['older_etf']['final'].update(mean_max_drawdown=-.13)]:
        c=cells();mutate(c);v=screen_gate(c,protocol());assert not v['passed'] and v['promoted_candidate'] is None
    assert not screen_gate(cells(),protocol())['objective_achieved']


def test_incomplete_and_nonfinite_rejected():
    c=cells();del c['original']
    with pytest.raises(ValueError):screen_gate(c,protocol())
    c=cells();c['older_etf']['final']['score']=float('nan')
    with pytest.raises(ValueError):screen_gate(c,protocol())
