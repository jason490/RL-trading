from test_history_gate import cells
from rl_stocks.training.diversity import protocol
from rl_stocks.reporting.history import screen_gate


def test_absolute_group_failure_cannot_be_rescued_by_control_gain():
    source=cells();c=dict(original=source['original'],expanded=source['older_etf'])
    assert screen_gate(c,protocol())['passed']
    c['expanded']['final']['groups']['etf']['excess_return']=-.03
    verdict=screen_gate(c,protocol())
    assert not verdict['passed'] and verdict['promoted_candidate'] is None
