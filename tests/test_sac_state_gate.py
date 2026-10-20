from test_history_gate import cells
from rl_stocks.reporting.history import screen_gate
from rl_stocks.training.sac_state import protocol


def test_sac_protocol_uses_both_capacity_information_controls_and_absolute_targets():
    old=cells();c=dict(legacy=old['original'],execution_masked=old['recent_etf'],execution_full=old['older_etf'])
    assert screen_gate(c,protocol())['passed']
    c['execution_full']['final']['groups']['etf']['excess_return']=-.03
    verdict=screen_gate(c,protocol())
    assert not verdict['passed'] and verdict['promoted_candidate'] is None
    assert not verdict['objective_achieved']
