from test_history_gate import cells
from rl_stocks.training.sac_discount import protocol
from rl_stocks.reporting.history import screen_gate


def test_discount_targets_cannot_be_relaxed_by_improvement_against_control():
    source=cells();panel={'control':source['original'],'short_horizon':source['older_etf']}
    assert screen_gate(panel,protocol())['passed']
    panel['short_horizon']['selected']['groups']['stock']['median_excess_return']=-.03
    assert not screen_gate(panel,protocol())['passed']
