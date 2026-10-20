from test_history_gate import cells
from rl_stocks.training.sac_regularization import protocol
from rl_stocks.reporting.history import screen_gate


def test_fixed_regularization_gate_requires_absolute_selected_and_final_targets():
    source=cells();panel={'control':source['original'],'weight_decay':source['older_etf']}
    assert screen_gate(panel,protocol())['passed']
    panel['weight_decay']['final']['groups']['etf']['excess_return']=-.03
    assert not screen_gate(panel,protocol())['passed']
