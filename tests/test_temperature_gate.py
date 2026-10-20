from copy import deepcopy
from types import SimpleNamespace
import torch
from test_history_gate import cells
from rl_stocks.reporting.temperature import gate,model_equal
from rl_stocks.training.sac_temperature import protocol


def test_two_arm_gate_requires_relative_and_absolute_targets():
    old=cells();c=dict(default=old['original'],cold_start=old['older_etf'])
    assert gate(c,protocol())['passed']
    c['cold_start']['final']['groups']['etf']['excess_return']=-.03
    assert not gate(c,protocol())['passed']


def test_reproduction_checks_network_temperature_and_update_count():
    with torch.random.fork_rng(devices=[]):
        a=SimpleNamespace(policy=torch.nn.Linear(2,2),log_ent_coef=torch.tensor([-2.]),_n_updates=49000)
        b=deepcopy(a);assert model_equal(a,b)
        b.log_ent_coef=torch.tensor([-3.]);assert not model_equal(a,b)
        b=deepcopy(a);b._n_updates+=1;assert not model_equal(a,b)
        b=deepcopy(a)
        with torch.no_grad():b.policy.weight[0,0]+=1
        assert not model_equal(a,b)
