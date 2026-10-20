from copy import deepcopy
import pytest
from rl_stocks.runtime.colab.jobs import temperature_plan,validate_plan
from rl_stocks.training.sac_temperature import protocol,validate_initialization


def test_fixed_temperature_plans_and_initialization_validation():
    for arm in protocol()['arms']:
        for smoke in (False,True):
            p=temperature_plan(arm,smoke=smoke);assert validate_plan(p)==p
            for mutate in [lambda v:v['jobs'][0]['overrides'].update(ent_coef='auto_0.1'),lambda v:v['jobs'][0]['overrides'].update(gamma=.95),lambda v:v['jobs'][0].update(reward_scale=100.),lambda v:v.update(accelerator='T4')]:
                bad=deepcopy(p);mutate(bad)
                with pytest.raises(ValueError):validate_plan(bad)
        r=dict(alpha=protocol()['initial_alpha'][arm],target_entropy=-4.,network_sha256={k:'a'*64 for k in ('actor','critic','critic_target')})
        validate_initialization(r,arm)
        r['alpha']*=10
        with pytest.raises(ValueError):validate_initialization(r,arm)
    for args in [('unknown',11,False),('default',True,False),('default',23,True)]:
        with pytest.raises(ValueError):temperature_plan(*args)


def test_initial_fingerprints_ignore_only_temperature_without_mutating_weights():
    import torch
    from types import SimpleNamespace
    from copy import deepcopy
    from rl_stocks.training.sac_temperature import initialization_record
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        a=SimpleNamespace(actor=torch.nn.Linear(2,2),critic=torch.nn.Linear(2,2),critic_target=torch.nn.Linear(2,2),log_ent_coef=torch.tensor([0.]),target_entropy=-4.)
        b=deepcopy(a);b.log_ent_coef=torch.tensor([.001]).log()
        ra=initialization_record(a);rb=initialization_record(b)
        assert ra['network_sha256']==rb['network_sha256']
        assert ra['alpha']==1. and rb['alpha']==pytest.approx(.001)
        assert initialization_record(a)==ra
