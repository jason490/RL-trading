from copy import deepcopy
import pytest
from rl_stocks.runtime.colab.jobs import regularization_plan,validate_plan
from rl_stocks.training.sac_regularization import protocol


def test_plans_reject_unregistered_changes():
    for arm in protocol()['arms']:
        for smoke in (True,False):
            p=regularization_plan(arm,smoke=smoke)
            assert validate_plan(p)==p
            for change in (lambda q:q['jobs'][0]['overrides']['policy_kwargs']['optimizer_kwargs'].update(weight_decay=.01),
                           lambda q:q['jobs'][0].update(steps=250000),
                           lambda q:q['jobs'][0].update(start_sampling='legacy_uniform'),
                           lambda q:q['jobs'][0]['overrides'].update(ent_coef='auto_0.001')):
                q=deepcopy(p);change(q)
                with pytest.raises(ValueError):validate_plan(q)
    with pytest.raises(ValueError):regularization_plan('control',True)
    with pytest.raises(ValueError):regularization_plan('weight_decay',23,True)
