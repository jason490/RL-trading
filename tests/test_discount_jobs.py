from copy import deepcopy
import pytest
from stable_baselines3 import SAC
from test_sac_regularization import Spaces
from rl_stocks.training.sac_discount import protocol,overrides,model_record,validate_record
from rl_stocks.runtime.colab.jobs import discount_plan,validate_plan


def test_fixed_plans_and_actual_initial_networks_without_fitting():
    records=[]
    for arm in protocol()['arms']:
        for smoke in (False,True):
            plan=discount_plan(arm,smoke=smoke);assert validate_plan(plan)==plan
            for change in (lambda p:p['jobs'][0]['overrides'].update(gamma=.9),lambda p:p['jobs'][0].update(steps=250000),lambda p:p.update(account_contract={'minimum_balance':0,'teardown_balance':20})):
                bad=deepcopy(plan);change(bad)
                with pytest.raises(ValueError):validate_plan(bad)
        model=SAC('MlpPolicy',Spaces(),seed=11,device='cpu',**overrides(arm))
        r=model_record(model);validate_record(r,arm,initial=True);records.append(r)
        assert not model.actor.optimizer.state
        bad=deepcopy(r);bad['gamma']=.9
        with pytest.raises(ValueError):validate_record(bad,arm,initial=True)
    assert records[0]['optimizer']==records[1]['optimizer']
    assert records[0]['gamma']==.99 and records[1]['gamma']==.95
