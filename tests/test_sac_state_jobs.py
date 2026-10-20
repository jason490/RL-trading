from copy import deepcopy
import pytest
from rl_stocks.runtime.colab.jobs import sac_state_plan,validate_plan
from rl_stocks.runtime.colab.fleet import validate_tasks
from rl_stocks.training.sac_state import protocol,validate_payload


def test_fixed_sac_state_parameters_and_modes():
    for arm in protocol()['arms']:
        for smoke in (False,True):
            p=sac_state_plan(arm,smoke=smoke);assert validate_plan(p)==p
            assert p['jobs'][0]['steps']==(1024 if smoke else 50000)
            for mutate in [lambda v:v.update(accelerator='A100'),lambda v:v.update(protocol_sha256='wrong'),lambda v:v['jobs'][0].update(algorithm='ppo'),lambda v:v['jobs'][0].update(initialization='wide'),lambda v:v['jobs'][0].update(steps=250000),lambda v:v['dataset_contract'].update(identity='wrong')]:
                bad=deepcopy(p);mutate(bad)
                with pytest.raises(ValueError):validate_plan(bad)
    for args in [('wrong',11,False),('legacy',23,True),('legacy',True,False)]:
        with pytest.raises(ValueError):sac_state_plan(*args)


def test_wrong_data_fails_before_allocation(tmp_path):
    folder=tmp_path/'wrong';folder.mkdir();(folder/'dataset.json').write_text('{}')
    with pytest.raises(ValueError,match='identity'):validate_tasks([dict(id='legacy',dataset=str(folder),plan=sac_state_plan('legacy'))],tmp_path)
    with pytest.raises(ValueError,match='identity'):validate_payload({})
