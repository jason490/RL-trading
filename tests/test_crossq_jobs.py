"""Fixed job contracts; no model construction or remote allocation."""
from copy import deepcopy

import pytest

from rl_stocks.runtime.colab.jobs import crossq_plan, validate_plan
from rl_stocks.training.crossq_spec import parameters
from test_crossq_runtime import accepted_profile


@pytest.mark.parametrize('arm', ['sac', 'crossq'])
def test_smoke_and_screen_exact_contract(tmp_path, arm):
    for smoke, profile, steps, every in ((True, None, 4096, 1024),
                                        (False, accepted_profile(tmp_path), 50000, 10000)):
        plan = crossq_plan(arm, smoke=smoke, profile=profile)
        assert validate_plan(plan) == plan
        job = plan['jobs'][0]
        assert (job['steps'], job['evaluation_every'], job['milestones']) == (steps, every, [steps])
        assert job['overrides'] == parameters(arm, smoke)
        assert job['overrides']['policy_kwargs']['net_arch'] == [64, 64]
        assert job['start_sampling'] == 'legacy_uniform'
        assert job['observation_profile'] == 'execution_full'
        assert job['algorithm'] == arm
        assert plan['accelerator'] == 'CPU'


@pytest.mark.parametrize('field,value', [('steps', 100000), ('device', 'cuda'),
                                        ('start_sampling', 'stock_cycle'), ('algorithm', 'ppo'),
                                        ('evaluation_every', 512), ('final_test', True)])
def test_job_mutation_rejected(field, value):
    plan = deepcopy(crossq_plan('crossq', smoke=True))
    plan['jobs'][0][field] = value
    with pytest.raises(ValueError):
        validate_plan(plan)


@pytest.mark.parametrize('kwargs', [dict(arm='crossq'), dict(arm='crossq', smoke=True, seed=23),
                                    dict(arm='crossq', smoke=True, seed=True),
                                    dict(arm='crossq', smoke=True, profile={}),
                                    dict(arm='tqc', smoke=True)])
def test_unaccepted_request_rejected(kwargs):
    with pytest.raises(ValueError):
        crossq_plan(**kwargs)
