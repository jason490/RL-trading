from copy import deepcopy
from types import SimpleNamespace
import pytest
from rl_stocks.runtime.colab.jobs import td3_state_plan, validate_plan
from rl_stocks.runtime.colab.fleet import validate_tasks
from rl_stocks.training.td3_state import initialization_record, validate_initialization


def test_fixed_plans_reject_configuration_drift(tmp_path):
    for arm in ('legacy', 'execution_masked', 'execution_full'):
        for smoke in (False, True):
            p = td3_state_plan(arm, smoke=smoke)
            assert validate_plan(p) == p
            for change in (lambda v: v.update(accelerator='A100'),
                           lambda v: v['jobs'][0].update(algorithm='sac'),
                           lambda v: v['jobs'][0].update(overrides={'policy_delay': 1}),
                           lambda v: v['jobs'][0].update(steps=250000),
                           lambda v: v['jobs'][0].update(observation_profile='legacy' if arm != 'legacy' else 'execution_full')):
                bad = deepcopy(p); change(bad)
                with pytest.raises(ValueError):
                    validate_plan(bad)
    folder = tmp_path / 'wrong'; folder.mkdir(); (folder / 'dataset.json').write_text('{}')
    with pytest.raises(ValueError, match='identity'):
        validate_tasks([dict(id='td3', dataset=str(folder), plan=td3_state_plan('legacy'))], tmp_path)
    for args in [('wrong',11,False), ('legacy',True,False), ('legacy',23,True)]:
        with pytest.raises(ValueError):
            td3_state_plan(*args)


def test_initial_fingerprints_preserve_rng_and_detect_noise_and_target_changes():
    import torch
    import numpy as np
    from stable_baselines3.common.noise import NormalActionNoise
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        m = SimpleNamespace(**{n: torch.nn.Linear(2,2) for n in ('actor','actor_target','critic','critic_target')},
                            action_noise=NormalActionNoise(np.zeros(4),np.full(4,.1)),
                            policy_delay=2,target_policy_noise=.2,target_noise_clip=.5)
        rng = torch.get_rng_state().clone()
        before = initialization_record(m)
        assert torch.equal(rng, torch.get_rng_state())
        assert initialization_record(deepcopy(m)) == before
        with torch.no_grad():
            m.actor_target.weight[0,0] += 1
        after = initialization_record(m)
        assert after['network_sha256']['actor'] == before['network_sha256']['actor']
        assert after['network_sha256']['actor_target'] != before['network_sha256']['actor_target']
        bad = deepcopy(before); bad['exploration_noise']['sigma'][0] = .2
        with pytest.raises(ValueError):
            validate_initialization(bad)
        m.action_noise = None
        with pytest.raises(ValueError):
            initialization_record(m)
