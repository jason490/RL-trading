import math
from copy import deepcopy
import numpy as np
import pytest
from rl_stocks.environment.rewards.equity import equity_reward
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.runtime.colab.jobs import reward_plan, validate_plan
from test_causal_sampling import episode


def test_objectives_telescope_and_can_rank_episode_distributions_differently():
    path=[100000.,95000.,110000.,90000.,125000.]
    assert sum(equity_reward(a,b) for a,b in zip(path,path[1:]))==pytest.approx(math.log(1.25))
    assert sum(equity_reward(a,b,'net_initial_equity_change') for a,b in zip(path,path[1:]))==pytest.approx(.25)
    # Same initial capital; arithmetic favors A, log favors B. No future label.
    a=[50000.,200000.];b=[120000.,120000.]
    score=lambda values,kind:sum(equity_reward(100000.,v,kind) for v in values)/2
    assert score(a,'net_initial_equity_change')>score(b,'net_initial_equity_change')
    assert score(a,'net_log_equity')<score(b,'net_log_equity')
    assert equity_reward(100000.,-1000.,'net_initial_equity_change')==pytest.approx(-1.01)
    assert equity_reward(100000.,0.)==-20.
    for args in [(1.,float('nan')), (1.,2.,'future'),(1.,2.,'net_initial_equity_change',0.)]:
        with pytest.raises(ValueError):equity_reward(*args)


def test_reward_changes_no_execution_observation_or_rng_and_reconciles_after_costs():
    envs=[TradingEnv([episode()],Normalizer(('fixture',),(0.,),(1.,)),['fixture'],
        random_start=True,episode_steps=16,observation_profile='execution_full',reward_kind=k)
        for k in ('net_log_equity','net_initial_equity_change')]
    observations=[e.reset(seed=11)[0] for e in envs]
    np.testing.assert_array_equal(*observations)
    reward_sum=0.
    for action in [[-.7,1.,1.,1.],[.5,1.,1.,1.],[0.,1.,1.,1.]]*3:
        results=[e.step(action) for e in envs]
        np.testing.assert_array_equal(results[0][0],results[1][0])
        info=[deepcopy(r[4]) for r in results]
        reward_sum+=results[1][1]
        assert results[1][1]==pytest.approx((info[1]['equity']-info[1]['previous_equity'])/100000.)
        for item in info:
            item.pop('unscaled_reward');item.pop('reward_kind')
        assert info[0]==info[1] and results[0][2:4]==results[1][2:4]
        assert envs[0].np_random.bit_generator.state==envs[1].np_random.bit_generator.state
    assert reward_sum==pytest.approx((info[1]['equity']-100000.)/100000.)


def test_remote_reward_plans_are_locked_and_default_log_control_unchanged():
    for kind in ('net_log_equity','net_initial_equity_change'):
        for smoke in (True,False):
            plan=reward_plan(kind,smoke=smoke);assert validate_plan(plan)==plan
            changed=deepcopy(plan);changed['jobs'][0]['reward_kind']='future'
            with pytest.raises(ValueError):validate_plan(changed)
    with pytest.raises(ValueError):reward_plan('short_bonus')
