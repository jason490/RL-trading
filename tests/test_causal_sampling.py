from copy import deepcopy
from dataclasses import asdict
from datetime import date,timedelta
import numpy as np
import pytest

from test_core import bar
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.environment.sampling.regimes import CausalStarts
from rl_stocks.environment.trading.env import TradingEnv


def episode():
    prices=[100.]*300+[100.-.35*i for i in range(200)]+[30.+.3*i for i in range(200)]
    bars=[asdict(bar(str(date(2016,1,1)+timedelta(days=i)),op=p,high=p+1,low=p-1,close=p,
                    adjustments='split_adjusted_with_dividends')) for i,p in enumerate(prices)]
    return {'symbol':'SPY','split':'train','bars':bars,'features':[[0.] for _ in bars],'atr':[2.]*len(bars)}


def env(ep,mode='legacy_uniform'):
    return TradingEnv([ep],Normalizer(('fixture',),(0.,),(1.,)),['fixture'],
                      random_start=True,episode_steps=512,start_sampling=mode)


def test_sampler_prefix_labels_cannot_see_later_training_prices():
    first=episode();changed=deepcopy(first)
    for b in changed['bars'][450:]:b['close']*=10
    a=CausalStarts([first],'ready_uniform');b=CausalStarts([changed],'ready_uniform')
    assert a.labels[0][:450]==b.labels[0][:450]
    assert a.all==b.all
    assert min(i for _,i in a.all)==252
    assert max(i for _,i in a.all)==len(first['bars'])-22


def test_balancing_changes_class_frequency_and_not_within_class_eligibility():
    ep=episode();uniform=CausalStarts([ep],'ready_uniform');balanced=CausalStarts([ep],'downtrend_balanced')
    assert uniform.all==balanced.all and uniform.down==balanced.down
    rng=np.random.default_rng(11)
    labels=[balanced.label(*balanced.choose(rng)) for _ in range(5000)]
    assert .47<sum(labels)/len(labels)<.53
    for pool in (balanced.down,balanced.other):
        assert pool and all(pair in uniform.all for pair in pool)
    assert uniform.summary['eligible']==len(uniform.down)+len(uniform.other)


def test_legacy_rng_path_is_exact_and_training_diagnostics_exclude_checks():
    ep=episode();legacy=env(ep);legacy.reset(seed=23)
    reference=np.random.default_rng(23);reference.integers(1)
    expected=int(reference.integers(len(ep['bars'])-21))
    assert legacy.i==expected
    assert legacy.np_random.bit_generator.state==reference.bit_generator.state
    assert legacy.sampling_statistics() is None
    sampled=env(ep,'downtrend_balanced');sampled.reset(seed=23)
    sampled.step([0,1,1,1])
    assert sampled.sampling_statistics()['transitions']==1
    assert sampled.sampling_statistics()['starts']==1
    sampled.reset_sampling_statistics();assert sampled.sampling_statistics()['transitions']==0
    sampled.reset(seed=23);sampled.step([0,1,1,1])
    assert sampled.sampling_statistics()['transitions']==1


def test_nontraining_short_history_adjustment_and_empty_class_rejections():
    ep=episode()
    for split in ('validation','test'):
        bad=deepcopy(ep);bad['split']=split
        with pytest.raises(ValueError):env(bad,'downtrend_balanced')
    bad=deepcopy(ep);bad['bars'][-1]['session']='2021-01-01'
    with pytest.raises(ValueError,match='training dates'):CausalStarts([bad],'ready_uniform')
    bad=deepcopy(ep);bad['bars'][0]['split']=2.
    with pytest.raises(ValueError,match='adjusted'):CausalStarts([bad],'ready_uniform')
    bad=deepcopy(ep);bad['bars']=bad['bars'][:270]
    with pytest.raises(ValueError,match='ready start'):CausalStarts([bad],'ready_uniform')
    bad=deepcopy(ep)
    for b in bad['bars']:b['close']=100.
    with pytest.raises(ValueError,match='both training classes'):CausalStarts([bad],'downtrend_balanced')
    assert CausalStarts([bad],'ready_uniform').summary['downtrend']==0


def test_sampling_plans_are_strict_cpu_and_keep_the_four_action_contract():
    from rl_stocks.runtime.colab.jobs import SAMPLING_ARMS,sampling_plan,validate_plan
    for arm in SAMPLING_ARMS:
        for smoke in (True,False):
            plan=sampling_plan(arm,smoke=smoke);assert validate_plan(plan)==plan
            assert plan['jobs'][0]['initialization']=='wide'
            for key,value in [('steps',100000),('start_sampling','legacy_uniform'),('device','cuda')]:
                bad=deepcopy(plan);bad['jobs'][0][key]=value
                with pytest.raises(ValueError):validate_plan(bad)
    for seed in (True,19):
        with pytest.raises(ValueError):sampling_plan('small_uniform',seed)


def test_sampled_environment_passes_actual_gym_and_sb3_checkers():
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check
    for mode in ('ready_uniform','downtrend_balanced'):
        sample=env(episode(),mode);check_env(sample,skip_render_check=True);sb3_check(sample);sample.close()


def test_sampler_gate_uses_its_declared_reference_instead_of_hardcoded_standard():
    import json
    from pathlib import Path
    from test_initialization_gate import GateTests
    from rl_stocks.reporting.initialization import screen_gate,confirmation_gate
    protocol=json.loads(Path('configs/experiments/causal_sampling.json').read_text())
    cells={arm:GateTests.cell(.1 if arm=='small_uniform' else .2) for arm in protocol['arms']}
    result=screen_gate(cells,protocol)
    assert result['promoted_candidate']=='small_balanced'
    paired={arm:{str(s):cells[arm] for s in (11,23,37)} for arm in ('small_uniform','small_balanced')}
    assert confirmation_gate(paired,'small_balanced',protocol)['relative_improvement_passed']
