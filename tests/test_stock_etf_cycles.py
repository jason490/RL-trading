from collections import Counter
from copy import deepcopy
from datetime import date,timedelta
import numpy as np
import pytest
from rl_stocks.environment.sampling.windows import StockEtfCycles


def episodes(stocks=107):
    days=[(date(2010,1,1)+timedelta(days=i)).isoformat() for i in range(600)]
    return [dict(symbol=s,split='train',bars=[dict(session=d) for d in days])
            for s in (*StockEtfCycles.etfs,*(f'S{i}' for i in range(stocks)))]


def test_group_cycles_and_fixed_probability_are_independent_of_stock_pool():
    for count in (22,107):
        pool=episodes(count);s=StockEtfCycles(pool,512);rng=np.random.default_rng(11);seen=Counter()
        for _ in range(196):
            index,start=s.choose(rng);seen[index]+=1
            assert 0<=start and start+512<len(pool[index]['bars'])
        for group,indices in s.buckets.items():
            counts=[seen[i] for i in indices]
            assert min(counts)>0 and max(counts)-min(counts)<=1
        assert s.summary['group_probability']==dict(stock_numerator=22,etf_numerator=4,denominator=26)


def test_seed_reset_and_price_outcomes_do_not_change_selection():
    a=episodes(22);b=deepcopy(a)
    for e in b:
        for row in e['bars']:row['close']=0.001
    left,right=StockEtfCycles(a,128),StockEtfCycles(b,128)
    def sequence(s):
        rng=np.random.default_rng(23);return [s.choose(rng) for _ in range(90)]
    expected=sequence(left);assert sequence(right)==expected
    left.reset_cycles();assert sequence(left)==expected


@pytest.mark.parametrize('change',[
    lambda e:e.pop(0),lambda e:e.append(deepcopy(e[0])),
    lambda e:e[0].update(split='validation'),lambda e:e[0]['bars'].pop(),
])
def test_invalid_pool_rejected(change):
    e=episodes(1)
    change(e)
    # Last mutation shortens a600-bar pool, still sufficient at512; demand599.
    with pytest.raises(ValueError):StockEtfCycles(e,599)


def test_real_environment_seed_checks_full_windows_and_count_snapshots():
    from copy import deepcopy
    from test_causal_sampling import episode
    from rl_stocks.environment.trading.env import TradingEnv
    from rl_stocks.data.generation.datasets import Normalizer
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check
    pool=[]
    for symbol in (*StockEtfCycles.etfs,'AAPL'):
        e=deepcopy(episode());e['symbol']=symbol
        for b in e['bars']:b['symbol']=symbol
        pool.append(e)
    env=TradingEnv(pool,Normalizer(('fixture',),(0.,),(1.,)),['fixture'],random_start=True,
                   episode_steps=128,start_sampling=StockEtfCycles.mode)
    check_env(env,skip_render_check=True);sb3_check(env)
    env.reset_sampling_statistics();env.reset(seed=11)
    for i in range(128):
        _,_,done,truncated,_=env.step([0,1,1,1]);assert not done and truncated==(i==127)
    snapshot=env.sampling_statistics();env.reset()
    assert snapshot['starts']==1 and snapshot['transitions']==128
    assert sum(v['transitions'] for v in snapshot['history_groups'].values())==128
    assert env.sampling_statistics()['starts']==2
