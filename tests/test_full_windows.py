import numpy as np
import pytest
from rl_stocks.environment.sampling.windows import FullWindows


def episode(symbol='SPY',year=2001,n=1000):
    # Sorted synthetic session identifiers; no model fitting or market inference.
    from datetime import date,timedelta
    start=date(year,1,1)
    return dict(symbol=symbol,split='train',bars=[dict(session=str(start+timedelta(days=i))) for i in range(n)])


def test_every_sample_allows_exact_window_and_is_seeded():
    sampler=FullWindows([episode(),episode('QQQ',2010,513)],512)
    a=np.random.default_rng(11);b=np.random.default_rng(11)
    draws=[sampler.choose(a) for _ in range(2000)]
    assert draws==[sampler.choose(b) for _ in range(2000)]
    assert all(0<=s<=487 if i==0 else s==0 for i,s in draws)
    assert 900<sum(i==1 for i,s in draws)<1100
    assert sampler.groups==['SPY:earlier','QQQ:recent']
    assert not sampler.summary['future_outcomes_used']


@pytest.mark.parametrize('episodes,steps',[([episode(n=512)],512),([episode(year=2021)],512),([episode()],None),([],512)])
def test_invalid_windows_rejected(episodes,steps):
    with pytest.raises(ValueError):FullWindows(episodes,steps)


def test_mixed_period_or_nontraining_rejected():
    e=episode(year=2009,n=730)
    with pytest.raises(ValueError):FullWindows([e],512)
    e=episode();e['split']='validation'
    with pytest.raises(ValueError):FullWindows([e],512)


def test_environment_full_horizon_and_realized_counts():
    from test_causal_sampling import episode as market_episode,env
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check
    e=env(market_episode(),'full_window_uniform')
    check_env(e,skip_render_check=True);sb3_check(e)
    e.reset_sampling_statistics();e.reset(seed=23)
    for i in range(512):
        _,_,terminated,truncated,_=e.step([0,1,1,1])
        assert not terminated and truncated==(i==511)
    stats=e.sampling_statistics()
    assert stats['history_groups']=={'SPY:recent':{'starts':1,'transitions':512}}
    e.reset();e.reset_sampling_statistics();e.step([0,1,1,1])
    assert e.sampling_statistics()['history_groups']['SPY:recent']['transitions']==1


def test_statistics_snapshot_is_not_mutated_by_parity_reset():
    from test_causal_sampling import episode as market_episode,env
    e=env(market_episode(),'full_window_uniform');e.reset(seed=11);e.step([0,1,1,1])
    snapshot=e.sampling_statistics();e.reset()
    assert snapshot['starts']==1 and snapshot['history_groups']['SPY:recent']['starts']==1
    assert e.sampling_statistics()['history_groups']['SPY:recent']['starts']==2
