from copy import deepcopy
from datetime import date,timedelta
import pytest
from rl_stocks.environment.sampling.years import YearCounts,validate_year_counts


def sample():
    c=YearCounts();c.start('SPY','2019-12-31');c.transition('SPY','2019-12-31','2020-01-02')
    c.transition('SPY','2020-01-02','2020-01-03');return c.snapshot(1,2)


def test_exact_boundary_pairs_and_snapshot_immutability():
    c=YearCounts();c.start('SPY','2019-12-31');c.transition('SPY','2019-12-31','2020-01-02')
    x=c.snapshot(1,1);c.transition('SPY','2020-01-02','2020-01-03')
    assert x['decision_execution_year_pairs']=={'2019->2020':1}
    assert x['decision_years']=={'2019':1} and x['execution_years']=={'2020':1}
    assert c.snapshot(1,2)==sample()


@pytest.mark.parametrize('corruption',['total','pair','future','symbol','fractional'])
def test_corrupt_margins_and_years_rejected(corruption):
    x=sample()
    if corruption=='total':x['start_years']['2019']=2
    if corruption=='pair':x['decision_execution_year_pairs']={'2019->2020':2}
    if corruption=='future':x['execution_years']={'2021':2}
    if corruption=='symbol':x['per_symbol']['SPY']['decision_years']={'2020':2}
    if corruption=='fractional':x['execution_years']={'2020':2.0}
    with pytest.raises(ValueError):validate_year_counts(x,1,2)


def test_environment_measures_actual_dates_across_year_and_reset_midwindow():
    from test_causal_sampling import episode,env
    ep=episode();ep['bars']=ep['bars'][:513];ep['features']=ep['features'][:513];ep['atr']=ep['atr'][:513]
    # Exactly one legal start; no random date selection or learning.
    # A full window crossing a calendar-year boundary, entirely before2021.
    for i,b in enumerate(ep['bars']):b['session']=str(date(2018,8,8)+timedelta(days=i))
    e=env(ep,'full_window_uniform');e.reset(seed=11)
    expected={}
    for i in range(512):
        y=ep['bars'][i+1]['session'][:4];expected[y]=expected.get(y,0)+1
        e.step([0,1,1,1])
    stats=e.sampling_statistics();assert stats['year_counts']['execution_years']==expected
    assert sum(stats['year_counts']['decision_execution_year_pairs'].values())==512
    rng=deepcopy(e.np_random.bit_generator.state);e.sampling_statistics();assert e.np_random.bit_generator.state==rng
    e.reset();e.reset_sampling_statistics();e.step([0,1,1,1])
    stats=e.sampling_statistics();assert stats['starts']==0 and stats['year_counts']['per_symbol']['SPY']['starts']==0
    assert stats['year_counts']['per_symbol']['SPY']['transitions']==1


def test_diversity_checks_years_against_independent_symbol_totals():
    from test_diversity_jobs import counts
    from rl_stocks.training.diversity import validate_counts
    c=counts('expanded',True)
    per={key.split(':')[0]:dict(**row,decision_years={'2010':row['transitions']},execution_years={'2010':row['transitions']}) for key,row in c['history_groups'].items()}
    c['year_counts']=dict(version=1,measured=True,start_years={'2010':c['starts']},decision_years={'2010':1024},execution_years={'2010':1024},decision_execution_year_pairs={'2010->2010':1024},per_symbol=per)
    validate_counts(c,'expanded',True,1024)
    # Internally consistent year totals cannot impersonate another symbol.
    old=next(iter(per));per['UNREGISTERED']=per.pop(old)
    with pytest.raises(ValueError,match='year counts differ'):validate_counts(c,'expanded',True,1024)
