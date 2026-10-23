from collections import Counter
from datetime import date,timedelta
import pytest
from rl_stocks.reporting.history_coverage import window_coverage


@pytest.mark.parametrize('n,steps',[(2,1),(8,3),(8,7),(9,1)])
def test_expected_execution_mass_matches_every_start(n,steps):
    sessions=[(date(2019,12,28)+timedelta(days=i)).isoformat() for i in range(n)]
    got=window_coverage(sessions,steps);expected=Counter()
    for start in range(n-steps):
        for t in range(start+1,start+steps+1):expected[sessions[t][:4]]+=1
    assert got['eligible_starts']==n-steps
    assert got['expected_execution_year_fraction']=={y:v/((n-steps)*steps) for y,v in sorted(expected.items())}
    assert not got['actual_year_visitation_measured']


@pytest.mark.parametrize('sessions,steps',[
    (['2021-01-01','2021-01-04'],1),(['2020-01-02','2020-01-01'],1),
    (['2020-01-01','2020-01-01'],1),(['2020-01-01'],1),
    (['2020-01-01','2020-01-02'],True),(['2020-01-01','2020-01-02'],0)])
def test_invalid_or_nontraining_sources_reject(sessions,steps):
    with pytest.raises(ValueError):window_coverage(sessions,steps)


def test_date_audit_requires_no_market_outcomes_and_preserves_source(tmp_path):
    import importlib.util,json
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('history_audit',Path(__file__).resolve().parents[1]/'scripts/audit_training_history.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sessions=[(date(2016,1,1)+timedelta(days=i)).isoformat() for i in range(600)]
    payload={'manifest':{'included_splits':['train','validation']},'episodes':{
        s:{'symbol':s,'split':'train','bars':[{'session':d} for d in sessions]}
        for s in ['AAPL','SPY']}}
    # A development episode without bars witnesses that its values stay unread.
    payload['episodes']['dev']={'split':'validation'}
    path=tmp_path/'data.json';path.write_text(json.dumps(payload));before=path.read_bytes()
    got=module.audit(path)
    assert got['source_unchanged'] and path.read_bytes()==before
    assert len(got['records'])==2 and not got['actual_year_visitation_measured']
    payload['manifest']['included_splits'].append('test');path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='train/development-only'):module.audit(path)
