"""Mixed-algorithm reporting requires an exact explicit panel."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import sys
import pytest
from test_participation import fixture

spec=importlib.util.spec_from_file_location('participation_cli',Path(__file__).resolve().parents[1]/'scripts/diagnose_participation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def setup(tmp_path, algorithms):
    universe=tmp_path/'universe.json'
    universe.write_text(json.dumps({'instruments':[dict(symbol=s,kind=k,role='tradable') for s,k in [('SPY','etf'),('AAPL','stock')]]}))
    completed={}
    for index,algorithm in enumerate(algorithms):
        root=tmp_path/str(index);run=root/'runs'/f'run-{index}';run.mkdir(parents=True)
        metadata=dict(status='complete',smoke=False,actual_steps=100000,algorithm=algorithm,dataset='fixture',seed=11,git={'revision':'source'},source_hash='identity',reward_config={'kind':'net_log_equity','scale':1})
        (run/'metadata.json').write_text(json.dumps(metadata))
        result={'SPY':fixture(),'AAPL':deepcopy(fixture())};result['AAPL']['symbol']='AAPL'
        for name in ['selected-validation.json','validation.json']:(run/name).write_text(json.dumps(result))
        completed[str(index)]={'root':str(root)}
    state=tmp_path/'state.json';state.write_text(json.dumps(dict(status='complete',cleanup_verified=True,completed=completed)))
    return ['diagnose','--state',str(state),'--universe',str(universe),'--steps','100000','--output',str(tmp_path/'result.json')]


def test_explicit_mixed_panel_and_recorded_algorithms(tmp_path,monkeypatch):
    argv=setup(tmp_path,['ppo','recurrent_ppo'])+['--algorithm','ppo','--algorithm','recurrent_ppo']
    monkeypatch.setattr(sys,'argv',argv);module.main()
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['declared_algorithm']==['ppo','recurrent_ppo']
    assert {r['algorithm'] for r in result['records']}=={'ppo','recurrent_ppo'}
    assert not result['training_performed'] and result['source_files_unchanged']


def test_undeclared_algorithm_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',setup(tmp_path,['ppo','recurrent_ppo']))
    with pytest.raises(ValueError,match='declared completed'):module.main()


def test_declared_but_absent_algorithm_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',setup(tmp_path,['ppo'])+['--algorithm','ppo','--algorithm','recurrent_ppo'])
    with pytest.raises(ValueError,match='exact declared'):module.main()
