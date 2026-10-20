import json
import runpy
import sys
from pathlib import Path
from copy import deepcopy
import pytest
from rl_stocks.environment.observations.state import observation_spec
from rl_stocks.rl.ppo.architecture import architecture_spec
from rl_stocks.rl.registry.factory import resolved_parameters
from rl_stocks.training.initialization import initialization_spec


@pytest.mark.parametrize('study',['reward_objectives','trend_context'])
def test_reward_report_matches_controls_and_rejects_unplanned_scale(tmp_path,monkeypatch,study):
    protocol_path=Path('configs/experiments')/(study+'.json')
    protocol=json.loads(protocol_path.read_text())
    context=study=='trend_context'
    state={'status':'complete','cleanup_verified':True,'completed':{}}
    paths=[]
    for i,arm in enumerate(protocol['arms']):
        root=tmp_path/arm;run=root/'runs'/'fixture';run.mkdir(parents=True)
        score=.1+.03*i
        m={'algorithm':'ppo','device':'cpu','status':'complete','smoke':False,'seed':11,
           'actual_steps':50000,'requested_steps':50000,'dataset':protocol['arm_config'][arm]['dataset'] if context else protocol['dataset'],
           'source_hash':'same-source','git':{'revision':'same','dirty':''},'simulation':{},'versions':{},
           'resolved_parameters':resolved_parameters('ppo',False,{'n_steps':1000,'batch_size':50}),
           'reward_config':{'kind':'net_log_equity' if context else arm,'scale':1.},'evaluation_every':10000,
           'initialization':initialization_spec('wide'),'start_sampling':{'mode':'legacy_uniform'},
           'observation':observation_spec('execution_full'),'policy_architecture':architecture_spec('native',7),
           'bundle_id':arm,'selected_score':score,'validation_checkpoints':[{'steps':50000,
             'phase':'post_update','score':score,'return_pct':score*100,'change_pp':0.,
             'mean_max_drawdown_pct':-10.,'mean_costs':1.}]}
        (run/'metadata.json').write_text(json.dumps(m));paths.append(run/'metadata.json')
        results={}
        for symbol in ['AAPL','SPY']:
            # Positive holding avoids synthetic declining-cohort attribution.
            results[symbol]={'symbol':symbol,'split':'validation',
                'metrics':{'net_return':score,'max_drawdown':-.1},
                'baselines':{'buy_hold':{'metrics':{'net_return':.2,'max_drawdown':-.2}}}}
        for filename in ['selected-validation.json','validation.json']:(run/filename).write_text(json.dumps(results))
        state['completed'][arm]={'root':str(root)}
    state_path=tmp_path/'state.json';state_path.write_text(json.dumps(state));output=tmp_path/'result.json'
    monkeypatch.setattr(sys,'argv',['report','--protocol',str(protocol_path),
        '--state',str(state_path),'--output',str(output)])
    runpy.run_path('scripts/report_observations.py',run_name='__main__')
    report=json.loads(output.read_text());assert report['screen']['passed']
    assert report['screen']['promoted_candidate']==('full' if context else 'net_initial_equity_change')
    assert not report['objective_achieved']
    changed=json.loads(paths[1].read_text());changed['reward_config']['scale']=10.
    paths[1].write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='context native/reward' if context else 'reward/native'):runpy.run_path('scripts/report_observations.py',run_name='__main__')
