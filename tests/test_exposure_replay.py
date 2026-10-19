from copy import deepcopy
from dataclasses import asdict
import numpy as np
import pytest
from test_core import bar
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.evaluation.exposure import filter_exposure
from rl_stocks.evaluation.runner import evaluate


def fixture():
    bars=[bar('2021-01-04',close=100),bar('2021-01-05',op=100,high=101,low=97,close=98),
          bar('2021-01-06',op=98,high=99,low=95,close=96),bar('2021-01-07',op=96,high=97,low=93,close=94)]
    ep={'split':'validation','symbol':'SPY','bars':[asdict(b) for b in bars],
        'features':[[0.]]*len(bars),'atr':[2.]*len(bars)}
    return {'episodes':{'SPY/validation':ep},'names':['fixture'],
            'normalizer':asdict(Normalizer(('fixture',),(0.,),(1.,)))}


def test_filter_changes_only_exposure_and_never_mutates_policy_action():
    for exposure in (-.6,0.,.6):
        action=np.array([exposure,.4,-.2,.8],dtype=np.float32);before=action.copy()
        for mode in ('unrestricted','long_only','short_only'):
            result=filter_exposure(action,mode)
            np.testing.assert_array_equal(result[1:],action[1:])
            assert result.dtype==action.dtype and not np.shares_memory(result,action)
            expected=action[0] if mode=='unrestricted' else max(0,action[0]) if mode=='long_only' else min(0,action[0])
            assert result[0]==expected
        np.testing.assert_array_equal(action,before)
    for action in ([0,0,0],[np.nan,0,0,0],[2.,0,0,0]):
        with pytest.raises(ValueError):filter_exposure(action,'short_only')
    with pytest.raises(ValueError):filter_exposure([0]*4,'future_decline')


class AccountAware:
    def __init__(self):self.seen=[]
    def predict(self,obs,deterministic=True):
        self.seen.append(obs.copy())
        return np.array([-.5 if obs[1]>=0 else .5,1.,1.,1.],dtype=np.float32),None


def test_replay_recomputes_state_and_profit_instead_of_subtracting_attribution():
    dataset=fixture();before=deepcopy(dataset)
    ordinary,short,long=AccountAware(),AccountAware(),AccountAware()
    reference=evaluate(ordinary,dataset)['SPY/validation']
    restricted=evaluate(short,dataset,exposure_mode='short_only')['SPY/validation']
    flat=evaluate(long,dataset,exposure_mode='long_only')['SPY/validation']
    assert dataset==before
    assert all(r['quantity']<=0 for r in restricted['rows'])
    assert restricted['metrics']['net_return']>0 and restricted['directional']['short_net_pnl']>0
    assert restricted['directional']['long_net_pnl']==0
    assert all(r['quantity']==0 for r in flat['rows'])
    assert flat['metrics']['net_return']==0
    # A changed path is fed back into the SAME frozen decision rule.
    assert not np.array_equal(ordinary.seen[1],long.seen[1])
    assert reference['directional']['short_net_pnl']!=restricted['directional']['short_net_pnl']
    for result in (reference,restricted,flat):
        d=result['directional'];assert abs(d['long_net_pnl']+d['short_net_pnl']-100000*result['metrics']['net_return'])<1e-5


def test_default_replay_keeps_original_path_and_rejects_unknown_mode_before_prediction():
    dataset=fixture();a=evaluate(AccountAware(),dataset);b=evaluate(AccountAware(),dataset,exposure_mode='unrestricted')
    assert a==b
    with pytest.raises(ValueError):evaluate(AccountAware(),dataset,exposure_mode='selected_decliners')


def test_reference_parity_rejects_wrong_path_even_with_equal_terminal_metrics():
    import runpy
    verify=runpy.run_path('scripts/replay_directions.py')['verify_reference']
    results=evaluate(AccountAware(),fixture())
    tolerance={'equity_dollars':.1,'action_absolute':2e-6,'return_absolute':1e-6,'cost_dollars':.01}
    assert verify(results,deepcopy(results),tolerance)['exact_report_digest']
    for field,value in (('quantity',123),('equity',50000.),('session','2030-01-01'),('raw_action',(0.,0.,0.,0.))):
        altered=deepcopy(results);altered['SPY/validation']['rows'][0][field]=value
        with pytest.raises((ValueError,AssertionError)):verify(results,altered,tolerance)
    altered=deepcopy(results);altered['SPY/validation']['rows'][0]['equity']+=.001
    assert verify(results,altered,tolerance)['passed']
    assert not verify(results,altered,tolerance)['exact_report_digest']


def test_policy_fingerprint_requires_identical_tensors_and_spaces_without_training(tmp_path):
    import runpy,torch
    from stable_baselines3 import PPO
    from rl_stocks.environment.trading.env import TradingEnv
    fingerprint=runpy.run_path('scripts/replay_directions.py')['policy_fingerprint']
    data=fixture();env=TradingEnv(list(data['episodes'].values()),Normalizer(('fixture',),(0.,),(1.,)),data['names'])
    model=PPO('MlpPolicy',env,n_steps=8,batch_size=8,seed=11,device='cpu')
    model.save(tmp_path/'weights')
    loaded=PPO.load(tmp_path/'weights.zip',device='cpu')
    original=fingerprint(model);assert original==fingerprint(loaded)
    with torch.no_grad():next(loaded.policy.parameters()).add_(.001)
    assert fingerprint(loaded)!=original


def test_filtered_replay_retains_original_risk_outputs_and_reports_blocking():
    import runpy
    stats=runpy.run_path('scripts/replay_directions.py')['action_statistics']
    results=evaluate(AccountAware(),fixture(),exposure_mode='short_only')
    rows=results['SPY/validation']['rows']
    assert any(row['policy_action'][0]>0 and row['raw_action'][0]==0 for row in rows)
    for row in rows:
        assert row['policy_action'][1:]==row['raw_action'][1:]
        assert row['exposure_mode']=='short_only'
    summary=stats(results)
    assert summary['blocked_signal_fraction']>0 and summary['end_long_fraction']==0
    assert summary['requested_long_fraction']>0 and summary['end_short_fraction']>0


def test_replay_protocol_bounds_existing_confirmation_seeds_and_profile_subsets():
    import json,runpy
    from pathlib import Path
    validate=runpy.run_path('scripts/replay_directions.py')['validate_protocol']
    p=json.loads(Path('configs/experiments/direction_replay.json').read_text());validate(p)
    for seed in (23,37):
        q={**p,'source_seed':seed,'profiles':['legacy'],'maximum_cells':6};validate(q)
        for key,value in (('source_seed',True),('source_seed',19),('profiles',['legacy','legacy']),('maximum_cells',18),('source_initialization','standard')):
            with pytest.raises(ValueError):validate({**q,key:value})


def test_reference_diagnostics_measure_discrete_drift_without_relaxing_gate():
    import runpy
    module=runpy.run_path('scripts/replay_directions.py');results=evaluate(AccountAware(),fixture())
    altered=deepcopy(results);row=altered['SPY/validation']['rows'][0];row['quantity']+=1;row['equity']+=.01
    d=module['reference_differences'](altered,results)
    assert d['quantity_mismatch_rows']==1 and d['maximum_quantity_difference']==1
    assert d['session_mismatch_rows']==0 and d['maximum_action_difference']==0
    assert d['quantity_examples'][0]['session']==row['session']
    with pytest.raises(ValueError):module['verify_reference'](altered,results,{'equity_dollars':.1,'action_absolute':2e-6,'return_absolute':1e-6,'cost_dollars':.01})


def test_declared_one_share_bridge_is_bounded_and_cannot_change_position_sign():
    import runpy
    verify=runpy.run_path('scripts/replay_directions.py')['verify_reference']
    results=evaluate(AccountAware(),fixture());altered=deepcopy(results)
    tolerance={'equity_dollars':2.,'action_absolute':.001,'return_absolute':2e-5,'cost_dollars':.05,
               'quantity_absolute':1,'quantity_mismatch_rows':10}
    altered['SPY/validation']['rows'][0]['quantity']+=1
    assert verify(altered,results,tolerance)['quantity_mismatch_rows']==1
    assert not verify(altered,results,tolerance)['quantity_path_exact']
    altered['SPY/validation']['rows'][0]['quantity']+=1
    with pytest.raises(ValueError):verify(altered,results,tolerance)
    for key,value in (('quantity_absolute',2),('quantity_mismatch_rows',100),('quantity_absolute',True)):
        with pytest.raises(ValueError):verify(results,results,{**tolerance,key:value})
    base=deepcopy(results);base['SPY/validation']['rows']=[deepcopy(results['SPY/validation']['rows'][0]) for _ in range(11)]
    altered=deepcopy(base)
    for row in altered['SPY/validation']['rows']:row['quantity']+=1
    with pytest.raises(ValueError,match='count exceeded'):verify(altered,base,tolerance)

    base=deepcopy(results);base["SPY/validation"]["rows"][0]["quantity"]=1
    altered=deepcopy(base);altered["SPY/validation"]["rows"][0]["quantity"]=0
    with pytest.raises(ValueError):verify(altered,base,tolerance)
