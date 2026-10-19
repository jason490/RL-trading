from copy import deepcopy
from dataclasses import asdict
import numpy as np
import pytest
from test_core import bar
from test_causal_sampling import episode
from rl_stocks.environment.execution.simulator import Simulator
from rl_stocks.environment.observations.state import observation_spec, portfolio_state
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.data.generation.datasets import Normalizer


def environment(profile):
    return TradingEnv([episode()], Normalizer(('fixture',),(0.,),(1.,)), ['fixture'],
                      random_start=True, episode_steps=16, observation_profile=profile)


def test_same_legacy_inputs_hide_stop_state_full_profile_separates_paths():
    tight=Simulator()
    tight.portfolio.cash=50000.;tight.portfolio.quantity=500;tight.portfolio.basis=100.
    tight.protection=(99.,110.,120.);tight.last_protective_action=(1.,1.,1.)
    wide=deepcopy(tight);wide.protection=(90.,110.,120.)
    for profile in ('legacy','execution_masked'):
        np.testing.assert_array_equal(portfolio_state(tight,100.,2.,profile),portfolio_state(wide,100.,2.,profile))
    assert portfolio_state(tight,100.,2.,'execution_full')!=portfolio_state(wide,100.,2.,'execution_full')
    previous=bar('2020-01-02',op=100,high=101,low=99,close=100)
    following=bar('2020-01-03',op=96,high=99,low=95,close=98)
    results=[sim.advance([.5,1.,1.,1.],previous,following,2.) for sim in (tight,wide)]
    assert results[0]['quantity']==0 and results[1]['quantity']>0
    assert results[0]['equity']!=results[1]['equity']


def test_profiles_preserve_legacy_prefix_rng_accounting_and_rewards():
    environments=[environment(p) for p in ('legacy','execution_masked','execution_full')]
    values=[e.reset(seed=23)[0] for e in environments]
    assert [len(v) for v in values]==[6,15,15]
    for _ in range(5):
        for value in values[1:]:np.testing.assert_array_equal(value[:6],values[0])
        np.testing.assert_array_equal(values[1][6:],np.zeros(9))
        assert all(e.np_random.bit_generator.state==environments[0].np_random.bit_generator.state for e in environments)
        results=[e.step([-.5,1.,1.,1.]) for e in environments]
        assert all(r[1:]==results[0][1:] for r in results)
        values=[r[0] for r in results]


def test_direction_normalization_partial_flags_flat_state_and_invalid_inputs():
    sim=Simulator();sim.portfolio.quantity=-500;sim.portfolio.cash=150000.;sim.portfolio.basis=100.
    sim.protection=(104.,96.,92.);sim.last_protective_action=(-1.,.5,1.);sim.near_done=True
    state=portfolio_state(sim,100.,2.,'execution_full')
    np.testing.assert_allclose(state[5:8],[-.1,.1,.2])
    assert state[8:13]==[1.,1.,-1.,.5,1.]
    # A split adjusts both held prices and known ATR, preserving all state values.
    split=deepcopy(sim);split.portfolio.quantity*=2;split.portfolio.basis/=2
    split.protection=tuple(x/2 for x in split.protection)
    split_state=portfolio_state(split,50.,1.,'execution_full')
    np.testing.assert_array_equal(state[:-1],split_state[:-1])
    assert split_state[-1]==state[-1]/2
    sim.trade(0,100.,'2020-01-01','close')
    assert portfolio_state(sim,100.,2.,'execution_full')[5:13]==[0.]*8
    for atr in (0.,-1.,float('nan'),None):
        with pytest.raises(ValueError):portfolio_state(sim,100.,atr,'execution_full')
    with pytest.raises(ValueError):observation_spec('guessed')


def test_actual_gymnasium_and_sb3_profile_checkers():
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check
    for profile in ('legacy','execution_masked','execution_full'):
        env=environment(profile)
        check_env(env,skip_render_check=True);sb3_check(env);env.close()


@pytest.mark.parametrize('profile',['legacy','execution_masked','execution_full'])
def test_frozen_profile_schema_prediction_parity_without_training(tmp_path,profile):
    import json
    from stable_baselines3 import PPO
    from rl_stocks.core.contracts import digest,canonical
    from rl_stocks.inference.bundle import freeze,FrozenPolicy,checksum
    env=environment(profile);obs,_=env.reset(seed=11)
    model=PPO('MlpPolicy',env,n_steps=8,batch_size=8,seed=11,device='cpu')
    model.save(tmp_path/'weights')
    dataset={'schema':'fixture-v1','names':['fixture'],
             'normalizer':asdict(env.normalizer),
             'manifest':{'provenance':{'capabilities':{'feed':'fixture'}}}}
    metadata={'algorithm':'ppo'}
    # Absent metadata is the historical legacy bundle contract.
    if profile!='legacy':metadata['observation']=observation_spec(profile)
    path=freeze(tmp_path/'weights.zip',dataset,metadata,tmp_path/'frozen')
    policy=FrozenPolicy(path,'fixture-v1','fixture')
    np.testing.assert_array_equal(policy.predict(obs),model.predict(obs,deterministic=True)[0])
    for bad in (obs[:-1],np.append(obs,0.),np.full_like(obs,np.nan)):
        with pytest.raises(ValueError):policy.predict(bad)
    manifest=json.loads((path/'manifest.json').read_text())
    manifest['metadata']['observation']=observation_spec('execution_full' if profile=='legacy' else 'legacy')
    (path/'manifest.json').write_bytes(canonical(manifest))
    hashes=json.loads((path/'checksums.json').read_text());hashes['manifest.json']=checksum(path/'manifest.json')
    (path/'checksums.json').write_bytes(canonical(hashes))
    renamed=path.with_name(digest(manifest));path.rename(renamed)
    with pytest.raises(ValueError,match='model observation schema'):FrozenPolicy(renamed,'fixture-v1','fixture')


def test_remote_observation_plans_reject_budget_and_schema_drift():
    from rl_stocks.runtime.colab.jobs import observation_plan, validate_plan
    for profile in ('legacy','execution_masked','execution_full'):
        for smoke in (True,False):
            plan=observation_plan(profile,smoke=smoke)
            assert validate_plan(plan)==plan
            for key,value in [('steps',100000),('observation_profile','guessed'),('device','cuda'),('start_sampling','ready_uniform')]:
                bad=deepcopy(plan);bad['jobs'][0][key]=value
                with pytest.raises(ValueError):validate_plan(bad)
    with pytest.raises(ValueError):observation_plan('execution_full',True)


def test_observation_promotion_requires_both_controls():
    import json,runpy
    from pathlib import Path
    module=runpy.run_path('scripts/report_observations.py')
    protocol=json.loads(Path('configs/experiments/execution_observations.json').read_text())
    def cell(score):
        point={'score':score,'mean_max_drawdown':-.2,
               'groups':{g:{'net_return':score} for g in ('stock','etf')}}
        return {'selected':deepcopy(point),'final':deepcopy(point)}
    cells={'legacy':cell(.10),'execution_masked':cell(.05),'execution_full':cell(.08)}
    assert not module['dual_screen'](cells,protocol)['passed']
    cells['execution_full']=cell(.13)
    assert module['dual_screen'](cells,protocol)['passed']
    cells['execution_full']['final']['groups']['etf']['net_return']=.09
    assert not module['dual_screen'](cells,protocol)['passed']
