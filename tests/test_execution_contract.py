from dataclasses import asdict
import numpy as np
import pytest
from test_exposure_replay import fixture
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.risk.exposure import execution_contract,validate_execution_contract,filter_exposure


def environment(mode):
    data=fixture()
    return TradingEnv(list(data['episodes'].values()),normalizer_from(data['normalizer']),data['names'],exposure_execution=mode)


def test_training_environment_unit_rule_matches_explicit_execution_and_reward():
    mapped=environment('unit_magnitude');manual=environment('unrestricted')
    np.testing.assert_array_equal(mapped.reset(seed=11)[0],manual.reset(seed=11)[0])
    for _ in range(3):
        raw=np.array([-.25,1.,1.,1.],dtype=np.float32)
        a=mapped.step(raw);b=manual.step(filter_exposure(raw,'unit_magnitude'))
        np.testing.assert_array_equal(a[0],b[0]);assert a[1:4]==b[1:4]
        assert a[4]['equity']==b[4]['equity'] and a[4]['costs']==b[4]['costs']
        assert a[4]['raw_action'][0]==-1 and a[4]['policy_action'][0]==-.25
        assert a[4]['raw_action'][1:]==tuple(raw[1:])


def test_contract_rejects_unknown_modes_fields_and_boolean_version():
    assert validate_execution_contract(execution_contract('unrestricted','unit_magnitude'))['version']==1
    for value in [None,{},dict(version=True,training_exposure_mode='unrestricted',evaluation_exposure_mode='unit_magnitude'),
                  dict(execution_contract(),extra='untrusted'),dict(execution_contract(),evaluation_exposure_mode='short_only')]:
        with pytest.raises(ValueError):validate_execution_contract(value)
    with pytest.raises(ValueError):environment('short_only')


@pytest.mark.parametrize('algorithm',['ppo','recurrent_ppo'])
def test_frozen_unit_execution_contract_preserves_raw_protection_and_memory(tmp_path,algorithm):
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    from rl_stocks.inference.bundle import freeze,FrozenPolicy
    env=environment('unit_magnitude');obs,_=env.reset(seed=11)
    cls=PPO if algorithm=='ppo' else RecurrentPPO
    model=cls('MlpPolicy' if algorithm=='ppo' else 'MlpLstmPolicy',env,n_steps=8,batch_size=8,seed=11,device='cpu')
    model.save(tmp_path/'policy')
    data=dict(schema='fixture',names=env.names,normalizer=asdict(env.normalizer),manifest={'provenance':{'capabilities':{'feed':'fixture'}}})
    metadata=dict(algorithm=algorithm,action_execution=execution_contract('unit_magnitude','unit_magnitude'))
    bundle=freeze(tmp_path/'policy.zip',data,metadata,tmp_path/'frozen');frozen=FrozenPolicy(bundle,'fixture','fixture')
    state=actual_state=None
    for i in range(3):
        if algorithm=='recurrent_ppo':
            raw,state=model.predict(obs,state=state,episode_start=np.array([i==0]),deterministic=True)
            actual,actual_state=frozen.predict_recurrent(obs,state=actual_state,episode_start=i==0)
            for left,right in zip(state,actual_state):np.testing.assert_array_equal(left,right)
        else:raw=model.predict(obs,deterministic=True)[0];actual=frozen.predict(obs)
        np.testing.assert_array_equal(actual,filter_exposure(raw,'unit_magnitude'))
        np.testing.assert_array_equal(actual[1:],raw[1:])
    assert model.num_timesteps==0 and not model.policy.optimizer.state


def test_alignment_arms_share_actual_initial_parameters_without_fitting():
    from test_sac_regularization import Spaces
    from rl_stocks.rl.registry.factory import create
    from rl_stocks.training.recurrent_alignment import modes,overrides,protocol,model_record,validate_record
    records=[]
    for arm in protocol()['arms']:
        model=create('recurrent_ppo',Spaces(),seed=11,smoke=True,overrides=overrides(arm,True))
        record=model_record(model);validate_record(record,arm,True,initial=True);records.append(record)
    assert records[0]==records[1]==records[2]
    assert modes('posthoc_unit')==execution_contract('unrestricted','unit_magnitude')
    assert modes('aligned_unit')==execution_contract('unit_magnitude','unit_magnitude')


def test_unit_environment_passes_gymnasium_and_sb3_checkers_without_fitting():
    from gymnasium.utils.env_checker import check_env as gym_check
    from stable_baselines3.common.env_checker import check_env as sb3_check
    env=environment('unit_magnitude')
    gym_check(env,skip_render_check=True);sb3_check(env)
