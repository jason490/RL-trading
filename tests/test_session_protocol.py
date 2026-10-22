from copy import deepcopy
import numpy as np
import pytest
import torch
from gymnasium import Env,spaces
from sb3_contrib import RecurrentPPO
from rl_stocks.rl.registry.factory import create
from rl_stocks.training.session_context import (protocol,arm_spec,overrides,modes,model_record,
    apply_initialization,validate_initialization_assets,validate_record,validate_initial_pairs)


def model(arm,seed=11,smoke=True):
    class Dummy(Env):
        observation_space=spaces.Box(-np.inf,np.inf,(arm_spec(arm)['observation_size'],),dtype=np.float32)
        action_space=spaces.Box(-1,1,(4,),dtype=np.float32)
    return create('recurrent_ppo',Dummy(),seed=seed,smoke=smoke,overrides=overrides(arm,smoke))


@pytest.mark.parametrize('seed,smoke',[(11,True),(11,False),(23,False),(37,False)])
def test_all_three_arms_restore_pinned_untrained_tensors_and_cpu_rng(seed,smoke):
    records={};draws={}
    for arm in protocol()['arms']:
        m=model(arm,seed,smoke)
        with torch.no_grad():
            for p in m.policy.parameters():p.fill_(.123)
        torch.manual_seed(999)
        apply_initialization(m,arm,seed,smoke)
        r=model_record(m);validate_record(r,arm,smoke,initial=True,seed=seed)
        assert not m.policy.optimizer.state and m._n_updates==m.num_timesteps==0
        assert validate_initialization_assets(arm,seed,smoke)['policy_tensor_sha256']==r['policy_tensor_sha256']
        records[arm]=r;draws[arm]=torch.randn(20)
    assert validate_initial_pairs(records,smoke,seed)
    assert torch.equal(draws['masked'],draws['full'])
    assert records['legacy']['observation_size']==21 and records['full']['observation_size']==25


def test_actual_save_load_retains_initialization_and_model_settings(tmp_path):
    m=model('full');apply_initialization(m,'full',11,True);r=model_record(m)
    m.save(tmp_path/'untrained');loaded=RecurrentPPO.load(tmp_path/'untrained',device='cpu')
    assert model_record(loaded)==r
    validate_record(model_record(loaded),'full',True,initial=True)


@pytest.mark.parametrize('field,value',[('observation_size',21),('target_kl',.01),('clip_range',.3),
    ('num_timesteps',True),('shared_initialization',{}),('policy_tensor_sha256','0'*64)])
def test_actual_contract_rejects_wrong_shape_guard_steps_or_initialization(field,value):
    m=model('full');apply_initialization(m,'full',11,True);r=deepcopy(model_record(m));r[field]=value
    with pytest.raises(ValueError):validate_record(r,'full',True,initial=True)


def test_fixed_controls_and_budget_gates_remain_strict():
    s=protocol()
    assert s['arms']==['legacy','masked','full'] and s['candidate']=='full'
    assert s['steps_per_cell']==100000 and s['screen_gate']['minimum_selected_and_final_gain_pp_vs_each_control']==2
    assert s['execution']['minimum_account_balance']==150 and s['execution']['account_teardown_balance']==170
    assert s['execution']['project_ceiling_units']==150 and s['execution']['working_stop_units']==130
    assert modes('legacy')==modes('masked')==modes('full')
    with pytest.raises(ValueError):overrides('other')
    with pytest.raises(ValueError):overrides('full',1)
