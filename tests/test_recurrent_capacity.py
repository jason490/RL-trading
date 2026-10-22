"""Fresh model construction, fixed tensors/RNG and capacity contracts; no learning."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from gymnasium import Env,spaces
from sb3_contrib import RecurrentPPO
from rl_stocks.rl.registry.factory import create
from rl_stocks.training.recurrent_capacity import (protocol,overrides,model_record,apply_initialization,
    validate_initialization_assets,validate_record,validate_initial_records,runtime_projection)


class Dummy(Env):
    observation_space=spaces.Box(-np.inf,np.inf,(21,),dtype=np.float32)
    action_space=spaces.Box(-1,1,(4,),dtype=np.float32)


def model(arm,seed=11,smoke=True):
    return create('recurrent_ppo',Dummy(),seed=seed,device='cpu',smoke=smoke,overrides=overrides(arm,smoke))


@pytest.mark.parametrize('seed,smoke',[(11,True),(11,False),(23,False),(37,False)])
def test_full_width_models_restore_exact_per_arm_starts_and_common_sampling_rng(seed,smoke):
    torch.set_num_threads(1);records={};draws={}
    for arm in protocol()['arms']:
        m=model(arm,seed,smoke)
        with torch.no_grad():
            for p in m.policy.parameters():p.fill_(.123)
        torch.manual_seed(999);apply_initialization(m,arm,seed,smoke);r=model_record(m)
        validate_record(r,arm,smoke,initial=True,seed=seed)
        assert r['optimizer']['parameter_count']==protocol()['parameter_counts'][arm]
        assert r['actor_lstm']['hidden_size']==protocol()['widths'][arm]
        assert validate_initialization_assets(arm,seed,smoke)['policy_tensor_sha256']==r['policy_tensor_sha256']
        assert not m.policy.optimizer.state and m.num_timesteps==m._n_updates==0
        records[arm]=r;draws[arm]=torch.randn(20)
    assert torch.equal(draws['control'],draws['larger'])
    assert records['control']['policy_tensor_sha256']!=records['larger']['policy_tensor_sha256']
    assert validate_initial_records(records,smoke,seed)==dict(pinned_per_arm_tensors=True,sampling_rng_identical=True,cross_width_tensors_identical=False)


def test_actual_larger_model_save_load_preserves_architecture_and_provenance(tmp_path):
    m=model('larger');apply_initialization(m,'larger',11,True);before=model_record(m)
    m.save(tmp_path/'untrained');loaded=RecurrentPPO.load(tmp_path/'untrained',device='cpu')
    assert model_record(loaded)==before
    validate_record(model_record(loaded),'larger',True,initial=True)


@pytest.mark.parametrize('field,value',[('observation_size',25),('target_kl',.01),('clip_range',.3),
    ('num_timesteps',True),('shared_initialization',{}),('policy_tensor_sha256','0'*64)])
def test_shape_guard_steps_and_wrong_initial_tensor_proof_reject(field,value):
    m=model('larger');apply_initialization(m,'larger',11,True);r=deepcopy(model_record(m));r[field]=value
    with pytest.raises(ValueError):validate_record(r,'larger',True,initial=True)


def test_runtime_projection_blocks_infeasible_allocation_without_resizing_or_new_hardware():
    assert runtime_projection(dict(control=10.,larger=10.))['screen_allocation_allowed']
    assert not runtime_projection(dict(control=10.,larger=100.))['screen_allocation_allowed']
    for values in [dict(control=0.,larger=10.),dict(control=10.,larger=float('nan')),dict(control=True,larger=10.),dict(larger=10.)]:
        with pytest.raises(ValueError):runtime_projection(values)
    assert overrides('control',True)['policy_kwargs']['net_arch']==[64,64]
    assert overrides('larger',True)['policy_kwargs']['net_arch']==[128,128]
    p=protocol();assert p['dataset']=='cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b'
    assert p['execution']['minimum_account_balance']==150 and p['execution']['account_teardown_balance']==170
    assert p['steps_per_cell']==100000 and p['screen_gate']['minimum_selected_and_final_gain_pp_vs_each_control']==2.
    with pytest.raises(ValueError):overrides('other')
    with pytest.raises(ValueError):overrides('control',1)
