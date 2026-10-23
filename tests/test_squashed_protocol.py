"""Fresh model construction, fixed tensors/RNG and squashed contracts; no learning."""
from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from gymnasium import Env,spaces
from sb3_contrib import RecurrentPPO
from rl_stocks.rl.registry.factory import create
from rl_stocks.training.squashed_actions import (protocol,overrides,model_record,apply_initialization,
    validate_initialization_assets,validate_record,validate_initial_records,runtime_projection,make_runtime_profile,validate_runtime_profile,PROTOCOL_SHA256)


class Dummy(Env):
    observation_space=spaces.Box(-np.inf,np.inf,(21,),dtype=np.float32)
    action_space=spaces.Box(-1,1,(4,),dtype=np.float32)


def model(arm,seed=11,smoke=True):
    return create(protocol()['algorithms'][arm],Dummy(),seed=seed,device='cpu',smoke=smoke,overrides=overrides(arm,smoke))


@pytest.mark.parametrize('seed,smoke',[(11,True),(11,False),(23,False),(37,False)])
def test_full_width_models_restore_exact_per_arm_starts_and_common_sampling_rng(seed,smoke):
    torch.set_num_threads(1);records={};draws={}
    for arm in protocol()['arms']:
        m=model(arm,seed,smoke)
        with torch.no_grad():
            for p in m.policy.parameters():p.fill_(.123)
        torch.manual_seed(999);apply_initialization(m,arm,seed,smoke);r=model_record(m)
        validate_record(r,arm,smoke,initial=True,seed=seed)
        assert r['optimizer']['parameter_count']==protocol()['parameter_count']
        assert r['actor_lstm']['hidden_size']==64
        assert validate_initialization_assets(arm,seed,smoke)['policy_tensor_sha256']==r['policy_tensor_sha256']
        assert not m.policy.optimizer.state and m.num_timesteps==m._n_updates==0
        records[arm]=r;draws[arm]=torch.randn(20)
    assert torch.equal(draws['control'],draws['squashed'])
    assert records['control']['policy_tensor_sha256']==records['squashed']['policy_tensor_sha256']
    assert validate_initial_records(records,smoke,seed)==dict(pinned_per_arm_tensors=True,sampling_rng_identical=True,cross_arm_tensors_identical=True)


def test_actual_larger_model_save_load_preserves_architecture_and_provenance(tmp_path):
    m=model('squashed');apply_initialization(m,'squashed',11,True);before=model_record(m)
    m.save(tmp_path/'untrained');loaded=RecurrentPPO.load(tmp_path/'untrained',device='cpu')
    assert model_record(loaded)==before
    validate_record(model_record(loaded),'squashed',True,initial=True)


@pytest.mark.parametrize('field,value',[('observation_size',25),('target_kl',.01),('clip_range',.3),
    ('num_timesteps',True),('shared_initialization',{}),('policy_tensor_sha256','0'*64)])
def test_shape_guard_steps_and_wrong_initial_tensor_proof_reject(field,value):
    m=model('squashed');apply_initialization(m,'squashed',11,True);r=deepcopy(model_record(m));r[field]=value
    with pytest.raises(ValueError):validate_record(r,'squashed',True,initial=True)


def test_runtime_projection_blocks_infeasible_allocation_without_resizing_or_new_hardware():
    assert runtime_projection(dict(control=10.,squashed=10.))['screen_allocation_allowed']
    assert not runtime_projection(dict(control=10.,squashed=100.))['screen_allocation_allowed']
    for values in [dict(control=0.,squashed=10.),dict(control=10.,squashed=float('nan')),dict(control=True,squashed=10.),dict(squashed=10.)]:
        with pytest.raises(ValueError):runtime_projection(values)
    assert overrides('control',True)['policy_kwargs']['net_arch']==[64,64]
    assert overrides('squashed',True)['policy_kwargs']['net_arch']==[64,64]
    p=protocol();assert p['dataset']=='cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b'
    assert p['execution']['minimum_account_balance']==150 and p['execution']['account_teardown_balance']==170
    assert p['steps_per_cell']==100000 and p['screen_gate']['minimum_selected_and_final_gain_pp_vs_each_control']==2.
    with pytest.raises(ValueError):overrides('other')
    with pytest.raises(ValueError):overrides('control',1)


def test_runtime_profile_binds_actual_acceptance_bytes_and_rejects_stale_or_failed_evidence(tmp_path):
    p=tmp_path/'.runtime/squashed-smoke-acceptance.json';p.parent.mkdir()
    fixture=dict(passed=True,protocol_sha256=PROTOCOL_SHA256,cleanup_verified=True,actual_sampling_equal=True,
        initialization_checks=dict(sampling_rng_identical=True),records={
            'control':dict(actual_saved_models_restored=True,duration_seconds=10.),
            'squashed':dict(actual_saved_models_restored=True,duration_seconds=12.)})
    p.write_text(json.dumps(fixture));profile=make_runtime_profile(tmp_path)
    assert validate_runtime_profile(profile,tmp_path)==profile
    fixture['records']['squashed']['duration_seconds']=13.;p.write_text(json.dumps(fixture))
    with pytest.raises(ValueError):validate_runtime_profile(profile,tmp_path)
    fixture['cleanup_verified']=False;p.write_text(json.dumps(fixture))
    with pytest.raises(ValueError):make_runtime_profile(tmp_path)
    fixture['cleanup_verified']=True;fixture['records']['squashed']['duration_seconds']=1000.;p.write_text(json.dumps(fixture))
    with pytest.raises(ValueError):make_runtime_profile(tmp_path)


@pytest.mark.parametrize('field,value',[('defaults_decoupled_weight_decay',0),('group_decoupled_weight_decay',[0]),('version',True)])
def test_coupling_evidence_requires_actual_boolean_flags(field,value):
    m=model('squashed');apply_initialization(m,'squashed',11,True);r=model_record(m)
    r['coupled_adam'][field]=value
    with pytest.raises(ValueError,match='coupled Adam'):validate_record(r,'squashed',True,initial=True)


@pytest.mark.parametrize('field,value',[('use_sde',True),('squash_output',False),('policy_class','RecurrentActorCriticPolicy')])
def test_actual_distribution_identity_cannot_be_relabelled(field,value):
    m=model('squashed');apply_initialization(m,'squashed',11,True);r=model_record(m);r[field]=value
    with pytest.raises(ValueError):validate_record(r,'squashed',True,initial=True)


def test_distribution_version_requires_exact_integer():
    m=model('squashed');apply_initialization(m,'squashed',11,True);r=model_record(m)
    r['action_distribution']['version']=True
    with pytest.raises(ValueError):validate_record(r,'squashed',True,initial=True)
