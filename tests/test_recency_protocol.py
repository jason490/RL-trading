from copy import deepcopy
import pytest
import torch
from test_recurrent_capacity import Dummy
from rl_stocks.rl.registry.factory import create
from rl_stocks.runtime.colab.jobs import recency_plan,validate_plan
from rl_stocks.training.training_history import (protocol,overrides,model_record,apply_initialization,
    validate_record,validate_initial_records,runtime_projection,PROTOCOL_SHA256)


def profile():
    durations=dict(full=10.,recent=12.)
    return dict(version=1,protocol_sha256=PROTOCOL_SHA256,smoke_acceptance_file='.runtime/recency-smoke-acceptance.json',
        smoke_acceptance_sha256='a'*64,durations_seconds=durations,projection=runtime_projection(durations))


@pytest.mark.parametrize('seed,smoke',[(11,True),(11,False),(23,False),(37,False)])
def test_exact_same_untrained_tensors_and_rng_both_sources(seed,smoke):
    torch.set_num_threads(1);records={};draws=[]
    for arm in protocol()['arms']:
        m=create('recurrent_ppo',Dummy(),seed=seed,smoke=smoke,overrides=overrides(arm,smoke))
        apply_initialization(m,arm,seed,smoke);r=model_record(m);validate_record(r,arm,smoke,initial=True,seed=seed)
        assert r['optimizer']['parameter_count']==61513 and not m.policy.optimizer.state
        records[arm]=r;draws.append(torch.randn(10))
    assert torch.equal(*draws)
    assert validate_initial_records(records,smoke,seed)['cross_arm_tensors_identical']


@pytest.mark.parametrize('arm',['full','recent'])
def test_plan_binds_correct_dataset_floor_and_profile(arm):
    p=recency_plan(arm,smoke=True);assert validate_plan(p)==p
    assert p['dataset_contract']['identity']==protocol()['datasets'][arm]
    assert p['account_contract']==dict(minimum_balance=150,teardown_balance=170)
    with pytest.raises(ValueError):recency_plan(arm)
    p=recency_plan(arm,profile=profile());assert validate_plan(p)==p
    for mutate in [lambda p:p['dataset_contract'].update(identity='0'*64),lambda p:p['runtime_acceptance']['projection'].update(projected_seconds=0),lambda p:p['account_contract'].update(minimum_balance=0)]:
        bad=deepcopy(p);mutate(bad)
        with pytest.raises(ValueError):validate_plan(bad)


import json
from rl_stocks.training.training_history import make_runtime_profile,validate_runtime_profile

def test_runtime_profile_binds_actual_acceptance_bytes_and_rejects_stale_or_failed_evidence(tmp_path):
    p=tmp_path/'.runtime/recency-smoke-acceptance.json';p.parent.mkdir()
    fixture=dict(passed=True,protocol_sha256=PROTOCOL_SHA256,cleanup_verified=True,actual_symbol_sampling_equal=True,
        initialization_checks=dict(sampling_rng_identical=True),records={
            'full':dict(actual_saved_models_restored=True,duration_seconds=10.),
            'recent':dict(actual_saved_models_restored=True,duration_seconds=12.)})
    p.write_text(json.dumps(fixture));profile=make_runtime_profile(tmp_path)
    assert validate_runtime_profile(profile,tmp_path)==profile
    fixture['records']['recent']['duration_seconds']=13.;p.write_text(json.dumps(fixture))
    with pytest.raises(ValueError):validate_runtime_profile(profile,tmp_path)
    fixture['cleanup_verified']=False;p.write_text(json.dumps(fixture))
    with pytest.raises(ValueError):make_runtime_profile(tmp_path)
    fixture['cleanup_verified']=True;fixture['records']['recent']['duration_seconds']=1000.;p.write_text(json.dumps(fixture))
    with pytest.raises(ValueError):make_runtime_profile(tmp_path)
