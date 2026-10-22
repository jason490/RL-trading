"""Portable untrained tensors/RNG; no learn or optimizer step calls."""
from copy import deepcopy
import hashlib
from io import BytesIO
from pathlib import Path
import numpy as np
import pytest
import torch
from sb3_contrib import RecurrentPPO
from test_hybrid_recurrent import Dummy
from rl_stocks.rl.registry.factory import create
from rl_stocks.training.recurrent_guard import (overrides,model_record,validate_record,
    initialization_entry,initialization_path,apply_shared_initialization,validate_initialization_assets)
from rl_stocks.training.shared_initialization import load_initialization,RNG_KEY


def model(arm='control',seed=11,smoke=True):
    return create('recurrent_ppo',Dummy(),seed=seed,smoke=smoke,overrides=overrides(arm,smoke))


@pytest.mark.parametrize('seed,smoke',[(11,True),(11,False),(23,False),(37,False)])
def test_shared_tensors_rng_and_optimizer_ownership_across_perturbed_initializations(seed,smoke):
    draws=[];hashes=[]
    for arm in ['control','kl_guard']:
        m=model(arm,seed,smoke);owners=[id(p) for g in m.policy.optimizer.param_groups for p in g['params']]
        with torch.no_grad():
            for p in m.policy.parameters():p.fill_(.123 if arm=='control' else -.456)
        torch.manual_seed(999 if arm=='control' else 1000)
        proof=apply_shared_initialization(m,seed,smoke);r=model_record(m)
        validate_record(r,arm,smoke,initial=True,seed=seed)
        assert r['policy_tensor_sha256']==proof['policy_tensor_sha256']
        assert hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()==proof['rng_sha256']
        assert owners==[id(p) for g in m.policy.optimizer.param_groups for p in g['params']]
        assert not m.policy.optimizer.state and m._n_updates==m.num_timesteps==0
        hashes.append(r['policy_tensor_sha256']);draws.append(torch.randn(32))
    assert hashes[0]==hashes[1] and torch.equal(draws[0],draws[1])
    assert validate_initialization_assets(seed,smoke)['policy_tensor_sha256']==hashes[0]


def test_shared_provenance_survives_actual_model_save_load(tmp_path):
    m=model('kl_guard');apply_shared_initialization(m,11,True);before=model_record(m)
    m.save(tmp_path/'policy');loaded=RecurrentPPO.load(tmp_path/'policy',device='cpu')
    assert model_record(loaded)==before
    validate_record(model_record(loaded),'kl_guard',True,initial=True)


@pytest.mark.parametrize('bad',['checksum','shape','dtype','rng','object'])
def test_corruption_rejects_before_model_or_rng_mutation(tmp_path,bad):
    spec=deepcopy(initialization_entry(11,True));original=initialization_path(11,True).read_bytes()
    if bad=='checksum':data=original[:-1]+bytes([original[-1]^1])
    else:
        with np.load(BytesIO(original),allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
        name=next(k for k in arrays if k!=RNG_KEY)
        if bad=='shape':arrays[name]=np.zeros((1,),dtype=arrays[name].dtype)
        if bad=='dtype':arrays[name]=arrays[name].astype(np.float64)
        if bad=='rng':arrays[RNG_KEY][0]^=1
        if bad=='object':arrays[name]=np.array([object()],dtype=object)
        stream=BytesIO();np.savez_compressed(stream,**arrays);data=stream.getvalue()
        spec['sha256']=hashlib.sha256(data).hexdigest()
    path=tmp_path/'bad.npz';path.write_bytes(data);m=model()
    before=model_record(m);rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):load_initialization(m,path,spec)
    assert model_record(m)==before and torch.equal(torch.get_rng_state(),rng)


@pytest.mark.parametrize('bad',['timesteps','epochs','optimizer','twice','version','model_shape'])
def test_only_fresh_compatible_models_can_load(bad):
    m=model();spec=deepcopy(initialization_entry(11,True));path=initialization_path(11,True)
    if bad=='timesteps':m.num_timesteps=1
    if bad=='epochs':m._n_updates=1
    if bad=='optimizer':m.policy.optimizer.state[next(m.policy.parameters())]={'step':torch.tensor(1.)}
    if bad=='twice':apply_shared_initialization(m,11,True)
    if bad=='version':spec['torch_version']='0.0.0'
    if bad=='model_shape':
        from gymnasium import spaces
        class Different(Dummy):observation_space=spaces.Box(-1,1,(22,),dtype=np.float32)
        m=create('recurrent_ppo',Different(),smoke=True,overrides=overrides('control',True))
    with pytest.raises(ValueError):load_initialization(m,path,spec)
