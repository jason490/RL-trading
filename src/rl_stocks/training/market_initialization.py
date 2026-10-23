"""Untrained common-column mapping for the fixed compact-market comparison."""
import json
from pathlib import Path
import numpy as np
import torch
from rl_stocks.core.contracts import digest
from rl_stocks.training.shared_initialization import tensor_fingerprint

ROOT=Path(__file__).resolve().parents[3]
PREREGISTRATION_SHA256='ca987adbfa0355b843cd9e0a545e9695db4b1bc046f5c63d6e83dc85ad0d52c7'
INPUT_KEYS=('lstm_actor.weight_ih_l0','lstm_critic.weight_ih_l0')


def preregistration():
    spec=json.loads((ROOT/'configs/experiments/market_context_preregistration.json').read_text())
    if digest(spec)!=PREREGISTRATION_SHA256:raise ValueError('fixed market-context preregistration changed')
    return spec


def parameters(smoke=False):
    if type(smoke) is not bool:raise ValueError('explicit smoke boolean required')
    spec=preregistration();params=dict(spec['parameters']);params.pop('net_arch')
    if smoke:params.update(spec['smoke_parameters'])
    return dict(params,target_kl=None,policy_kwargs=dict(spec['recurrent_parameters'],net_arch=[64,64],
        optimizer_kwargs=dict(eps=1e-5,weight_decay=0.)))


def column_fingerprint(arrays):
    return tensor_fingerprint({k:arrays[k][:,7:10] for k in INPUT_KEYS})


def validate_mapping(original,augmented,new_columns_sha256):
    if set(original)!=set(augmented) or set(INPUT_KEYS)-set(original):raise ValueError('exact recurrent tensor keys required')
    for key,old in original.items():
        new=augmented[key]
        if old.dtype!=new.dtype or not np.isfinite(old).all() or not np.isfinite(new).all():raise ValueError('finite matching tensor dtype required')
        if key in INPUT_KEYS:
            if old.shape!=(256,21) or new.shape!=(256,24) or not np.array_equal(old[:,:7],new[:,:7]) or not np.array_equal(old[:,7:],new[:,10:]):
                raise ValueError('original market/account LSTM columns must remain exact')
        elif not np.array_equal(old,new):raise ValueError('all common policy tensors must remain exact')
    if column_fingerprint(augmented)!=new_columns_sha256:raise ValueError('new input columns changed after native generation')
    if sum(a.size for a in original.values())!=61513 or sum(a.size for a in augmented.values())!=63049:
        raise ValueError('fixed legacy/augmented parameter counts required')
    return dict(common_tensors_exact=True,original_input_columns_exact=True,new_columns_sha256=new_columns_sha256,
        legacy_parameters=61513,augmented_parameters=63049)


def map_initial_tensors(model,original):
    if (model.num_timesteps!=0 or model._n_updates!=0 or model.policy.optimizer.state
            or getattr(model,'_research_initialization',None) is not None
            or any(p.device.type!='cpu' for p in model.policy.parameters())):
        raise ValueError('fresh untrained model and empty optimizer required')
    state=model.policy.state_dict();draw={k:v.detach().cpu().numpy().copy() for k,v in state.items()}
    new_sha=column_fingerprint(draw);draw_sha=tensor_fingerprint(draw)
    if set(draw)!=set(original):raise ValueError('same policy tensor names required')
    mapped={}
    for key,new in draw.items():
        old=original[key]
        if key in INPUT_KEYS:
            if old.shape!=(256,21) or new.shape!=(256,24) or old.dtype!=new.dtype:raise ValueError('fixed input column mapping required')
            new[:,:7]=old[:,:7];new[:,10:]=old[:,7:];mapped[key]=new
        else:mapped[key]=old.copy()
    proof=validate_mapping(original,mapped,new_sha)
    model.policy.load_state_dict({k:torch.from_numpy(v) for k,v in mapped.items()},strict=True)
    actual={k:v.detach().cpu().numpy() for k,v in model.policy.state_dict().items()}
    validate_mapping(original,actual,new_sha)
    return dict(proof,native_draw_sha256=draw_sha)
