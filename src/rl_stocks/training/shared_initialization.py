"""Checksum-pinned untrained policy tensors and CPU RNG, never fitted weights."""
import hashlib
from io import BytesIO
from pathlib import Path
import numpy as np
import torch

RNG_KEY='__torch_rng_state__'


def tensor_fingerprint(arrays):
    h=hashlib.sha256()
    for name,array in sorted(arrays.items()):
        h.update(name.encode());h.update(str(tuple(array.shape)).encode());h.update(array.tobytes())
    return h.hexdigest()


def _untrained(model):
    optimizer=getattr(model.policy,'optimizer',None)
    if optimizer is not None:
        optimizers=[optimizer]
    else:
        optimizers=[getattr(getattr(model,name,None),'optimizer',None) for name in ('actor','critic')]
        entropy=getattr(model,'ent_coef_optimizer',None)
        if entropy is not None:
            optimizers.append(entropy)
    if (model.num_timesteps!=0 or model._n_updates!=0
            or any(o is None or o.state for o in optimizers)):
        raise ValueError('untrained policy and empty optimizer required')
    replay=getattr(model,'replay_buffer',None)
    if replay is not None and (replay.pos!=0 or replay.full):
        raise ValueError('fresh empty replay required')
    if any(p.device.type!='cpu' for p in model.policy.parameters()):raise ValueError('CPU initialization required')
    if getattr(model,'_research_initialization',None) is not None:raise ValueError('initialization already applied')


def write_initialization(model,path):
    _untrained(model);path=Path(path)
    arrays={k:v.detach().cpu().numpy().copy() for k,v in model.policy.state_dict().items()}
    if any(not np.isfinite(a).all() for a in arrays.values()):raise ValueError('finite initial tensors required')
    rng=torch.get_rng_state().numpy().copy()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:np.savez_compressed(stream,**arrays,**{RNG_KEY:rng})
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),policy_tensor_sha256=tensor_fingerprint(arrays),
        rng_sha256=hashlib.sha256(rng.tobytes()).hexdigest(),rng_bytes=rng.size,
        torch_version=torch.__version__.split('+')[0],
        tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()})


def read_initialization(data,spec):
    if hashlib.sha256(data).hexdigest()!=spec['sha256']:raise ValueError('initialization checksum mismatch')
    with np.load(BytesIO(data),allow_pickle=False) as archive:
        if set(archive.files)!={*spec['tensors'],RNG_KEY} or len(archive.files)!=len(set(archive.files)):
            raise ValueError('initialization tensor schema mismatch')
        arrays={k:archive[k].copy() for k in spec['tensors']};rng=archive[RNG_KEY].copy()
    for name,value in arrays.items():
        schema=spec['tensors'][name]
        if list(value.shape)!=schema['shape'] or str(value.dtype)!=schema['dtype'] or not np.isfinite(value).all():
            raise ValueError('initialization shape/dtype/finite mismatch')
    if (rng.dtype!=np.uint8 or rng.ndim!=1 or rng.size!=spec['rng_bytes']
            or hashlib.sha256(rng.tobytes()).hexdigest()!=spec['rng_sha256']
            or tensor_fingerprint(arrays)!=spec['policy_tensor_sha256']):
        raise ValueError('initialization tensor/RNG identity mismatch')
    return arrays,rng


def load_initialization(model,path,spec):
    _untrained(model)
    if torch.__version__.split('+')[0]!=spec['torch_version']:raise ValueError('initialization Torch version mismatch')
    arrays,rng=read_initialization(Path(path).read_bytes(),spec)
    current=model.policy.state_dict()
    if (set(current)!=set(arrays) or any(tuple(current[k].shape)!=arrays[k].shape
            or current[k].detach().cpu().numpy().dtype!=arrays[k].dtype for k in current)
            or tuple(torch.get_rng_state().shape)!=rng.shape):
        raise ValueError('initialization incompatible with current model/RNG')
    try:torch.Generator(device='cpu').set_state(torch.from_numpy(rng))
    except RuntimeError as exc:raise ValueError('invalid CPU RNG state') from exc
    # All validation precedes mutation; load_state_dict copies into existing
    # Parameter objects, preserving optimizer ownership and an empty state.
    model.policy.load_state_dict({k:torch.from_numpy(v) for k,v in arrays.items()},strict=True)
    torch.set_rng_state(torch.from_numpy(rng))
    actual={k:v.detach().cpu().numpy() for k,v in model.policy.state_dict().items()}
    if (tensor_fingerprint(actual)!=spec['policy_tensor_sha256']
            or hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()!=spec['rng_sha256']):
        raise ValueError('initial tensors/RNG failed exact restoration')
    return dict(version=1,artifact_sha256=spec['sha256'],policy_tensor_sha256=spec['policy_tensor_sha256'],
                rng_sha256=spec['rng_sha256'],torch_version=spec['torch_version'],generated_without_fitting=True)
