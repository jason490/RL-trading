"""Fresh portable comparison starts, without data access or optimizer updates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
import torch

from rl_stocks.core.contracts import canonical
from rl_stocks.training.crossq_inspection import SpacesOnly
from rl_stocks.training.distributional_critics import capture_global_rng, rng_record, restore_global_rng
from rl_stocks.training.recurrent_comparison import protocol, PROTOCOL_SHA256
from rl_stocks.training.recurrent_sac_checkpoint import save_checkpoint, load_checkpoint, validate_learner
from rl_stocks.training.recurrent_sac_learner import RecurrentSACLearner
from rl_stocks.rl.recurrent_sac.networks import RecurrentSACNetworks
from rl_stocks.training.sac_update_worker import preserved_rng
from rl_stocks.training.shared_initialization import write_initialization, load_initialization, _untrained


def construct(arm, seed, *, env=None):
    """Seed and construct only; caller controls preservation of global streams."""
    spec=protocol()
    if arm not in spec['arms'] or type(seed) is not int or seed not in spec['confirmation_seeds']:
        raise ValueError('registered arm and seed required')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if arm=='gru_sac':
        if env is not None:
            raise ValueError('GRU learner collection is separate from construction')
        return RecurrentSACLearner(RecurrentSACNetworks(21,64),max_updates=99000)
    parameters=deepcopy(spec['arm_contracts'][arm])
    parameters={k:v for k,v in parameters.items() if k not in ('algorithm','observation_profile')
                and not k.startswith('expected_')}
    environment=env if env is not None else SpacesOnly(dict(observation_size=21,action_size=4))
    cls,policy=(SAC,'MlpPolicy') if arm=='native_sac' else (RecurrentPPO,'MlpLstmPolicy')
    return cls(policy,environment,seed=seed,device='cpu',verbose=0,**parameters)


def lineage():
    return dict(source='recurrent-comparison-starts-'+PROTOCOL_SHA256,
                dataset=protocol()['dataset'],observation_schema='execution-full-v1-normalized21')


def build(destination):
    """Publish all nine starts atomically; preserve caller RNG on every exit."""
    destination=Path(destination)
    if destination.exists():
        raise FileExistsError('immutable starts destination exists')
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.comparison-starts-',dir=destination.parent))
    try:
        entries={}
        with preserved_rng():
            for arm in protocol()['arms']:
                for seed in protocol()['confirmation_seeds']:
                    model=construct(arm,seed)
                    name=f'{arm}-{seed}'
                    streams=capture_global_rng()
                    stream_hashes=rng_record()
                    if arm=='gru_sac':
                        save_checkpoint(model,stage/name,lineage=lineage())
                        asset=dict(kind='recurrent_checkpoint',file=name)
                    else:
                        asset=dict(kind='native_tensors',file=name+'.npz',
                                   tensors=write_initialization(model,stage/(name+'.npz')))
                    entries[name]=dict(arm=arm,seed=seed,asset=asset,
                                       global_rng=streams,rng_hashes=stream_hashes)
                    del model
        files={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in stage.rglob('*') if p.is_file()}
        manifest=dict(schema='recurrent-comparison-starts-v1',protocol_sha256=PROTOCOL_SHA256,
                      torch_version=torch.__version__.split('+')[0],generated_without_fitting=True,
                      entries=entries,files=files)
        (stage/'manifest.json').write_bytes(canonical(manifest))
        stage.rename(destination)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def accept(directory):
    """Restore saved tensors exactly, independently of host initialization arithmetic."""
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    expected={f'{arm}-{seed}' for arm in protocol()['arms'] for seed in protocol()['confirmation_seeds']}
    if (manifest['schema']!='recurrent-comparison-starts-v1'
            or manifest['protocol_sha256']!=PROTOCOL_SHA256
            or manifest['torch_version']!=torch.__version__.split('+')[0]
            or manifest['generated_without_fitting'] is not True
            or set(manifest['entries'])!=expected):
        raise ValueError('fixed unfitted starts manifest required')
    actual={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file() and p!=directory/'manifest.json'}
    if actual!=manifest['files']:
        raise ValueError('complete initialization checksum inventory required')
    with preserved_rng():
        for name,entry in manifest['entries'].items():
            if name!=f"{entry['arm']}-{entry['seed']}":
                raise ValueError('initialization role/seed identity mismatch')
            asset=entry['asset']
            expected_file=name if entry['arm']=='gru_sac' else name+'.npz'
            if asset['file']!=expected_file:
                raise ValueError('fixed initialization asset path required')
            model=construct(entry['arm'],entry['seed'])
            if entry['arm']=='gru_sac':
                restored,_=load_checkpoint(directory/name,expected_lineage=lineage(),restore_torch_rng=True)
                validate_learner(restored)
                if restored.updates!=0 or restored.log_alpha.item()!=0:
                    raise ValueError('unfitted GRU start required')
            else:
                load_initialization(model,directory/asset['file'],asset['tensors'])
                _untrained(model)
            # Verify the saved global states describe the constructor's actual streams,
            # before accepting restoration of those streams.
            if capture_global_rng()!=entry['global_rng'] or rng_record()!=entry['rng_hashes']:
                raise ValueError('seeded initialization stream identity mismatch')
            restore_global_rng(entry['global_rng'])
    return dict(accepted=True,cells=len(expected),protocol_sha256=PROTOCOL_SHA256,
                training_performed=False,performance_interpretation_allowed=False)


def load_start(directory,arm,seed,*,env=None):
    """Load one already accepted matrix cell, restoring its registered streams."""
    accept(directory)  # all validation before the returned learner can be used
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    model=construct(arm,seed,env=env)
    entry=manifest['entries'][f'{arm}-{seed}']
    if arm=='gru_sac':
        model,_=load_checkpoint(directory/entry['asset']['file'],expected_lineage=lineage(),
                                restore_torch_rng=True)
    else:
        load_initialization(model,directory/entry['asset']['file'],entry['asset']['tensors'])
    restore_global_rng(entry['global_rng'])
    return model
