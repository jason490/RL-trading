"""Checksummed learner snapshots; replay/environment continuation is separate."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import tempfile

import torch

from rl_stocks.rl.recurrent_sac.networks import RecurrentSACNetworks
from rl_stocks.training.recurrent_replay import _integer
from rl_stocks.training.recurrent_sac_learner import RecurrentSACLearner

SCHEMA = 'recurrent-sac-checkpoint-v1'


def _finite_tree(value):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite checkpoint tensor')
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item)


def validate_learner(learner):
    _integer('updates', learner.updates)
    if learner.failed or learner.updates != learner.attempts or learner.updates > learner.max_updates:
        raise ValueError('partial/failed learner cannot be checkpointed')
    if learner.completed_steps != dict.fromkeys(('actor', 'critic', 'entropy'), learner.updates):
        raise ValueError('completed optimizer counts mismatch')
    _finite_tree(learner.networks.state_dict())
    _finite_tree(learner.log_alpha)
    for role, opt in learner.optimizers.items():
        _finite_tree(opt.state_dict())
        parameters = [p for g in opt.param_groups for p in g['params']]
        if learner.updates == 0:
            if opt.state:
                raise ValueError('initial checkpoint requires empty optimizer state')
        else:
            if len(opt.state) != len(parameters):
                raise ValueError('every parameter requires actual Adam state')
            for p in parameters:
                state = opt.state[p]
                if (set(state) != {'step', 'exp_avg', 'exp_avg_sq'}
                        or state['step'].numel() != 1 or state['step'].item() != learner.updates
                        or state['exp_avg'].shape != p.shape or state['exp_avg_sq'].shape != p.shape):
                    raise ValueError(f'actual {role} Adam state/counter mismatch')


def save_checkpoint(learner, destination: Path, *, lineage: dict):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('immutable checkpoint destination exists')
    validate_learner(learner)
    # Require JSON-only public lineage; caller must supply source/data/schema identities.
    clean_lineage = json.loads(json.dumps(lineage, allow_nan=False))
    if (type(clean_lineage) is not dict or any(type(clean_lineage.get(k)) is not str
            or not clean_lineage[k] for k in ('source', 'dataset', 'observation_schema'))):
        raise ValueError('source, dataset and observation schema lineage required')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.recurrent-sac-', dir=destination.parent))
    try:
        value = dict(networks=learner.networks.state_dict(), log_alpha=learner.log_alpha.detach(),
                     optimizers={k: v.state_dict() for k, v in learner.optimizers.items()},
                     torch_rng=torch.get_rng_state())
        torch.save(value, stage/'state.pt')
        manifest = dict(schema=SCHEMA, observation_size=learner.networks.observation_size,
                        hidden_size=learner.networks.hidden_size, max_updates=learner.max_updates,
                        learning_rate=learner.learning_rate, gamma=learner.gamma, tau=learner.tau,
                        updates=learner.updates, completed_steps=learner.completed_steps,
                        lineage=clean_lineage,
                        sha256=hashlib.sha256((stage/'state.pt').read_bytes()).hexdigest())
        (stage/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_checkpoint(source: Path, *, expected_lineage: dict, restore_torch_rng=False):
    if (type(expected_lineage) is not dict or any(type(expected_lineage.get(k)) is not str
            or not expected_lineage[k] for k in ('source', 'dataset', 'observation_schema'))):
        raise ValueError('expected source/data/schema lineage required')
    source = Path(source)
    manifest = json.loads((source/'manifest.json').read_text())
    if (manifest['schema'] != SCHEMA or manifest['lineage'] != expected_lineage
            or type(restore_torch_rng) is not bool):
        raise ValueError('checkpoint schema/lineage mismatch')
    raw = (source/'state.pt').read_bytes()
    if len(raw) > 32*1024*1024 or hashlib.sha256(raw).hexdigest() != manifest['sha256']:
        raise ValueError('checkpoint size/checksum mismatch')
    _integer('updates', manifest['updates'])
    # Constructing a verifier/load target must not perturb caller RNG by default.
    with torch.random.fork_rng():
        networks = RecurrentSACNetworks(manifest['observation_size'], manifest['hidden_size'])
        learner = RecurrentSACLearner(networks, **{k: manifest[k] for k in
                  ('max_updates', 'learning_rate', 'gamma', 'tau')})
        value = torch.load(BytesIO(raw), weights_only=True, map_location='cpu')
        networks.load_state_dict(value['networks'], strict=True)
        alpha = value['log_alpha']
        if alpha.shape != learner.log_alpha.shape or alpha.dtype != torch.float32:
            raise ValueError('raw entropy parameter geometry mismatch')
        learner.log_alpha.data.copy_(alpha)
        for role, optimizer in learner.optimizers.items():
            state = value['optimizers'][role]
            if state['param_groups'] != optimizer.state_dict()['param_groups']:
                raise ValueError('saved optimizer configuration mismatch')
            optimizer.load_state_dict(state)
        learner.updates = learner.attempts = manifest['updates']
        learner.completed_steps = deepcopy(manifest['completed_steps'])
        validate_learner(learner)
        generator = torch.Generator()
        generator.set_state(value['torch_rng'])
    if restore_torch_rng:
        torch.set_rng_state(value['torch_rng'])
    return learner, manifest
