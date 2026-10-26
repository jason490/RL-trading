"""Pinned portable untrained SAC/CrossQ archives; no model construction."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rl_stocks.core.contracts import digest
from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.training.crossq_archives import verify_arrays
from rl_stocks.training.crossq_spec import ROOT, PROTOCOL_SHA256, protocol
from rl_stocks.training.distributional_critics import parsed_global_rng
from rl_stocks.training.shared_initialization import read_initialization

BINDING_SHA256 = '84452236ccf86e66e6527660b89d98da8030864f8390f335e6b875f7d36f42bb'


def portable_path(name):
    relative = Path(name)
    resolved = (ROOT / relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not resolved.is_relative_to((ROOT / 'configs/initializations').resolve()):
        raise ValueError('portable repository initialization path required')
    return resolved


def manifest():
    binding = json.loads((ROOT / 'configs/experiments/crossq_initialization_binding.json').read_text())
    if digest(binding) != BINDING_SHA256 or binding['protocol_sha256'] != PROTOCOL_SHA256:
        raise ValueError('fixed SAC/CrossQ initialization binding required')
    path = portable_path(binding['manifest'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != binding['manifest_sha256']:
        raise ValueError('fixed SAC/CrossQ initialization manifest checksum required')
    value = json.loads(path.read_text())
    if (value['protocol_sha256'] != PROTOCOL_SHA256 or value['generated_without_fitting'] is not True
        or value['no_dataset_or_provider_access'] is not True or value['optimizer_steps'] != 0
        or value['transitions'] != 0 or value['smoke_reuses_full64'] is not True
        or set(value['artifacts']) != {f'{a}-{s}' for a in ['sac', 'crossq'] for s in [11, 23, 37]}
        or set(value['groups']) != {'11', '23', '37'}):
        raise ValueError('complete unfitted phase-independent CrossQ starts required')
    return value


def initialization_entry(arm, seed=11, smoke=False):
    if (arm not in protocol()['arms'] or type(seed) is not int or seed not in (11, 23, 37)
        or type(smoke) is not bool or smoke and seed != 11):
        raise ValueError('fixed SAC/CrossQ arm/seed/phase required')
    value = manifest()
    entry = value['artifacts'][f'{arm}-{seed}']
    if (entry['arm'] != arm or type(entry['seed']) is not int or entry['seed'] != seed
        or entry['smoke'] is not False or entry['phase_independent'] is not True):
        raise ValueError('original full64 source entry reused for both phases required')
    portable_path(entry['file'])
    return entry, value['groups'][str(seed)]['global_rng']


def initialization_spec(arm, seed=11, smoke=False):
    entry, _ = initialization_entry(arm, seed, smoke)
    return dict(version=1, artifact_file=entry['file'], artifact_sha256=entry['sha256'],
                policy_tensor_sha256=entry['policy_tensor_sha256'], rng_sha256=entry['rng_sha256'],
                global_rng_sha256=entry['global_rng_sha256'], torch_version=entry['torch_version'],
                generated_without_fitting=True, arm=arm, seed=seed, smoke=smoke,
                protocol_sha256=PROTOCOL_SHA256, binding_sha256=BINDING_SHA256)


def validate_assets():
    value = manifest()
    spec = protocol()
    source_binding = ROOT / spec['shared_initialization']['source_binding']
    if hashlib.sha256(source_binding.read_bytes()).hexdigest() != spec['shared_initialization']['source_binding_sha256']:
        raise ValueError('original unfitted SAC binding changed')
    old_binding = json.loads(source_binding.read_text())
    old_path = portable_path(old_binding['manifest'])
    if hashlib.sha256(old_path.read_bytes()).hexdigest() != old_binding['manifest_sha256']:
        raise ValueError('original unfitted SAC manifest changed')
    old = json.loads(old_path.read_text())
    snapshots = value['original_generation_source_snapshots']
    if set(snapshots) != set(old['implementation_sha256']):
        raise ValueError('complete portable original-generation provenance required')
    for name, entry in snapshots.items():
        if hashlib.sha256(portable_path(entry['file']).read_bytes()).hexdigest() != entry['sha256'] or entry['sha256'] != old['implementation_sha256'][name]:
            raise ValueError('original generation snapshot hash mismatch')
    records = {}
    for seed in (11, 23, 37):
        original = old['artifacts'][f'sac_native-screen-{seed}']
        source_bytes = portable_path(original['file']).read_bytes()
        arrays, rng = read_initialization(source_bytes, original)
        sac, _ = initialization_entry('sac', seed)
        if portable_path(sac['file']).read_bytes() != source_bytes:
            raise ValueError('original SAC bytes must remain exact')
        read_initialization(portable_path(sac['file']).read_bytes(), sac)
        candidate, globals_record = initialization_entry('crossq', seed)
        target, target_rng = read_initialization(portable_path(candidate['file']).read_bytes(), candidate)
        if not np.array_equal(rng, target_rng):
            raise ValueError('shared original SAC CPU RNG required')
        torch.Generator(device='cpu').set_state(torch.from_numpy(rng))
        numpy_state, python_state = parsed_global_rng(globals_record)
        actual = dict(torch=hashlib.sha256(rng.tobytes()).hexdigest(), numpy=fingerprint(numpy_state), python=fingerprint(python_state))
        if (actual != original['global_rng_sha256'] or actual != candidate['global_rng_sha256']
            or actual != sac['global_rng_sha256'] or globals_record != old['groups'][f'screen-{seed}']['global_rng']):
            raise ValueError('all original Torch/NumPy/Python streams required')
        proof = verify_arrays(arrays, target, value['groups'][str(seed)]['mapping'])
        if proof != value['groups'][str(seed)]['array_verification']:
            raise ValueError('archived common Linear proof changed')
        records[str(seed)] = proof
    return dict(passed=True, records=records, portable_source_snapshots_verified=len(snapshots),
                smoke_reuses_full64=True, models_constructed=False, optimizer_steps=0)
