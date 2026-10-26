"""Create six unfitted full64 SAC/CrossQ starts; no training or data access."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil

import numpy as np
from sb3_contrib import CrossQ
from stable_baselines3 import SAC
import torch

from rl_stocks.core.files import write_once
from rl_stocks.core.provenance import repository_state
from rl_stocks.training.crossq_archives import snapshot_sources, verify_arrays
from rl_stocks.training.crossq_inspection import SpacesOnly
from rl_stocks.training.crossq_mapping import map_linears
from rl_stocks.training.crossq_spec import ROOT, PROTOCOL_SHA256, parameters, protocol
from rl_stocks.training.distributional_assets import initialization_entry, manifest as original_manifest
from rl_stocks.training.distributional_critics import restore_global_rng, rng_record
from rl_stocks.training.shared_initialization import load_initialization, read_initialization, write_initialization


def generate(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to((ROOT / 'configs/initializations').resolve()):
        raise ValueError('new directory beneath repository configs/initializations required')
    state = repository_state(ROOT)
    if state['dirty']:
        raise ValueError('commit generator before creating immutable initialization evidence')
    spec = protocol()
    old = original_manifest()
    binding = ROOT / spec['shared_initialization']['source_binding']
    if hashlib.sha256(binding.read_bytes()).hexdigest() != spec['shared_initialization']['source_binding_sha256']:
        raise ValueError('fixed unfitted original SAC binding required')
    # Snapshot pinned original code while it still matches, before registry edits.
    output.mkdir(parents=True)
    snapshots = snapshot_sources(old['implementation_sha256'], output / 'source-provenance')
    numpy_state, python_state = np.random.get_state(), random.getstate()
    entries, groups = {}, {}
    with torch.random.fork_rng(devices=[]):
        try:
            for seed in spec['confirmation_seeds']:
                source, globals_record = initialization_entry('sac_native', seed, False)
                source_path = ROOT / source['file']
                source_bytes = source_path.read_bytes()
                source_arrays, native_rng = read_initialization(source_bytes, source)
                env = SpacesOnly(dict(observation_size=21, action_size=4))
                sac = SAC('MlpPolicy', env, seed=seed, device='cpu', **parameters('sac'))
                load_initialization(sac, source_path, source)
                crossq = CrossQ('MlpPolicy', env, seed=seed, device='cpu', **parameters('crossq'))
                mapping = map_linears(sac, crossq)
                torch.set_rng_state(torch.from_numpy(native_rng))
                restore_global_rng(globals_record)
                if rng_record() != source['global_rng_sha256']:
                    raise ValueError('original SAC Torch/NumPy/Python streams did not restore')
                sac_path = output / f'sac-seed{seed}.npz'
                shutil.copyfile(source_path, sac_path)
                if sac_path.read_bytes() != source_bytes:
                    raise ValueError('original SAC archive bytes changed')
                entries[f'sac-{seed}'] = dict(source, file=str(sac_path.relative_to(ROOT)),
                                             arm='sac', source_file=source['file'], phase_independent=True)
                path = output / f'crossq-seed{seed}.npz'
                cross_entry = write_initialization(crossq, path)
                arrays, rng = read_initialization(path.read_bytes(), cross_entry)
                if not np.array_equal(rng, native_rng):
                    raise ValueError('CrossQ archive did not preserve original SAC CPU RNG')
                arrays_check = verify_arrays(source_arrays, arrays, mapping)
                if rng_record() != source['global_rng_sha256']:
                    raise ValueError('writing mapped archives changed sampling streams')
                entries[f'crossq-{seed}'] = dict(cross_entry, file=str(path.relative_to(ROOT)),
                                                arm='crossq', seed=seed, smoke=False,
                                                phase_independent=True, global_rng_sha256=source['global_rng_sha256'])
                groups[str(seed)] = dict(global_rng=globals_record, mapping=mapping,
                                          array_verification=arrays_check,
                                          original_sac_archive_sha256=source['sha256'])
                env.close()
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
    code = ['scripts/build_crossq_initializations.py', 'src/rl_stocks/training/crossq_archives.py',
            'src/rl_stocks/training/crossq_mapping.py', 'src/rl_stocks/training/crossq_spec.py']
    result = dict(schema='crossq-shared-initializations-v1', protocol_sha256=PROTOCOL_SHA256,
                  generated_without_fitting=True, no_dataset_or_provider_access=True,
                  optimizer_steps=0, transitions=0, smoke_reuses_full64=True,
                  initial_entropy_coefficient=1.0, artifacts=entries, groups=groups,
                  original_generation_source_snapshots=snapshots, git=state,
                  python=platform.python_version(), torch=torch.__version__,
                  implementation_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in code})
    write_once(output/'manifest.json', (json.dumps(result, indent=2)+'\n').encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.output)
    print(f"Generated{len(result['artifacts'])} unfitted starts; common Linear/RNG proof for11/23/37; no learning/data/Colab.")


if __name__ == '__main__':
    main()
