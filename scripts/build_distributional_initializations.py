"""Generate native untrained SAC/TQC tensors and common native SAC sampling RNG."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
import stable_baselines3
import sb3_contrib
from gymnasium import Env, spaces

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.core.provenance import repository_state
from rl_stocks.rl.registry.factory import create
from rl_stocks.training.distributional_critics import (
    PROTOCOL_SHA256, ROOT, capture_global_rng, model_record, overrides, protocol,
    restore_global_rng, validate_initial_comparison,
)
from rl_stocks.training.shared_initialization import read_initialization, write_initialization


class ModelSpaces(Env):
    observation_space = spaces.Box(-np.inf, np.inf, (21,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        raise ValueError('new repository-local initialization directory required')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    artifacts, groups = {}, {}
    spec = protocol()
    for smoke, seed in [(True, 11), (False, 11), (False, 23), (False, 37)]:
        key = ('smoke' if smoke else 'screen') + '-' + str(seed)
        baseline = create('sac', ModelSpaces(), seed=seed, device='cpu', smoke=smoke)
        native_record = model_record(baseline)
        native_rng = torch.get_rng_state().clone()
        globals_record = capture_global_rng()
        records = {}
        for arm, arm_spec in spec['arms'].items():
            model = create(arm_spec['algorithm'], ModelSpaces(), seed=seed, device='cpu',
                           smoke=smoke, overrides=overrides(arm, smoke))
            if arm == 'sac_native' and canonical(model_record(model)) != canonical(native_record):
                raise ValueError('study overrides changed native standard SAC initialization/RNG')
            torch.set_rng_state(native_rng)
            restore_global_rng(globals_record)
            record = model_record(model)
            path = output / f'{arm}-{key}.npz'
            artifact = write_initialization(model, path)
            entry = dict(file=str(path.relative_to(ROOT)), seed=seed, smoke=smoke, arm=arm,
                         global_rng_sha256=record['rng'], **artifact)
            read_initialization(path.read_bytes(), entry)
            if model_record(model) != record:
                raise ValueError('writing initialization mutated native tensors/optimizer/RNG')
            artifacts[arm + '-' + key] = entry
            records[arm] = record
        proof = validate_initial_comparison(records, smoke)
        groups[key] = dict(seed=seed, smoke=smoke, global_rng=globals_record,
                           actual_initial_records=records, comparison=proof,
                           native_standard_initialization_unchanged=True)
        print('Verified native actors, separate critic tensors and shared RNG:', key, flush=True)
    code = ['scripts/build_distributional_initializations.py',
            'src/rl_stocks/training/distributional_critics.py',
            'src/rl_stocks/training/shared_initialization.py',
            'src/rl_stocks/rl/registry/factory.py', 'src/rl_stocks/rl/tqc/config.py',
            'src/rl_stocks/rl/tqc/geometry.py']
    result = dict(schema='distributional-critic-initializations-v1', protocol_sha256=PROTOCOL_SHA256,
                  artifacts=artifacts, groups=groups, generated_without_fitting=True,
                  no_dataset_or_provider_access=True, optimizer_steps=0,
                  git=repository_state(ROOT),
                  versions=dict(python=platform.python_version(), torch=torch.__version__,
                                numpy=np.__version__, sb3=stable_baselines3.__version__,
                                sb3_contrib=sb3_contrib.__version__),
                  implementation_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in code})
    write_once(output / 'manifest.json', (json.dumps(result, indent=2) + '\n').encode())
    print('Generated12 untrained tensor artifacts; zero optimizer steps and no market-data access.')


if __name__ == '__main__':
    main()
