"""Inference-only acceptance on the pinned real dataset before remote optimization."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
import torch

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.evaluation.offpolicy_fit import fingerprint
from rl_stocks.inference.task_bundle import FrozenTaskPolicy, freeze_task
from rl_stocks.risk.exposure import execution_contract, filter_exposure
from rl_stocks.rl.sac.task_factory import create_task_sac
from rl_stocks.training.task_conditioning import protocol, validate_type_manifest
from rl_stocks.training.task_environment import build_task_environment


@contextmanager
def prohibit_optimization():
    def forbidden(*args, **kwargs):
        raise RuntimeError('local optimization prohibited')
    targets = [(SAC, 'learn'), (SAC, 'train'), (torch.optim.Adam, 'step'),
               (torch.Tensor, 'backward')]
    original = [(owner, name, getattr(owner, name)) for owner, name in targets]
    try:
        for owner, name in targets:
            setattr(owner, name, forbidden)
        yield
    finally:
        for owner, name, value in original:
            setattr(owner, name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh acceptance output required')
    spec = protocol()
    data = read_dataset(Path('data/datasets') / spec['dataset'])
    evidence = json.loads(Path('configs/universes/task_types_research_initial_v1.json').read_text())
    evidence_sha = validate_type_manifest(data, evidence)
    before = digest(data)
    if before != spec['dataset']:
        raise ValueError('pinned dataset required')
    args.output.mkdir(parents=True)
    results = []
    with prohibit_optimization(), torch.no_grad():
        torch.set_num_threads(1)
        for role in ('masked_shared', 'tagged_shared', 'tagged_routed'):
            env = build_task_environment(data, evidence, task_role=role, smoke=True)
            try:
                model = create_task_sac(env, task_role=role, smoke=True)
                state = fingerprint(model.policy.state_dict())
                normalizer = deepcopy(data['normalizer'])
                target = args.output / role
                target.mkdir()
                model.save(target / 'initial.zip')
                bundle = freeze_task(target / 'initial.zip', data, task_role=role,
                    type_manifest=evidence,
                    metadata={'algorithm': 'sac', 'observation': env.unwrapped.observation_spec,
                              'action_execution': execution_contract('unrestricted', 'unit_magnitude')},
                    output=target / 'frozen')
                frozen = FrozenTaskPolicy(bundle, feature_schema=data['schema'],
                    feed=data['manifest']['provenance']['capabilities']['feed'])
                observation, _ = env.reset(seed=11)
                if observation.shape != (22,) or not np.isfinite(observation).all():
                    raise ValueError('real reset observation rejected')
                # Exercise real normalized market/account values under each reviewed type.
                base = observation[:21].copy()
                for record in evidence['records']:
                    tag = 0 if role == 'masked_shared' else (1 if record['kind'] == 'etf' else -1)
                    expected = filter_exposure(model.predict(np.append(base, tag), deterministic=True)[0],
                                               'unit_magnitude')
                    actual = frozen.predict(base, symbol=record['symbol'], session=record['known_from'])
                    if not np.array_equal(expected, actual):
                        raise ValueError('real-data frozen parity failed')
                if (fingerprint(model.policy.state_dict()) != state
                        or data['normalizer'] != normalizer or model.num_timesteps != 0):
                    raise ValueError('inference mutated state or performed optimization')
                results.append({'role': role, 'symbols_checked': len(evidence['records']),
                                'exact_frozen_parity': True, 'learned_transitions': 0,
                                'policy_unchanged': True, 'normalizer_unchanged': True,
                                'parameters': sum(p.numel() for p in model.policy.parameters()),
                                'bundle': str(bundle)})
            finally:
                env.close()
    if digest(data) != before:
        raise ValueError('dataset mutated')
    receipt = {'dataset': before, 'type_manifest_sha256': evidence_sha,
               'roles': results, 'local_optimization_prohibited': True,
               'training_performed': False, 'development_evaluated': False,
               'test_evaluated': False, 'colab_allocated': False}
    (args.output / 'acceptance.json').write_bytes(canonical(receipt))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
